// Range and radial velocity consistent with two measured endpoint directions.
// Additional exposures carve out regions in these two coordinates.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <vector>

namespace returns {
using Clock = std::chrono::steady_clock;
constexpr double rad = 4.84813681109536e-6;
constexpr double mu = .0002959122082855911*1.0013418;
inline double sq(double x) { return x*x; }
struct V {
    double x,y;
    V operator+(V b) const {return {x+b.x,y+b.y};}
    V operator-(V b) const {return {x-b.x,y-b.y};}
    V operator*(double s) const {return {x*s,y*s};}
    double dot(V b) const {return x*b.x+y*b.y;}
    double norm2() const {return dot(*this);}
};
struct Target {float x,y,m; int e;};
struct Exposure {double t,ox,oy,oz; int night,pad;};
struct Point {float x,y,m,q; int id;};
struct Pair {int a,b;float g,sg,h,sh,score;};
struct Seed {int rows[5];float g,score;};
struct Triple {int rows[3];float g,sg,h,sh,score;};
struct RawCore {double x[6],ref,merit;int count,rows[12],pad;};
static_assert(sizeof(RawCore)==120 && sizeof(Triple)==32 && sizeof(Seed)==28,"orbit catalogue layout");
struct Midpoint {double mag,color,var,cvar,cov;};
struct Normal {
    double aa=1.,ab=0.,bb=1.,ax=0.,bx=0.,rr=0.;
    void add(double x,double y,double z,double w) {
        aa+=x*x*w;ab+=x*y*w;bb+=y*y*w;ax+=x*z*w;bx+=y*z*w;rr+=z*z*w;
    }
    double fit(double &u,double &v) const {
        double d=aa*bb-ab*ab;
        if(!(d>0.))return 1.e9;
        u=(ax*bb-bx*ab)/d;v=(bx*aa-ax*ab)/d;
        return std::max(0.,rr-u*ax-v*bx);
    }
};

struct Image {
    int bx=0,by=0,nx=0,ny=0;
    double cell=0.,xmin=0.,xmax=0.,ymin=0.,ymax=0.;
    std::vector<int> edges;
    std::vector<Point> pts;
    std::array<double,55> density{};
    void build(const std::vector<Point>& data,double arc_cell) {
        if(data.empty())return;
        cell=arc_cell; xmin=xmax=data[0].x;ymin=ymax=data[0].y;
        for(const auto &p:data){
            xmin=std::min(xmin,double(p.x));xmax=std::max(xmax,double(p.x));
            ymin=std::min(ymin,double(p.y));ymax=std::max(ymax,double(p.y));
        }
        cell=std::max(cell,std::sqrt((xmax-xmin)*(ymax-ymin)/50000.));
        bx=std::floor(xmin/cell);by=std::floor(ymin/cell);
        nx=int(std::floor(xmax/cell))-bx+1;ny=int(std::floor(ymax/cell))-by+1;
        edges.resize(size_t(nx)*ny+1);
        auto key=[&](const Point&p){return int(std::floor(p.x/cell))-bx+nx*(int(std::floor(p.y/cell))-by);};
        std::array<int,55> hist{};
        for(const auto &p:data){
            edges[key(p)+1]++;
            ++hist[std::clamp(int(std::floor((p.m-15.)*4.)),0,54)];
        }
        int live=std::count_if(edges.begin(),edges.end(),[](int n){return n!=0;});
        double area=std::max(10.*cell*cell,std::min((xmax-xmin+cell)*(ymax-ymin+cell),
                                    std::max(6.e5,live*cell*cell*1.8)));
        for(int i=0;i<55;i++){
            density[i]=(1.+hist[i]+hist[std::max(0,i-1)]+hist[std::min(i+1,54)])/area;
        }
        std::partial_sum(edges.begin(),edges.end(),edges.begin());
        pts.resize(data.size());auto cursor=edges;
        for(const auto &p:data)pts[cursor[key(p)]++]=p;
    }
    bool touches(V p,double wx,double wy) const {
        return nx && p.x+wx>=xmin && p.x-wx<=xmax && p.y+wy>=ymin && p.y-wy<=ymax;
    }
    template<class F> void ellipse(V center,double xx,double xy,double yy,double z,F fn) const {
        double wx=z*std::sqrt(xx),wy=z*std::sqrt(yy);
        if(!touches(center,wx,wy))return;
        double var=std::max(.0001,xx-xy*xy/yy);
        int y0=std::max(0,int(std::floor((center.y-wy)/cell))-by);
        int y1=std::min(ny-1,int(std::floor((center.y+wy)/cell))-by);
        for(int iy=y0;iy<=y1;iy++){
            double dy0=std::max(-wy,(iy+by)*cell-center.y);
            double dy1=std::min(wy,(iy+by+1)*cell-center.y);
            double mdy=std::max({0.,dy0,-dy1});
            double dx=std::sqrt(var*std::max(0.,z*z-mdy*mdy/yy));
            double x0=center.x+std::min(dy0*xy/yy,dy1*xy/yy)-dx;
            double x1=center.x+std::max(dy0*xy/yy,dy1*xy/yy)+dx;
            int lo=std::max(0,int(std::floor(x0/cell))-bx),hi=std::min(nx-1,int(std::floor(x1/cell))-bx);
            if(lo>hi)continue;
            for(int k=edges[iy*nx+lo];k<edges[iy*nx+hi+1];k++){
                const auto &p=pts[k];
                double dy=p.y-center.y,dx=p.x-center.x-dy*xy/yy;
                double chi=dy*dy/yy+dx*dx/var;
                if(chi<=z*z)fn(p,chi);
            }
        }
    }
};

// Lagrange f,g expansion, including longitudinal gravity and its time
// derivative. Times here are relative to the light-emission epoch of the
// first measurement. The subsequent full fit uses retarded observations.
struct State {
    V p,v;
    double g,h,k,rd,beta,home_distance;
    void rates(){
        double l=1.+p.norm2();
        k=mu*g*g*g/(l*std::sqrt(l));
        rd=(h+p.dot(v))/l;
        beta=(h*h+v.norm2())/l;
    }
    void fg(double t,double &f,double &s) const {
        double z=k*t*t, w=rd*t, b=beta*t*t;
        f=1.-.5*z+.5*z*w+z*(3.*b-2.*z-15.*w*w)/24.
             +z*w*(30.*z-45.*b+105.*w*w)/120.;
        s=t*(1.-z/6.+z*w/4.+z*(9.*b-8.*z-45.*w*w)/120.
             +z*w*(150.*z-180.*b+420.*w*w)/720.);
    }
    State(const Target&a,const Target&b,const Exposure&A,const Exposure&B,double gi,double hi):g(gi),h(hi){
        p={a.x*(1.-g*A.oz)+g*A.ox,a.y*(1.-g*A.oz)+g*A.oy};
        home_distance=(1.-g*A.oz)*std::sqrt(1.+sq(a.x)+sq(a.y))/g;
        double t=B.t-A.t,f=1.,s=t;
        for(int it=0;it<3;it++){
            double z=f+s*h-g*B.oz;
            v={(b.x*z+g*B.ox-f*p.x)/s,(b.y*z+g*B.oy-f*p.y)/s};
            rates();
            t=B.t-A.t-(z*std::sqrt(1.+sq(b.x)+sq(b.y))/g-home_distance)/173.1446327;
            fg(t,f,s);
        }
    }
    V position(double t,const Exposure&e) const {
        double f,s;fg(t,f,s);
        double x=f*p.x+s*v.x-g*e.ox,y=f*p.y+s*v.y-g*e.oy,z=f+s*h-g*e.oz;
        t-=(std::sqrt(x*x+y*y+z*z)/g-home_distance)/173.1446327;
        fg(t,f,s);
        double scale=1./((f+s*h-g*e.oz)*rad);
        return {(f*p.x+s*v.x-g*e.ox)*scale,(f*p.y+s*v.y-g*e.oy)*scale};
    }
};

struct Bank {
    const Target*d;
    const Exposure*ex;
    const float *sn,*quality;
    const int *peer;
    const double *depth,*lever;
    std::vector<Image> fields,ends;
    int ne;
    V east;
    Bank(const Target*cat,int n,const Exposure*epochs,int count,const float*s,const float*q,
         const int*peers,const double*md,const double*c,const int *endpoint,int nept,V direction):
         d(cat),ex(epochs),sn(s),quality(q),peer(peers),depth(md),lever(c),fields(count),ends(count),
         ne(count),east(direction){
        std::vector<std::vector<Point>> bins(count),ept(count);
        auto point=[&](int i){return Point{float(d[i].x/rad),float(d[i].y/rad),d[i].m,quality[i],i};};
        for(int i=0;i<n;i++)if(std::isfinite(d[i].m)&&std::isfinite(d[i].x+d[i].y))
            bins[d[i].e].push_back(point(i));
        for(int k=0;k<nept;k++){
            int i=endpoint[k];
            if(std::isfinite(d[i].m)&&std::isfinite(d[i].x+d[i].y))ept[d[i].e].push_back(point(i));
        }
        for(int j=0;j<ne;j++){fields[j].build(bins[j],55.);ends[j].build(ept[j],105.);}
    }
    double noise(int i) const {return std::hypot(.105,1.03/std::max(4.5f,sn[i]));}
    Midpoint phot(int a,int b=-1) const {
        double sw=.0001,sl=0.,sll=1./sq(.24),sm=.0001*d[a].m,slm=0.;
        std::array<int,4> ids={a,peer[a],b,b<0?-1:peer[b]};
        for(int j=0;j<4;j++){
            int i=ids[j];if(i<0 || std::find(ids.begin(),ids.begin()+j,i)!=ids.begin()+j)continue;
            double l=lever[d[i].e],m=d[i].m;
            double w=1./(.13*.13+sq(1.086/std::max(4.5f,sn[i])));
            sw+=w;sl+=w*l;sll+=w*l*l;sm+=w*m;slm+=w*l*m;
        }
        double det=sw*sll-sl*sl;
        return {(sm*sll-slm*sl)/det,(sw*slm-sl*sm)/det,sll/det,sw/det,-sl/det};
    }
    double phot_chi(const Midpoint &m,int j) const{
        double l=lever[d[j].e],v=m.var+2.*l*m.cov+l*l*m.cvar+
                            sq(.18)+sq(1.086/std::max(4.5f,sn[j]));
        return sq(d[j].m-m.mag-m.color*l)/v;
    }
};
template<class T> void best(std::vector<T>&v,size_t cap){
    if(v.size()<=cap)return;
    std::nth_element(v.begin(),v.begin()+cap,v.end(),[](const T&a,const T&b){return a.score>b.score;});
    v.resize(cap);
}

// Marginalize a displacement over a local bound velocity distribution.
// The fit uses centroid differences inside either endpoint night as well.
struct Prior {
    const Bank &bank;int a;
    double g,h,sg,sh,k;V p,east,north,meanv;
    double vs,ws;
    Prior(const Bank&data,int anchor,double r,double width,double av,double bv):
        bank(data),a(anchor),east(data.east),north{-east.y,east.x},vs(av),ws(bv){
        const auto &d=bank.d[a];
        p={d.x,d.y};
        double l=std::sqrt(1.+p.norm2());
        g=l/r;sg=g*width;k=.01720209895/std::pow(r,1.5);
        meanv=east*(k*l);h=-p.dot(meanv)/(l*l);sh=k*(width+.035);
    }
    void mean(int e,V &pos,V &u,double &along,double &cross) const {
        const auto &A=bank.ex[bank.d[a].e],&B=bank.ex[e];double t=B.t-A.t;
        V o={B.ox-A.ox,B.oy-A.oy};o=o-p*(B.oz-A.oz);
        double den=1.+h*t-g*B.oz;
        V m=(meanv*t-o*g)*(1./den);pos=(p+m)*(1./rad);
        u=(meanv*(1.5*t/g)-o+m*B.oz)*(sg/(den*rad));
        along=vs*k*t/rad;cross=ws*k*t/rad;
    }
    Pair posterior(int b,double chi0) const {
        const auto &A=bank.ex[bank.d[a].e],&B=bank.ex[bank.d[b].e];
        double dt=B.t-A.t;V bxy={bank.d[b].x,bank.d[b].y};
        double g0=g,h0=h;Normal last;
        for(int it=0;it<2;it++){
            State s(bank.d[a],bank.d[b],A,B,g0,h0);
            State dg(bank.d[a],bank.d[b],A,B,g0+.015*sg,h0);
            State dh(bank.d[a],bank.d[b],A,B,g0,h0+.015*sh);
            Normal f;f.ax=(g-g0)/sg;f.bx=(h-h0)/sh;
            f.rr=sq(f.ax)+sq(f.bx);
            auto trans=[&](const State&st){
                return (st.v-st.p*st.h)*(1./(.01720209895*std::pow(st.g/std::sqrt(1.+st.p.norm2()),1.5)));
            };
            V v=trans(s),vg=(trans(dg)-v)*(1./.015),vh=(trans(dh)-v)*(1./.015);
            double cen=std::hypot(bank.noise(a),bank.noise(b))*rad/(k*std::abs(dt));
            f.add(vg.dot(east),vh.dot(east),1.-v.dot(east),1./(vs*vs+cen*cen));
            f.add(vg.dot(north),vh.dot(north),-v.dot(north),1./(ws*ws+cen*cen));
            for(int row:{a,b}){
                int pe=bank.peer[row];if(pe<0)continue;
                const auto &e=bank.ex[bank.d[pe].e];double t=e.t-A.t;
                V c=s.position(t,e),x=(dg.position(t,e)-c)*(1./.015),y=(dh.position(t,e)-c)*(1./.015);
                V delta={bank.d[pe].x/rad-c.x,bank.d[pe].y/rad-c.y};
                double w=1./(sq(bank.noise(row))+sq(bank.noise(pe))+.065*.065);
                f.add(x.x,y.x,delta.x,w);f.add(x.y,y.y,delta.y,w);
            }
            double u=0.,v2=0.;double chi=f.fit(u,v2);
            g0+=sg*std::clamp(u,-2.5,2.5);h0+=sh*std::clamp(v2,-2.5,2.5);
            if(!(g0>.0001&&g0<1./14.)||chi>30.)return {a,b,0,0,0,0,-1.e10f};
            last=f;chi0=chi;
        }
        double det=last.aa*last.bb-last.ab*last.ab;
        return {a,b,float(g0),float(sg*std::sqrt(last.bb/det)),float(h0),
                float(sh*std::sqrt(last.aa/det)),float(-.5*chi0-.5*std::log(det))};
    }
};

void endpoint_pairs(const Bank &bank,int a,int family,std::vector<Pair>&out){
    const double radius[]={44.,39.,83.},width[]={.105,.24,.46},av[]={.09,.22,.48},bv[]={.14,.39,.70};
    Prior prior(bank,a,radius[family],width[family],av[family],bv[family]);
    auto mag=bank.phot(a);int ea=bank.d[a].e;
    for(int e=0;e<bank.ne;e++){
        const auto &im=bank.ends[e];
        if(!im.nx||bank.ex[e].night==bank.ex[ea].night||bank.depth[e]+.8<mag.mag)continue;
        double dt=std::abs(bank.ex[e].t-bank.ex[ea].t);if(dt<.5)continue;
        V pos,u;double l,w;prior.mean(e,pos,u,l,w);
        double xx=sq(u.x)+sq(l*prior.east.x)+sq(w*prior.north.x)+.16;
        double xy=u.x*u.y+sq(l)*prior.east.x*prior.east.y+sq(w)*prior.north.x*prior.north.y;
        double yy=sq(u.y)+sq(l*prior.east.y)+sq(w*prior.north.y)+.16;
        double area=std::sqrt(std::max(.01,xx*yy-xy*xy));
        double rho=im.density[std::clamp(int((mag.mag-15.)*4.),0,54)];
        struct Choice {int row;float score;};
        std::vector<Choice> short_list;
        im.ellipse(pos,xx,xy,yy,3.2,[&](const Point &b,double chi){
            if(bank.peer[a]>=0&&bank.peer[b.id]>=0&&a>b.id)return;
            double pc=bank.phot_chi(mag,b.id);
            if(pc>10.)return;
            float v=float(.66*b.q+.9*(bank.peer[b.id]>=0)-.5*pc-.38*chi);
            short_list.push_back({b.id,v});
            if(short_list.size()>=40)best(short_list,12);
        });
        best(short_list,10);
        for(auto &c:short_list){
            auto q=prior.posterior(c.row,0.);
            if(!q.g)continue;
            q.score+=float(c.score+.67*bank.quality[a]-.52*std::log(.018+rho*area));
            out.push_back(q);
        }
    }
}

struct Band {int id,night; V z,u,v;double sig,mark;};
#include "three_visit_returns.hpp"
struct View {int e;V center,u,v;double sig,xx,xy,yy,utility;};
struct Finder {
    const Bank &bank;const Pair &p;
    const Target &a,&b;const Exposure &A,&B;
    Midpoint mag;
    State base,dg,dh;
    std::vector<Band> bands;
    std::vector<Seed> seeds;
    std::array<long long,5> counter{};
    Finder(const Bank&bank_,const Pair&pair):
       bank(bank_),p(pair),a(bank.d[p.a]),b(bank.d[p.b]),A(bank.ex[a.e]),B(bank.ex[b.e]),
       mag(bank.phot(p.a,p.b)),base(a,b,A,B,p.g,p.h),
       dg(a,b,A,B,p.g+.02*p.sg,p.h),dh(a,b,A,B,p.g,p.h+.02*p.sh){}
    void measure(){
        std::vector<View> views;
        for(int e=0;e<bank.ne;e++){
            const auto &E=bank.ex[e];const auto &im=bank.fields[e];
            if(!im.nx||E.night==A.night||E.night==B.night||bank.depth[e]+.83<mag.mag)continue;
            double t=E.t-A.t,f=t/(B.t-A.t);
            V pos=base.position(t,E);
            V u=(dg.position(t,E)-pos)*50.,v=(dh.position(t,E)-pos)*50.;
            // A first-order map is used only for finding intersections. Give
            // curved maps a wider envelope before fitting the actual rows.
            double nonlinear=.005*std::sqrt(u.norm2())*std::min(2.,std::abs(t)*p.sh*40.+
                                                          sq(t)*base.k*p.sg/p.g*18.);
            double sig=std::sqrt(.18*.18+sq((1.-f)*bank.noise(p.a))+sq(f*bank.noise(p.b))+
                                 sq(std::min(40.,nonlinear)));
            double xx=u.x*u.x+v.x*v.x+sig*sig,xy=u.x*u.y+v.x*v.y,yy=u.y*u.y+v.y*v.y+sig*sig;
            if(!im.touches(pos,3.*std::sqrt(xx),3.*std::sqrt(yy)))continue;
            double area=std::sqrt(std::max(.0001,xx*yy-xy*xy));
            double rho=im.density[std::clamp(int((mag.mag-15.)*4.),0,54)];
            double visible=1./(1.+std::exp(std::clamp((mag.mag-bank.depth[e]-.12)/.27,-20.,20.)));
            double utility=std::log(.02+visible)-.52*std::log(.018+rho*area);
            views.push_back({e,pos,u,v,sig,xx,xy,yy,utility});
        }
        std::sort(views.begin(),views.end(),[](const auto&x,const auto&y){return x.utility>y.utility;});
        // Interleave nights with many dithers. The per-visit likelihood
        // still uses the actual shutter time, filter and detector bounds.
        std::vector<int> seen(bank.ne+1);
        for(auto &v:views)v.utility-=1.15*std::log1p(seen[bank.ex[v.e].night]++);
        std::sort(views.begin(),views.end(),[](const auto&x,const auto&y){return x.utility>y.utility;});
        for(auto &view:views){
            const auto &im=bank.fields[view.e];
            if(view.utility< -3.5 || bands.size()>500)break;
            std::vector<Band> local;
            im.ellipse(view.center,view.xx,view.xy,view.yy,3.05,[&](const Point &pt,double chi){
                double pc=bank.phot_chi(mag,pt.id);if(pc>10.)return;
                double sig=std::hypot(view.sig,bank.noise(pt.id));
                V diff=V{pt.x,pt.y}-view.center;
                double mark=.38*std::clamp(double(pt.q),-6.,6.)-.42*pc-.10*chi+
                            .66*(bank.peer[pt.id]>=0)+.48*view.utility;
                local.push_back({pt.id,bank.ex[view.e].night,diff,view.u,view.v,sig,mark});
            });
            if(local.size()>55){
                std::nth_element(local.begin(),local.begin()+55,local.end(),
                                 [](const auto &x,const auto &y){return x.mark>y.mark;});local.resize(55);
            }
            bands.insert(bands.end(),local.begin(),local.end());
        }
        counter[0]+=views.size();counter[1]+=bands.size();
    }
    void matches(const std::vector<int>&ids,double gx,double gy,double wx,double wy){
        for(size_t j=1;j<ids.size();j++)for(size_t i=0;i<j;i++){
            auto &x=bands[ids[i]],&y=bands[ids[j]];if(x.night==y.night)continue;
            Normal n;
            for(const auto *m:{&x,&y}){
                double w=1./sq(m->sig);
                n.add(m->u.x,m->v.x,m->z.x,w);n.add(m->u.y,m->v.y,m->z.y,w);
            }
            double u=0.,v=0.;double chi=n.fit(u,v);
            if(chi>21. || std::abs(u-gx)>wx || std::abs(v-gy)>wy)continue;
            int third=-1;double witness=-1.e10;
            double determinant=n.aa*n.bb-n.ab*n.ab;
            for(int k:ids){
                auto &z=bands[k];if(z.night==x.night||z.night==y.night)continue;
                V d=z.z-z.u*u-z.v*v;
                double vx=(n.bb*z.u.x*z.u.x-2*n.ab*z.u.x*z.v.x+n.aa*z.v.x*z.v.x)/determinant+sq(z.sig);
                double vy=(n.bb*z.u.y*z.u.y-2*n.ab*z.u.y*z.v.y+n.aa*z.v.y*z.v.y)/determinant+sq(z.sig);
                double xy=(n.bb*z.u.x*z.u.y-n.ab*(z.u.x*z.v.y+z.v.x*z.u.y)+n.aa*z.v.x*z.v.y)/determinant;
                double det=vx*vy-xy*xy;
                if(!(det>0.))continue;
                double cost=(d.x*(vy*d.x-2.*xy*d.y)+vx*d.y*d.y)/det;
                if(cost>20.)continue;
                double gain=z.mark-.5*cost-.30*std::log(det)+5.;
                if(gain>witness){witness=gain;third=z.id;}
            }
            double rank=.35*p.score+x.mark+y.mark-.5*chi-.21*std::log(determinant);
            if(third>=0)rank+=std::clamp(witness,1.,12.);
            double g=p.g+u*p.sg;
            if(g>1./2000. && g<1./14.){
                seeds.push_back({{p.a,p.b,x.id,y.id,third},float(g),float(rank)});
                if(seeds.size()>=80)best(seeds,20);
                counter[4]+=third>=0;counter[3]++;
            }
        }
    }
    void divide(const std::vector<int>&ids,double gx,double gy,double wx,double wy,int level){
        if(ids.size()<2||counter[2]>=2200)return;
        counter[2]++;
        if(ids.size()<=12 || level>=15){
            matches(ids,gx,gy,wx,wy);return;
        }
        // Divide along the parameter that moves these measurements most.
        double sumx=0.,sumy=0.;
        for(int k:ids){const auto &b=bands[k];sumx+=b.u.norm2()/sq(b.sig);sumy+=b.v.norm2()/sq(b.sig);}
        bool split=wx*wx*sumx>=wy*wy*sumy;
        if(split)wx*=.5;else wy*=.5;
        for(int sign:{-1,1}){
            double x0=gx+(split?sign*wx:0.),y0=gy+(split?0.:sign*wy);
            std::vector<int> child;int night=-1;bool different=false;
            for(int k:ids){
                const auto &b=bands[k];V d=b.z-b.u*x0-b.v*y0;
                if(std::abs(d.x)>3.9*b.sig+wx*std::abs(b.u.x)+wy*std::abs(b.v.x) ||
                   std::abs(d.y)>3.9*b.sig+wx*std::abs(b.u.y)+wy*std::abs(b.v.y))continue;
                different|=night>=0&&night!=b.night;night=b.night;child.push_back(k);
            }
            if(different)divide(child,x0,y0,wx,wy,level+1);
        }
    }
    void run(){
        measure();std::vector<int> ids(bands.size());std::iota(ids.begin(),ids.end(),0);
        divide(ids,0.,0.,3.2,3.2,0);best(seeds,12);
    }
};
}

extern "C" {
void *return_bank(const returns::Target*d,int n,const returns::Exposure*e,int ne,const float *sn,
    const float*q,const int*peers,const double*depth,const double*lever,const int*end,int count,double x,double y){
    try {return new returns::Bank(d,n,e,ne,sn,q,peers,depth,lever,end,count,{x,y});}
    catch(...){return nullptr;}
}
void release_return_bank(void *bank){delete static_cast<returns::Bank*>(bank);}
int return_three(void *handle,const returns::Triple*triples,int count,int threads,double seconds,int cap,
                 returns::Seed*out,int *tested,returns::RawCore*cores){
    using namespace returns;
    try{
        auto &bank=*static_cast<Bank*>(handle);std::atomic<int> next{0};
        auto end=Clock::now()+std::chrono::duration<double>(seconds);
        std::vector<std::vector<Seed>> results(threads);std::vector<std::thread> pool;
        for(int t=0;t<threads;t++)pool.emplace_back([&,t]{
            auto &r=results[t];
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=count)break;
                three_returns(bank,triples[i],r,cores?cores+i:nullptr);
                if(r.size()>size_t(2*cap/threads))best(r,cap/threads);
            }
        });
        for(auto &thread:pool)thread.join();
        std::vector<Seed> all;
        for(auto &r:results)all.insert(all.end(),r.begin(),r.end());
        best(all,cap);*tested=std::min(count,next.load());std::copy(all.begin(),all.end(),out);
        return all.size();
    }catch(...){return -1;}
}
int return_pairs(void *handle,const int *anchors,int count,int family,int threads,double seconds,int cap,
                 returns::Pair*out,int *tested){
    using namespace returns;
    try{
        auto &bank=*static_cast<Bank*>(handle);std::atomic<int> next{0};
        auto deadline=Clock::now()+std::chrono::duration<double>(seconds);
        std::vector<std::vector<Pair>> result(threads);std::vector<std::thread> pool;
        for(int t=0;t<threads;t++)pool.emplace_back([&,t]{
            auto &r=result[t];std::vector<Pair> one;
            while(Clock::now()<deadline){
                int i=next.fetch_add(1);if(i>=count)break;
                one.clear();endpoint_pairs(bank,anchors[i],family,one);best(one,54);
                r.insert(r.end(),one.begin(),one.end());
                if(r.size()>size_t(2*cap/threads))best(r,cap/threads);
            }
        });
        for(auto &t:pool)t.join();std::vector<Pair> all;
        for(auto &r:result)all.insert(all.end(),r.begin(),r.end());
        best(all,cap);std::sort(all.begin(),all.end(),[](auto &a,auto&b){return a.score>b.score;});
        *tested=std::min(count,next.load());std::copy(all.begin(),all.end(),out);return all.size();
    }catch(...){return -1;}
}
int return_search(void *handle,const returns::Pair*p,int count,int threads,double seconds,int cap,
                  returns::Seed*out,long long *stats){
    using namespace returns;
    try{
        auto &bank=*static_cast<Bank*>(handle);std::atomic<int> next{0};
        auto deadline=Clock::now()+std::chrono::duration<double>(seconds);
        std::vector<std::vector<Seed>> result(threads);std::vector<std::thread> pool;
        std::vector<std::array<long long,5>> counts(threads);
        for(int t=0;t<threads;t++)pool.emplace_back([&,t]{
            auto &r=result[t];
            while(Clock::now()<deadline){
                int j=next.fetch_add(1);if(j>=count)break;
                Finder f(bank,p[j]);f.run();
                r.insert(r.end(),f.seeds.begin(),f.seeds.end());
                for(int k=0;k<5;k++)counts[t][k]+=f.counter[k];
                if(r.size()>size_t(2*cap/threads))best(r,cap/threads);
            }
        });
        for(auto &t:pool)t.join();std::vector<Seed> all;
        for(auto &r:result)all.insert(all.end(),r.begin(),r.end());
        stats[0]=std::min(count,next.load());
        for(int k=0;k<5;k++){stats[k+1]=0;for(auto &c:counts)stats[k+1]+=c[k];}
        best(all,cap);std::copy(all.begin(),all.end(),out);return all.size();
    }catch(...){return -1;}
}
}
