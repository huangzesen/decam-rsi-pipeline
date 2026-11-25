// Join two measured angular displacements. Eliminating the transverse
// position and velocity leaves four residual coordinates for two unknowns.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace chord {
constexpr double arc=206264.80624709636, ksun=.01720209895, cell=530.;
inline double sq(double x){return x*x;}
struct V {
    double x,y;
    V operator+(V p) const {return {x+p.x,y+p.y};}
    V operator-(V p) const {return {x-p.x,y-p.y};}
    V operator*(double c) const {return {x*c,y*c};}
    double dot(V p) const {return x*p.x+y*p.y;}
};
struct Visit {double t,ox,oy,oz,depth,lever;int night,pad;};
struct Target {float x,y,mag;int exp;};
struct Edge {int a,b;float score,g;};
struct Quad {int rows[4];float g,score;};
using Clock=std::chrono::steady_clock;
struct Obs {V p,d;double t,z;int night;};
struct Arc {
    Obs a,b;
    V mid,D,rate,par,velocity,slope;
    double t,dt,mag,color,sg,rv,g,st,sc,mean_z;
    int id[2];
    Arc(const Target *s,const Visit *vis,const Edge &edge,V east,V north){
        id[0]=edge.a;id[1]=edge.b;
        auto make=[&](int i){
            const Target &z=s[i];const Visit &e=vis[z.exp];V p{z.x*arc,z.y*arc};
            return Obs{p,V{e.ox*arc,e.oy*arc}-p*e.oz,e.t,e.oz,e.night};
        };
        a=make(id[0]);b=make(id[1]);if(a.t>b.t){std::swap(a,b);std::swap(id[0],id[1]);}
        dt=b.t-a.t;t=.5*(a.t+b.t);mid=(a.p+b.p)*.5;D=(a.d+b.d)*.5;
        rate=(b.p-a.p)*(1./dt);par=(b.d-a.d)*(1./dt);mean_z=.5*(a.z+b.z);
        mag=.5*(s[id[0]].mag+s[id[1]].mag);
        color=.5*(vis[s[id[0]].exp].lever+vis[s[id[1]].exp].lever);
        sc=edge.score;
        // Condition the prior transverse velocity and distance on this
        // chord. A very short repeat measures both components poorly.
        double norm=std::sqrt(1.+mid.dot(mid)/sq(arc));
        double g0=std::clamp(double(edge.g),1./1800.,1./19.);
        double vc=ksun*std::pow(g0,1.5)*arc/std::sqrt(norm);
        V prior=east*(vc*(g0<1./100.?.67:1.));
        double zeta=(g0<1./100.?.52:.31);
        double vv=sq(zeta*vc),noise=sq(.25/dt),u=vv/(vv+noise);
        V der=east*(1.5*vc/g0);
        double prec=1./sq(g0*(g0<1./100.?.46:.24));
        V h=par-der,res=rate+par*g0-prior;
        double info=prec+h.dot(h)/(vv+noise);
        g=std::clamp(g0-h.dot(res)/((vv+noise)*info),g0*.55,g0*1.55);
        sg=1./std::sqrt(info);
        velocity=prior+der*(g-g0)+(rate+par*g-prior-der*(g-g0))*u;
        slope=der*(1.-u)+par*u;
        st=std::sqrt(vv*noise/(vv+noise));
        rv=.50*ksun*std::pow(g,1.5);
    }
};

constexpr double ref_g=1./44.;
struct Slice {
    double time=0,z=0;V D{0,0};int count=0;
    double dx=0,dy=0;
    V lower{1.e30,1.e30},upper{-1.e30,-1.e30};
    int bx=0,by=0,nx=0,ny=0;
    std::vector<int> local,starts,ids;
    std::vector<V> positions;
    void build(const std::vector<Arc> &chords,V drift){
        if(local.empty())return;
        for(int id:local){const auto &s=chords[id];time+=s.t;D=D+s.D;z+=s.mean_z;}
        count=local.size();time/=count;D=D*(1./count);z/=count;
        for(int id:local){
            const auto &s=chords[id];
            V p=s.mid+s.D*ref_g-drift*(s.t-time);
            positions.push_back(p);
            lower={std::min(lower.x,p.x),std::min(lower.y,p.y)};
            upper={std::max(upper.x,p.x),std::max(upper.y,p.y)};
            dx=std::max(dx,std::abs(D.x-s.D.x));dy=std::max(dy,std::abs(D.y-s.D.y));
        }
        bx=std::floor(lower.x/cell);by=std::floor(lower.y/cell);
        nx=int(std::floor(upper.x/cell))-bx+1;ny=int(std::floor(upper.y/cell))-by+1;
        if(int64_t(nx)*ny>2000000){nx=ny=0;return;}
        starts.assign(nx*ny+1,0);ids.resize(count);
        auto bin=[&](V p){return (int(std::floor(p.y/cell))-by)*nx+int(std::floor(p.x/cell))-bx;};
        for(V p:positions)starts[bin(p)+1]++;
        for(size_t i=1;i<starts.size();i++)starts[i]+=starts[i-1];
        auto cursor=starts;
        for(size_t i=0;i<local.size();i++)ids[cursor[bin(positions[i])]++]=local[i];
    }
};

struct Equation {
    double xx,xy,yy,hx=0,hy=0,chi=0;
    void add(double r,double a,double b,double wt){
        xx+=wt*a*a;xy+=wt*a*b;yy+=wt*b*b;
        hx+=wt*a*r;hy+=wt*b*r;chi+=wt*r*r;
    }
    bool solve(double &g,double &h,double &res,double &logdet) const{
        double det=xx*yy-xy*xy;
        if(!(det>0.))return false;
        g=(xy*hy-yy*hx)/det;h=(xy*hx-xx*hy)/det;
        res=std::max(0.,chi+g*hx+h*hy);logdet=std::log(det);
        return true;
    }
};

// At the centre of the time span, f and tau=s/f are the Kepler
// coefficients. The short chord at either end is compared with the long
// chord, which supplies a much more accurate transverse velocity.
struct Curve {double t,invf;};
Curve coefficients(double dt,double n2,double radial){
    double z=n2*dt*dt,u=radial*dt;
    double f=1.-.5*z*(1.-u)+z*(z-15.*u*u)/24.;
    double tau=dt*(1.-z/6.+z*u/4.+z*(z-45.*u*u)/120.)/f;
    return {tau,1./f};
}
bool match(const Arc &left,const Arc &right,V east,V north,double &gamma,double &score){
    double span=right.t-left.t;
    if(std::abs(span)<1.2*std::max(left.dt,right.dt))return false;
    const Obs p[4]={left.a,left.b,right.a,right.b};
    if(p[0].night==p[2].night||p[0].night==p[3].night||p[1].night==p[2].night||p[1].night==p[3].night)return false;
    if(p[0].night==p[1].night && p[2].night==p[3].night)return false;
    double dt=p[2].t-p[0].t;
    if(std::abs(dt)<1.)return false;
    double mid=.5*(p[0].t+p[2].t),t[4];
    for(int j=0;j<4;j++)t[j]=p[j].t-mid;
    double g0=.5*(left.g+right.g),g=g0,h=0.;
    double norm=1.+.25*(left.mid+right.mid).dot(left.mid+right.mid)/sq(arc);
    double chi=0,ld=0.,motion=0.;
    double phot=sq((left.mag-right.mag)/std::hypot(.26,.27*(left.color-right.color)));
    if(phot>16.)return false;
    double gw=.44*g0,hw=.48*ksun*std::pow(g0,1.5);
    for(int step=0;step<2;step++){
        double n2=step?sq(ksun)*g*g*g/std::pow(norm,1.5):0.;
        double radial=step?h+(left.velocity+right.velocity).dot(left.mid+right.mid)*.25/sq(arc):0.;
        Curve cv[4];for(int j=0;j<4;j++)cv[j]=coefficients(t[j],n2,radial);
        double baseline=cv[2].t-cv[0].t;
        Equation fit{1.,0.,1.};
        for(int j:{1,3}){
            double f=(cv[j].t-cv[0].t)/baseline;
            V R=p[j].p-p[0].p-(p[2].p-p[0].p)*f;
            V G=p[j].d*cv[j].invf-p[0].d*cv[0].invf-(p[2].d*cv[2].invf-p[0].d*cv[0].invf)*f;
            V H=(p[j].p-p[0].p)*cv[j].t-(p[2].p-p[0].p)*(cv[2].t*f);
            R=R+G*g0;
            double chord=j==1?left.dt:right.dt;
            double err2=.055+sq(.14)*(sq(f)+sq(1.-f));
            double ang=ksun*std::pow(g,1.5)*std::abs(span);
            err2+=sq((step?.012:.90)*ksun*std::pow(g,1.5)*arc*chord*ang*ang);
            fit.add(R.x,G.x*gw,H.x*hw,1./err2);fit.add(R.y,G.y*gw,H.y*hw,1./err2);
        }
        V U=(p[2].p-p[0].p)*(1./baseline);
        V K=(p[2].d*cv[2].invf-p[0].d*cv[0].invf)*(1./baseline);
        V Z=(p[2].p*cv[2].t-p[0].p*cv[0].t)*(1./baseline);
        V vel=U+K*g0;
        double vc=ksun*std::pow(g,1.5)*arc/std::pow(norm,.25);
        fit.add(vel.dot(east)-vc*(1.+1.5*(g0-g)/g),gw*(K.dot(east)-1.5*vc/g),
                hw*Z.dot(east),1./sq(.30*vc+.2/std::abs(dt)));
        fit.add(vel.dot(north),K.dot(north)*gw,Z.dot(north)*hw,1./sq(.39*vc+.2/std::abs(dt)));
        double u,v;
        if(!fit.solve(u,v,chi,ld)||chi>(step?28.:40.))return false;
        g=g0+gw*u;h=hw*v;
        if(!(g>1./2500.&&g<1./17.)||std::abs(u)>3.3||std::abs(v)>3.4)return false;
        V w=U+K*g+Z*h;
        motion=(w.dot(w)+sq(h*arc))/(sq(ksun)*g*g*g*arc*arc/std::sqrt(norm));
        // A large radial solar velocity in a tilted tangent plane does not
        // imply a large speed along the line of sight. Keep a margin in
        // this preliminary energy gate for weakly constrained short arcs.
        if(motion>(step?2.40:3.25))return false;
    }
    gamma=g;
    score=-.5*chi-.40*phot-.24*ld+.47*(left.sc+right.sc)
          +.26*std::log1p(std::abs(span)/60.)-.75*std::max(0.,motion-1.7);
    return true;
}

void trim(std::vector<Quad> &v,int n){
    if(int(v.size())<=n)return;
    std::nth_element(v.begin(),v.begin()+n,v.end(),[](auto &a,auto &b){return a.score>b.score;});v.resize(n);
}
}

extern "C" void pack_motion_chords(const chord::Target *target,const chord::Visit *vis,
        const chord::Edge *edges,int n,double ex,double ey,float *packed,int *rows){
    using namespace chord;
    V east{ex,ey},north{-ey,ex};
    for(int i=0;i<n;i++){
        Arc a(target,vis,edges[i],east,north);
        float *v=packed+24*i;
        const double values[]={a.t-vis[0].t,a.dt,a.mid.x,a.mid.y,a.rate.x,a.rate.y,
                 a.D.x,a.D.y,a.par.x,a.par.y,a.mag,a.color,a.g,a.sg,a.velocity.x,a.velocity.y,
                 a.slope.x,a.slope.y,a.st,a.rv,a.sc,a.mean_z,0.,0.};
        for(int j=0;j<24;j++)v[j]=values[j];
        rows[4*i]=a.id[0];rows[4*i+1]=a.id[1];rows[4*i+2]=a.a.night;rows[4*i+3]=a.b.night;
    }
}

extern "C" int join_motion_chords(const chord::Target *targets,const chord::Visit *vis,const chord::Edge *edges,int n,
             double eastx,double easty,int threads,double seconds,int limit,chord::Quad *out,long long *stats){
    using namespace chord;
    try{
        if(!n)return 0;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        V east{eastx,easty},north{-easty,eastx};
        std::vector<Arc> arcs;arcs.reserve(n);
        double first=1.e10,last=-1.e10;
        for(int k=0;k<n;k++){
            auto &s=arcs.emplace_back(targets,vis,edges[k],east,north);
            first=std::min(first,s.t);last=std::max(last,s.t);
        }
        std::vector<Slice> slices(int((last-first)/4.)+1);
        for(int k=0;k<n;k++)slices[int((arcs[k].t-first)/4.)].local.push_back(k);
        V drift=east*(ksun*std::pow(ref_g,1.5)*arc);
        for(auto &s:slices)s.build(arcs,drift);
        std::vector<std::vector<Quad>> banks(threads);std::atomic<int> next{0};
        std::vector<std::array<long long,3>> counts(threads);
        std::vector<std::thread> workers;
        for(int thread=0;thread<threads;thread++)workers.emplace_back([&,thread]{
            auto &bank=banks[thread];auto &count=counts[thread];int cap=std::max(1000,3*limit/threads);
            for(;;){
                int i=next.fetch_add(1);if(i>=n||Clock::now()>=end)break;
                const Arc &a=arcs[i];count[0]++;
                // Only a resolved multi-day chord starts an ellipse. A
                // second local motion can be measured within one night.
                if(a.a.night==a.b.night)continue;
                std::vector<Quad> local;
                for(const auto &s:slices){
                    if(!s.nx)continue;
                    double dt=s.time-a.t,days=std::abs(dt);
                    if(days<1.||days>4900.)continue;
                    double phase=ksun*std::pow(a.g,1.5)*dt;
                    V shift=a.velocity*(dt*(1.+phase*phase/3.));
                    V pos=a.mid+a.D*a.g+shift-s.D*(a.g-ref_g);
                    V axis=(a.D+a.slope*dt-s.D)*a.sg;
                    double thick=22.+days*(a.st+.035)+.83*a.rv*days*(a.g*arc+std::sqrt(shift.dot(shift)));
                    double floor=sq(thick)+sq(2.1*(a.velocity-drift).dot(north));
                    double xx=floor+sq(axis.x)+sq(2.1*(a.velocity.x-drift.x))+sq((std::abs(a.g-ref_g)+a.sg)*s.dx);
                    double yy=floor+sq(axis.y)+sq(2.1*(a.velocity.y-drift.y))+sq((std::abs(a.g-ref_g)+a.sg)*s.dy);
                    double xy=axis.x*axis.y;
                    double wx=3.6*std::sqrt(xx),wy=3.6*std::sqrt(yy);
                    if(pos.x+wx<s.lower.x||pos.x-wx>s.upper.x||pos.y+wy<s.lower.y||pos.y-wy>s.upper.y)continue;
                    int yl=std::max(0,int(std::floor((pos.y-wy)/cell))-s.by);
                    int yh=std::min(s.ny-1,int(std::floor((pos.y+wy)/cell))-s.by);
                    double slope=xy/yy,base=std::sqrt(std::max(1.,xx-xy*slope));
                    for(int by=yl;by<=yh;by++){
                        double y=(by+s.by+.5)*cell-pos.y,center=pos.x+slope*y;
                        double width=3.6*base+.5*std::abs(slope)*cell;
                        int xl=std::max(0,int(std::floor((center-width)/cell))-s.bx);
                        int xh=std::min(s.nx-1,int(std::floor((center+width)/cell))-s.bx);
                        if(xl>xh)continue;
                        for(int j=s.starts[by*s.nx+xl];j<s.starts[by*s.nx+xh+1];j++){
                            int k=s.ids[j];const Arc &b=arcs[k];
                            if((k<=i&&b.a.night!=b.b.night)||std::abs(a.mag-b.mag)>1.10||
                                std::abs(a.t-b.t)<1.2*std::max(a.dt,b.dt))continue;
                            if(std::abs(a.g-b.g)>4.2*std::hypot(a.sg,b.sg)+.80*std::max(a.rv,b.rv)*days*std::max(a.g,b.g))continue;
                            count[1]++;
                            double g,score;
                            if(match(a,b,east,north,g,score)){
                                Quad q{{a.id[0],a.id[1],b.id[0],b.id[1]},float(g),float(score)};
                                local.push_back(q);count[2]++;
                            }
                        }
                    }
                }
                // A single real or unusually crowded exposure pair can have
                // many spurious continuations. Preserve independent anchors.
                trim(local,28);
                bank.insert(bank.end(),local.begin(),local.end());
                if(int(bank.size())>2*cap)trim(bank,cap);
            }
        });
        for(auto &w:workers)w.join();
        std::vector<Quad> all;
        for(int j=0;j<threads;j++){all.insert(all.end(),banks[j].begin(),banks[j].end());
            for(int k=0;k<3;k++)stats[k]+=counts[j][k];}
        trim(all,limit);
        std::sort(all.begin(),all.end(),[](auto &a,auto &b){return a.score>b.score;});
        std::copy(all.begin(),all.end(),out);return int(all.size());
    }catch(...){return -1;}
}
