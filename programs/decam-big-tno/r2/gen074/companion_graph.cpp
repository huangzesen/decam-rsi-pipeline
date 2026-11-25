// Marked point-process likelihoods for independent DECam visits.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {
constexpr double arc = 206264.80624709636;
constexpr double gauss = .01720209895;
constexpr double pi = 3.141592653589793;
constexpr int bands = 36;
constexpr int max_links = 10;
constexpr int cell = 35;
constexpr int macro = 12;
inline double square(double x) { return x*x; }
int magbin(double m) {return std::clamp(int((m-12.)*2.),0,bands-1);}

struct Visit {double t,ox,oy,oz,depth,lever; int night,pad;};
struct Point {float x,y,mag,sn,mark; int row;};
struct Link {int row=-1; float score=-100,g=0;};
struct Result {
    float evidence=0, burst=0, three=0, control=0;
    int rows[3]={-1,-1,-1};
    float gamma=0;
    Link pair[2];
};
struct Grid {
    int bx=0,by=0,nx=0,ny=0,mx=0;
    double cx=0,cy=0,xmin=0,ymin=0,xmax=0,ymax=0;
    std::vector<int> start,occupancy;
    std::vector<Point> pts;
    std::vector<float> rho;
    int bin(int x,int y) const {return x+nx*y;}
    int coarse(int x,int y) const {return x/macro+mx*(y/macro);}
    bool inside(int x,int y) const {return x>=0&&y>=0&&x<nx&&y<ny;}
    void build(const float *xy,const float *mag,const float *sn,const float *mark,
               const int *ids,int n) {
        if(!n)return;
        xmin=xmax=xy[2*ids[0]];ymin=ymax=xy[2*ids[0]+1];
        for(int i=0;i<n;i++){
            int r=ids[i];xmin=std::min(xmin,double(xy[2*r]));xmax=std::max(xmax,double(xy[2*r]));
            ymin=std::min(ymin,double(xy[2*r+1]));ymax=std::max(ymax,double(xy[2*r+1]));
        }
        bx=int(std::floor(xmin/cell));by=int(std::floor(ymin/cell));
        nx=int(std::floor(xmax/cell))-bx+1;ny=int(std::floor(ymax/cell))-by+1;
        if(int64_t(nx)*ny>4000000) {nx=ny=0;return;}
        cx=.5*(xmin+xmax);cy=.5*(ymin+ymax);
        mx=(nx+macro-1)/macro;
        const int nc=mx*((ny+macro-1)/macro);
        start.assign(nx*ny+1,0);std::vector<int> h(nc*bands),total(bands);
        occupancy.assign(nc,0);
        for(int i=0;i<n;i++){
            int r=ids[i],x=int(std::floor(xy[2*r]/cell))-bx,y=int(std::floor(xy[2*r+1]/cell))-by;
            start[bin(x,y)+1]++;int k=magbin(mag[r]);
            h[coarse(x,y)*bands+k]++;total[k]++;
        }
        int occupied=0;
        for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
            int b=bin(x,y);
            if(start[b+1]){occupancy[coarse(x,y)]++;occupied++;}
            start[b+1]+=start[b];
        }
        auto cursor=start;pts.resize(n);rho.resize(nc*bands);
        for(int i=0;i<n;i++){
            int r=ids[i],x=int(std::floor(xy[2*r]/cell))-bx,y=int(std::floor(xy[2*r+1]/cell))-by;
            pts[cursor[bin(x,y)]++]={xy[2*r],xy[2*r+1],mag[r],sn[r],mark[r],r};
        }
        auto smooth=[](const int *v,int k){
            return (v[k]+.50*(v[std::max(0,k-1)]+v[std::min(bands-1,k+1)]));
        };
        const double area=double(cell*cell)*std::max(1,occupied);
        for(int k=0;k<bands;k++){
            double global=(smooth(total.data(),k)+1.)/area;
            for(int z=0;z<nc;z++){
                double rate=(smooth(&h[z*bands],k)+26000.*global+.13)/(26000.+cell*cell*occupancy[z]);
                rho[z*bands+k]=std::max(1.e-8,rate);
            }
        }
    }
};

struct Gaussian {
    double meanx,meany,stretch,xx,xy,yy,normal,g,wx,wy,weight;
};
// Mean motion of a circular orbit and the derivatives with respect to
// log inverse distance, tangential speed and inclination. All offsets are
// in arcseconds. The change in projected observer depth is affine in sky.
struct Transfer {
    int e;
    double dt;
    std::array<Gaussian,4> model;
    bool same;
    Transfer(int j,const Visit &a,const Visit &b,const Grid &im,double east,double north):e(j),dt(b.t-a.t),same(a.night==b.night){
        const double r[]={44.,39.,76.,330.},width[]={.095,.26,.35,.72},weight[]={.48,.40,.085,.035};
        double x=im.cx/arc,y=im.cy/arc,norm=std::sqrt(1.+x*x+y*y);
        for(int s=0;s<4;s++){
            double g=norm/r[s],vc=gauss/std::pow(r[s],1.5)*norm*arc*dt;
            double d=1.-g*b.oz;
            double ox=(b.ox-a.ox-x*(b.oz-a.oz))*arc;
            double oy=(b.oy-a.oy-y*(b.oz-a.oz))*arc;
            double dx=(vc*east-g*ox)/d,dy=(vc*north-g*oy)/d;
            double kx=width[s]*(1.5*vc*east-g*ox+g*b.oz*dx)/d;
            double ky=width[s]*(1.5*vc*north-g*oy+g*b.oz*dy)/d;
            double u=(s==0?.095:s==1?.24:.48)*vc/d;
            double v=(s==0?.12:s==1?.35:.62)*vc/d;
            double floor=.26*.26+square(.004*dt);
            double xx=floor+kx*kx+square(east*u)+square(north*v);
            double yy=floor+ky*ky+square(north*u)+square(east*v);
            double xy=kx*ky+east*north*(u*u-v*v),det=std::max(1.e-8,xx*yy-xy*xy);
            model[s]={dx,dy,g*(b.oz-a.oz)/d,yy/det,-xy/det,xx/det,
                      weight[s]/(2*pi*std::sqrt(det)),g,3.1*std::sqrt(xx),3.1*std::sqrt(yy),weight[s]};
        }
    }
};

struct Companion {int row=-1,night=-1; double loglr=-100.,g=0.;};
struct Match {double odds=0.,stationary=0.,fast=0.,coverage=0.; Companion best;};
struct Search {
    const float *xy,*mag,*sn,*marks;
    const int *exp;
    const Visit *vis;
    double east,north;
    std::vector<Grid> images;
    std::vector<std::vector<Transfer>> visits;

    Match inspect(int i,int from,const Transfer &tr,int sign) const {
        const auto &im=images[tr.e];const auto &a=images[from];
        const double x=xy[2*i],y=xy[2*i+1],m=mag[i];
        double accessible=1./(1.+std::exp(std::clamp((m-vis[tr.e].depth-.16)/.28,-30.,30.)));
        Match out;
        if(accessible<.025)return out;
        std::array<double,4> cx,cy;
        double xl=1.e30,yl=1.e30,xh=-xl,yh=-yl;
        for(int k=0;k<4;k++){
            const auto &h=tr.model[k];
            cx[k]=x+sign*(h.meanx+h.stretch*(x-a.cx));cy[k]=y+sign*(h.meany+h.stretch*(y-a.cy));
            xl=std::min(xl,cx[k]-h.wx);xh=std::max(xh,cx[k]+h.wx);
            yl=std::min(yl,cy[k]-h.wy);yh=std::max(yh,cy[k]+h.wy);
            int bx=int(std::floor(cx[k]/cell))-im.bx,by=int(std::floor(cy[k]/cell))-im.by;
            if(im.inside(bx,by)){
                int c=im.coarse(bx,by);
                out.coverage+=h.weight*std::min(1.,im.occupancy[c]/35.);
            }
        }
        if(out.coverage<.025)return out;
        // Short bursts usually contain unresolved stellar repeats. Compare
        // the orbit marginal with both fixed sky and faster moving objects.
        const double starvar=.26*.26;
        const double fastvar=starvar+square(std::min(48.,550.*std::abs(tr.dt)));
        if(tr.same&&sign==1){
            double reach=std::min(80.,2.8*std::sqrt(fastvar));
            xl=std::min(xl,x-reach);xh=std::max(xh,x+reach);
            yl=std::min(yl,y-reach);yh=std::max(yh,y+reach);
        }
        int l=std::max(0,int(std::floor(xl/cell))-im.bx),h=std::min(im.nx-1,int(std::floor(xh/cell))-im.bx);
        int b=std::max(0,int(std::floor(yl/cell))-im.by),t=std::min(im.ny-1,int(std::floor(yh/cell))-im.by);
        if(h<l||t<b)return out;
        double color=.24*(vis[tr.e].lever-vis[from].lever);
        double base=.16*.16+square(1.0857/std::max(4.5f,sn[i]))+square(color);
        double best=0;
        for(int iy=b;iy<=t;iy++){
            int end=im.start[im.bin(h,iy)+1];
            for(int j=im.start[im.bin(l,iy)];j<end;j++){
                const auto &p=im.pts[j];double dm=p.mag-m;
                if(std::abs(dm)>1.4)continue;
                double variance=base+square(1.0857/std::max(4.5f,p.sn));
                if(dm*dm>13.*variance)continue;
                double density=0.,gs=0.;
                for(int k=0;k<4;k++){
                    const auto &d=tr.model[k];double u=p.x-cx[k],v=p.y-cy[k];
                    double chi=d.xx*u*u+2*d.xy*u*v+d.yy*v*v;
                    if(chi<13.){double w=d.normal*std::exp(-.5*chi);density+=w;gs+=w*d.g;}
                }
                if(density==0.&&!tr.same)continue;
                int ix=int(std::floor(p.x/cell))-im.bx;
                double rho=im.rho[im.coarse(ix,iy)*bands+magbin(p.mag)];
                double factor=std::exp(-.5*dm*dm/variance+p.mark)/(std::sqrt(2*pi*variance)*rho);
                double lr=density*factor;out.odds+=lr;
                if(lr>best){best=lr;out.best={p.row,vis[tr.e].night,0.,gs/density};}
                if(tr.same&&sign==1){
                    double r2=square(p.x-x)+square(p.y-y);
                    if(r2<12.*starvar)out.stationary+=factor*std::exp(-.5*r2/starvar)/(2*pi*starvar);
                    if(r2<12.*fastvar)out.fast+=factor*std::exp(-.5*r2/fastvar)/(2*pi*fastvar);
                }
            }
        }
        out.coverage=std::min(.87,.78*accessible*out.coverage);
        if(best>0.)out.best.loglr=std::log(best*out.coverage+.00001);
        return out;
    }

    double triangle(int a,int b,int c,double &g) const {
        const auto &A=vis[exp[a]],&B=vis[exp[b]],&C=vis[exp[c]];
        // Interpolate between the endpoints, keeping the time and
        // observatory of each individual shutter. The weak range prior
        // regularizes near-collinear geometry at opposition.
        if(A.t>B.t)return triangle(b,a,c,g);
        if(B.t>C.t)return triangle(a,c,b,g);
        double span=C.t-A.t;
        if(span<.6||span>48.)return -100.;
        double f=(B.t-A.t)/span,h=1.-f;
        double ox[3],oy[3],sig2=0.;
        const int row[3]={a,b,c};const double coeff[]={h,-1.,f};
        for(int k=0;k<3;k++){
            const auto &e=vis[exp[row[k]]];double x=xy[2*row[k]]/arc,y=xy[2*row[k]+1]/arc;
            ox[k]=(e.ox-e.oz*x)*arc;oy[k]=(e.oy-e.oz*y)*arc;
            sig2+=square(coeff[k])*(.085*.085+square(1.05/std::max(4.5f,sn[row[k]])));
        }
        sig2+=square(.043+.00033*f*h*span*span);
        double dx=h*xy[2*a]+f*xy[2*c]-xy[2*b],dy=h*xy[2*a+1]+f*xy[2*c+1]-xy[2*b+1];
        double px=h*ox[0]+f*ox[2]-ox[1],py=h*oy[0]+f*oy[2]-oy[1];
        double g0=g,prior=1./square(.23*g0),norm=prior+(px*px+py*py)/sig2;
        g=(prior*g0-(dx*px+dy*py)/sig2)/norm;
        if(g<=1./8000.||g>1./18.)return -100.;
        double chi=(square(dx+g*px)+square(dy+g*py))/sig2+prior*square(g-g0);
        if(chi>18.)return -100.;
        double u=(xy[2*c]-xy[2*a]+g*(ox[2]-ox[0]))/span;
        double v=(xy[2*c+1]-xy[2*a+1]+g*(oy[2]-oy[0]))/span;
        double vc=gauss*std::pow(g,1.5)*arc;
        double along=(u*east+v*north)/vc,across=(v*east-u*north)/vc;
        // A short chord still has a large range-speed degeneracy.
        double unum=(ox[2]-ox[0])*east+(oy[2]-oy[0])*north;
        double vnum=(oy[2]-oy[0])*east-(ox[2]-ox[0])*north;
        double ug=unum/(span*vc)-1.5*along/g,vg=vnum/(span*vc)-1.5*across/g;
        chi+=square(along-1.)/(.31*.31+ug*ug/norm)+square(across)/(.39*.39+vg*vg/norm);
        return 5.-.50*chi-.5*std::log(norm/prior);
    }

    Result measure(int i) const {
        Result result;
        int from=exp[i],last=-1,used=0;
        std::array<Companion,max_links> links;
        double logbf=0,reverse=0,burst=0,weight=0,nightbest=-100.,backbest=-100.;
        Companion best;
        auto finish=[&](){
            if(last<0||nightbest< -99)return;
            if(last==vis[from].night)burst=nightbest;
            else{
                logbf+=std::clamp(nightbest,-.95,6.);reverse+=std::clamp(backbest,-.95,6.);weight++;
            }
            if(best.row>=0){
                if(used<max_links)links[used++]=best;
                else{
                    auto j=std::min_element(links.begin(),links.end(),[](auto &a,auto &b){return a.loglr<b.loglr;});
                    if(best.loglr>j->loglr)*j=best;
                }
            }
        };
        int count=0;
        for(const auto &tr:visits[from]){
            int night=vis[tr.e].night;
            if(night!=last){finish();last=night;nightbest=backbest=-100.;best=Companion();count=0;}
            if(count>=(tr.same?4:3))continue;
            auto pos=inspect(i,from,tr,1);
            if(pos.coverage<.025)continue;
            count++;
            double f=pos.coverage;
            double s=std::log1p(f*(pos.odds-1.));
            if(tr.same){
                s-=std::log(1.+.26*f*pos.stationary+.055*f*pos.fast);
            }else{
                auto neg=inspect(i,from,tr,-1);
                double back=std::log1p(neg.coverage*(neg.odds-1.));
                backbest=std::max(backbest,back);
            }
            nightbest=std::max(nightbest,s);
            if(pos.best.row>=0){
                if(tr.same)pos.best.loglr-=std::log1p(.26*f*pos.stationary+.055*f*pos.fast);
                if(pos.best.loglr>best.loglr)best=pos.best;
            }
        }
        finish();
        result.evidence=logbf/std::sqrt(std::max(1.,weight/2.));
        result.control=reverse/std::sqrt(std::max(1.,weight/2.));
        result.burst=std::clamp(burst,-5.,7.);
        std::sort(links.begin(),links.begin()+used,[](auto &a,auto &b){return a.loglr>b.loglr;});
        for(int j=0;j<std::min(2,used);j++)result.pair[j]={links[j].row,float(links[j].loglr),float(links[j].g)};
        double highest=-100.;
        for(int j=0;j<used;j++)for(int k=j+1;k<used;k++){
            if(links[k].loglr< -1.5)continue;
            double g=.5*(links[j].g+links[k].g);
            double fit=triangle(i,links[j].row,links[k].row,g);
            if(fit< -7.)continue;
            double merit=fit+.65*(std::min(6.,links[j].loglr)+std::min(6.,links[k].loglr));
            if(merit>highest){highest=merit;result.gamma=g;
                result.rows[0]=i;result.rows[1]=links[j].row;result.rows[2]=links[k].row;}
        }
        result.three=std::max(0.,highest);
        return result;
    }
};
}

extern "C" int companion_graph(const float *xy,const float *mag,const float *sn,const float *marks,
             const int *exp,const int *ids,const int *edges,const Visit *vis,int ne,double east,double north,
             int threads,Result *out){
    try{
        Search search{xy,mag,sn,marks,exp,vis,east,north,{},{}};
        search.images.resize(ne);search.visits.resize(ne);
        for(int e=0;e<ne;e++)search.images[e].build(xy,mag,sn,marks,ids+edges[e],edges[e+1]-edges[e]);
        for(int e=0;e<ne;e++){
            auto &list=search.visits[e];const auto &a=search.images[e];
            if(!a.nx)continue;
            for(int j=0;j<ne;j++){
                const auto &b=search.images[j];double days=std::abs(vis[j].t-vis[e].t);
                if(e==j||!b.nx||days>18.5)continue;
                double pad=135.*days+75.;
                if(a.xmax+pad<b.xmin||a.xmin-pad>b.xmax||a.ymax+pad<b.ymin||a.ymin-pad>b.ymax)continue;
                list.emplace_back(j,vis[e],vis[j],a,east,north);
            }
            std::sort(list.begin(),list.end(),[&](const Transfer &a,const Transfer &b){
                if(vis[a.e].night!=vis[b.e].night)return vis[a.e].night<vis[b.e].night;
                double qa=vis[a.e].depth+.22*std::log1p(std::abs(a.dt)*24.);
                double qb=vis[b.e].depth+.22*std::log1p(std::abs(b.dt)*24.);
                return qa>qb;
            });
        }
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int t=0;t<threads;t++)workers.emplace_back([&]{
            for(;;){int b=next.fetch_add(128);if(b>=edges[ne])break;
                for(int i=b;i<std::min(b+128,edges[ne]);i++)out[i]=search.measure(ids[i]);}
        });
        for(auto &worker:workers)worker.join();
        return edges[ne];
    }catch(...){return -1;}
}
