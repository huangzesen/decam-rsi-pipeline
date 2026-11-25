// In a frame following a circular guide orbit, nearby orbits have small
// residual drift. Fit that drift, inverse range and radial motion to the
// individual directions that meet in each occupied cell.
#ifdef __CUDACC__
#define SH __device__ __forceinline__
#else
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>
#define SH inline
using std::floor;
using std::sqrt;
using std::fmax;
using std::fmin;
#endif
namespace rotating {
constexpr double arc=206264.80624709636,kepler=.01720209895;
struct Source {double ux,uy,uz;float mag,q;int row,ex;};
struct View {double t,ox,oy,oz,g[3],c[3],s[3];int night,pad;};
struct Cells {float x,y,gx,gy,hx,hy;};
struct Params {double radius,angle,speed,radial,epoch,cell,magshift,xshift,yshift,window;};
struct Seed {int rows[4];float gamma,score;int witness=-1;};
using Key=unsigned long long;
struct Entry {Key key;int point;};
SH double sq(double x){return x*x;}
SH double clip(double x,double a,double b){return fmax(a,fmin(x,b));}

SH void map(const Source &a,const View &e,double polex,double poley,int k,double &x,double &y){
    double g=e.g[k],d=a.ux*e.ox+a.uy*e.oy+a.uz*e.oz;
    double range=sqrt(1.-g*g*(sq(e.ox)+sq(e.oy)+sq(e.oz)-d*d))-g*d;
    double X=g*e.ox+range*a.ux,Y=g*e.oy+range*a.uy,Z=g*e.oz+range*a.uz;
    double scalar=polex*X+poley*Y,c=e.c[k],s=e.s[k];
    double z=c*Z+s*(polex*Y-poley*X);
    x=arc*(X*c+s*poley*Z+polex*scalar*(1.-c))/z;
    y=arc*(Y*c-s*polex*Z+poley*scalar*(1.-c))/z;
}
SH Key transform(const Source &a,const View &e,const Params &p,double px,double py,Cells &v){
    double x,y,dx,dy;
    map(a,e,px,py,0,x,y);
    v.x=x;v.y=y;
    map(a,e,px,py,1,dx,dy);v.gx=(dx-x)*1000.;v.gy=(dy-y)*1000.;
    map(a,e,px,py,2,dx,dy);v.hx=(dx-x)*1000.;v.hy=(dy-y)*1000.;
    // One small integer cell in time, magnitude and each sky dimension.
    int ix=(int)floor(x/p.cell+p.xshift)+524288,iy=(int)floor(y/p.cell+p.yshift)+524288;
    int band=(int)clip(floor(a.mag/.84+p.magshift)+40.,0.,127.);
    int slot=(int)clip(floor((e.t-p.epoch)/p.window+p.xshift)+128.,0.,255.);
    return (((Key(slot)*128+band)*1048576+(ix&1048575))*1048576)+(iy&1048575);
}
}

#ifdef __CUDACC__
extern "C" __global__ void guide_cells(const rotating::Source *src,int n,const rotating::View *vis,
        const rotating::Params *p,double px,double py,rotating::Cells *xy,rotating::Key *keys){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)keys[i]=rotating::transform(src[i],vis[src[i].ex],*p,px,py,xy[i]);
}
#else
namespace rotating {
struct Constraint {double x,y,gx,gy,hx,hy,sig;};
struct Normal {
    double aa=1./sq(.13),ab=0.,bb=1./sq(.33),ax=0.,bx=0.,ss=0.;
    void add(const Constraint &c){
        double w=1./sq(c.sig);
        aa+=w*(sq(c.gx)+sq(c.gy));ab+=w*(c.gx*c.hx+c.gy*c.hy);
        bb+=w*(sq(c.hx)+sq(c.hy));ax-=w*(c.gx*c.x+c.gy*c.y);
        bx-=w*(c.hx*c.x+c.hy*c.y);ss+=w*(sq(c.x)+sq(c.y));
    }
    bool solve(double &u,double &h,double &chi,double &volume)const{
        double d=aa*bb-ab*ab;
        if(!(d>0.))return false;
        u=(bb*ax-ab*bx)/d;h=(aa*bx-ab*ax)/d;
        chi=std::max(0.,ss-u*ax-h*bx);
        volume=std::log1p(d*sq(.13*.33));
        return fabs(u)<.31&&fabs(h)<.82&&chi<21.;
    }
};
struct Trial {
    int row,night;
    double mag,q,t,noise;
    Cells pos;
};
Constraint constraint(const Trial &a,const Trial &b,const Trial &c){
    double fraction=(c.t-a.t)/(b.t-a.t),first=1.-fraction;
    auto d=[&](float Cells::*field){return c.pos.*field-first*(a.pos.*field)-fraction*(b.pos.*field);};
    // The tangent-plane guide is linearized about distance and radial
    // velocity. A longer extrapolation inherits both endpoint errors.
    double scale=std::max(1.,fabs(fraction));
    double err=std::hypot(.40,.055*std::sqrt(fabs(b.t-a.t)/10.));
    return {d(&Cells::x),d(&Cells::y),d(&Cells::gx),d(&Cells::gy),
            d(&Cells::hx),d(&Cells::hy),scale*err};
}
struct Rank {
    double merit=-1000.;int i=-1,j=-1;
    double gamma=0.;
};
void trim(std::vector<Seed> &v,int cap){
    if(int(v.size())<=cap)return;
    auto better=[](const Seed &a,const Seed &b){return a.score>b.score;};
    std::nth_element(v.begin(),v.begin()+cap,v.end(),better);v.resize(cap);
}
void cell(const Source *src,const View *vis,const Params &p,const Cells *xy,const int *order,
          int count,std::vector<Seed> &triples,std::vector<Seed> &quads){
    // A nearly stationary clump can contain many shutters of the same
    // night. Keep at most two representatives from that night in this cell.
    std::array<Trial,40> t;int n=0;
    std::array<int,150> choices;
    int length=std::min(count,int(choices.size()));
    for(int k=0;k<length;k++)choices[k]=order[k*count/length];
    if(length>16)std::sort(choices.begin(),choices.begin()+length,[&](int a,int b){return src[a].q>src[b].q;});
    int limit=length>22?22:length;
    for(int j=0;j<length&&n<limit;j++){
        int k=choices[j];auto &s=src[k];auto &e=vis[s.ex];
        if(length>12){
            int duplicates=0;
            for(int v=0;v<n;v++)duplicates+=int(t[v].night==e.night);
            if(duplicates>=2)continue;
        }
        t[n++]={s.row,e.night,s.mag,s.q,e.t,0.,xy[k]};
    }
    if(n<3)return;
    std::sort(t.begin(),t.begin()+n,[](const Trial &a,const Trial &b){return a.t<b.t;});
    if(t[n-1].t-t[0].t<26.)return;
    Rank by_size[2][3]{};
    Seed local[2][3]{};
    auto record=[&](int k,const Seed &s){
        int worst=0;
        for(int i=1;i<3;i++)if(by_size[k][i].merit<by_size[k][worst].merit)worst=i;
        if(s.score>by_size[k][worst].merit){by_size[k][worst].merit=s.score;local[k][worst]=s;}
    };
    for(int i=0;i<n-2;i++)for(int k=i+2;k<n;k++){
        auto &a=t[i],&b=t[k];double span=b.t-a.t;
        if(span<26.)continue;
        std::array<Constraint,40> c;std::array<int,40> use;int nc=0;
        for(int j=i+1;j<k;j++){
            if(t[j].night==a.night||t[j].night==b.night)continue;
            auto z=constraint(a,b,t[j]);
            // A cross product measures the component unreachable by
            // inverse distance even when the reference orbit is uncertain.
            double L=std::hypot(z.gx,z.gy);
            double det=z.hx*z.gy-z.hy*z.gx;
            if(fabs(z.x*z.gy-z.y*z.gx)>3.8*z.sig*(1.+L)+.83*fabs(det))continue;
            c[nc]=z;use[nc++]=j;
        }
        for(int j=0;j<nc;j++){
            auto &one=t[use[j]];
            Normal first;first.add(c[j]);double u,v,chi,vol;
            if(!first.solve(u,v,chi,vol))continue;
            double maxmag=std::max({a.mag,b.mag,one.mag}),minmag=std::min({a.mag,b.mag,one.mag});
            double quality=a.q+b.q+one.q;
            double base=.56*quality-.5*sq((maxmag-minmag)/.34)-.45*std::log(1.+n*.2);
            double preference=.46*std::log1p(span/70.);
            double score=base+preference-.45*chi-.40*vol;
            double gamma=std::exp(-u)/p.radius;
            record(0,{{a.row,one.row,b.row,-1},float(gamma),float(score)});
            for(int l=j+1;l<nc;l++){
                auto &two=t[use[l]];
                if(one.night==two.night)continue;
                Normal both=first;both.add(c[l]);
                if(!both.solve(u,v,chi,vol)||chi>17.)continue;
                double extent=std::max(maxmag,two.mag)-std::min(minmag,two.mag);
                double merit=.57*(quality+two.q)+preference-.46*chi-.40*vol
                             -.50*sq(extent/.34)-.85*std::log1p(n*.23);
                double confirmation=0.;int witness=-1;
                for(int h=0;h<nc;h++){
                    const auto &test=t[use[h]];
                    if(test.night==one.night||test.night==two.night)continue;
                    auto &z=c[h];
                    double rr=(sq(z.x+u*z.gx+v*z.hx)+sq(z.y+u*z.gy+v*z.hy))/sq(z.sig);
                    double support=5.7+.30*test.q-.27*rr;
                    if(rr<18.&&support>confirmation){confirmation=support;witness=test.row;}
                }
                record(1,{{a.row,one.row,two.row,b.row},float(std::exp(-u)/p.radius),float(merit+confirmation),witness});
            }
        }
    }
    for(int i=0;i<3;i++){
        if(by_size[0][i].merit>-999.)triples.push_back(local[0][i]);
        if(by_size[1][i].merit>-999.)quads.push_back(local[1][i]);
    }
}
}
#include "drift_search.hpp"
extern "C" void guide_host(const rotating::Source *src,int n,const rotating::View *vis,
                const rotating::Params *p,double px,double py,rotating::Cells *xy,rotating::Key *keys){
    for(int i=0;i<n;i++)keys[i]=rotating::transform(src[i],vis[src[i].ex],*p,px,py,xy[i]);
}
extern "C" int guide_groups(const rotating::Source *src,const rotating::View *vis,const rotating::Params *p,
        const rotating::Cells *xy,const int *order,const int *start,const int *length,int groups,
        rotating::Seed *out,int cap,int *size,double seconds){
    try{
        std::vector<rotating::Seed> tri,quad;
        auto end=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
        int g=0;for(;g<groups;g++){
            if((g%256)==0 && std::chrono::steady_clock::now()>end)break;
            rotating::cell(src,vis,*p,xy,order+start[g],length[g],tri,quad);
            if(int(tri.size())>3*cap)rotating::trim(tri,cap);
            if(int(quad.size())>3*cap)rotating::trim(quad,cap);
        }
        rotating::trim(tri,cap);rotating::trim(quad,cap);
        std::copy(tri.begin(),tri.end(),out);std::copy(quad.begin(),quad.end(),out+cap);
        size[0]=tri.size();size[1]=quad.size();return g;
    }catch(...){return -1;}
}
extern "C" int drift_groups(const rotating::Source *src,const rotating::View *vis,const rotating::Params *p,
        const rotating::Cells *xy,const int *order,const int *start,const int *length,int groups,
        rotating::Seed *out,int cap,int *size,double seconds,double east,double north){
    try{
        std::vector<rotating::Seed> tri,quad;
        auto end=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
        int g=0;for(;g<groups;g++){
            if((g%24)==0 && std::chrono::steady_clock::now()>end)break;
            rotating::drifting_cell(src,vis,*p,xy,order+start[g],length[g],east,north,tri,quad);
            if(int(tri.size())>3*cap)rotating::trim(tri,cap);
            if(int(quad.size())>3*cap)rotating::trim(quad,cap);
        }
        rotating::trim(tri,cap);rotating::trim(quad,cap);
        std::copy(tri.begin(),tri.end(),out);std::copy(quad.begin(),quad.end(),out+cap);
        size[0]=tri.size();size[1]=quad.size();return g;
    }catch(...){return -1;}
}
#endif
