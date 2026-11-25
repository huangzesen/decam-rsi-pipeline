// Joint statistical ranging on two independent visits. Positions and their
// first and second range derivatives are evaluated once per endpoint pair,
// then shared by all candidate associations in that pair's CUDA block.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define VISIT_GPU
#define VF __device__ __forceinline__
#else
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>
#define VF inline
#endif
namespace visit_device {
constexpr double arc=206264.80624709636,step=96./arc;
constexpr int lanes=64;
struct V{double x,y;};
VF V add(V a,V b){return {a.x+b.x,a.y+b.y};}
VF V sub(V a,V b){return {a.x-b.x,a.y-b.y};}
VF V mul(V a,double b){return {a.x*b,a.y*b};}
VF double dot(V a,V b){return a.x*b.x+a.y*b.y;}
VF double sqr(double x){return x*x;}
VF int lower(int a,int b){return a<b?a:b;}
VF int upper(int a,int b){return a>b?a:b;}
VF double clip(double x,double lo,double hi){return fmax(lo,fmin(hi,x));}
struct Sky {
    const double *ex;
    const int *grid,*edges;
    const float *points,*density;
    const int *row;
    int ne;
};
struct Obs {
    float dx,dy,gx,gy,hx,hy,ggx,ggy,hhx,hhy,ghx,ghy,l,r,noise,score,photo,logrho;
    int row,night;
};
struct Best {double score,u,v;int c,d,fifth;};
VF Best empty(){return {-1.e20,0.,0.,-1,-1,-1};}
VF V propagate(const double *m,const double *e,double t,double &f,double &s){
    double tt=t*t,z=m[6]*tt,w=m[7]*t,ww=w*w,kt=m[8]*tt;
    f=1.-.5*z*(1.-w)+z*(3.*kt-2.*z-15.*ww)/24.;
    s=t*(1.-z/6.+z*w/4.+z*(9.*kt-8.*z-45.*ww)/120.);
    double iz=1./(f+s*m[5]-m[4]*e[3]);
    return {(m[0]*f+s*m[2]-m[4]*e[1])*iz,(m[1]*f+s*m[3]-m[4]*e[2])*iz};
}
VF void update(Obs *top,Obs value){
    if(value.score>top[0].score){top[1]=top[0];top[0]=value;}
    else if(value.score>top[1].score)top[1]=value;
}
VF void inspect(const Sky &sky,const double *m,int j,Obs *top){
    const double *a=m+54,*e=sky.ex+12*j;
    int night=int(e[6]);
    if(night==int(a[9])||night==int(a[10])||e[4]+.95<a[7]||e[7]>e[8])return;
    double t=e[0]-a[0],f,s;
    V c=propagate(m,e,t,f,s);
    double frac=t/a[1],pad=.029+.000046*fabs(t)+.00010*fabs(frac);
    if(c.x+pad<e[7]||c.x-pad>e[8]||c.y+pad<e[9]||c.y-pad>e[10])return;
    double l=(1.-frac)*a[4],r=frac*a[5];
    V up=propagate(m+9,e,t,f,s),down=propagate(m+18,e,t,f,s);
    V hg=propagate(m+27,e,t,f,s),hl=propagate(m+36,e,t,f,s);
    V gh=propagate(m+45,e,t,f,s);
    V g=mul(sub(up,down),.5*arc),h=mul(sub(hg,hl),.5*arc);
    V gg=mul(sub(add(up,down),mul(c,2.)),.5*arc),hh=mul(sub(add(hg,hl),mul(c,2.)),.5*arc);
    gh=mul(add(sub(sub(gh,up),hg),c),arc);
    double ph=fmax(fabs(t),fabs(t-a[1]))*sqrt(m[6]),err=.235+.0014*arc*ph*ph*ph*ph*ph;
    double noise=err*err+1.e-5;
    double base=noise+l*l+r*r;
    double xx=base+sqr(g.x)+sqr(h.x)+2.*sqr(gg.x)+2.*sqr(hh.x)+sqr(gh.x);
    double yy=base+sqr(g.y)+sqr(h.y)+2.*sqr(gg.y)+2.*sqr(hh.y)+sqr(gh.y);
    double xy=g.x*g.y+h.x*h.y+2.*gg.x*gg.y+2.*hh.x*hh.y+gh.x*gh.y;
    double det=xx*yy-xy*xy;
    if(!(det>0.)||det>2.e17)return;
    int mb=upper(0,lower(47,int((a[7]-11.)*2.)));
    double area=sqrt(det);
    int layer=j*2+(area*sky.density[j*96+mb]>.85);
    double rho=fmax(1.e-8,double(sky.density[layer*48+mb]));
    double clutter=rho*area;
    if(clutter>95.)return;
    const int *config=sky.grid+5*layer;
    if(!config[2])return;
    double width=fmin(3.1,sqrt(74./(.005+clutter)));
    double wy=width*sqrt(yy)/arc;
    int yl=upper(0,int(floor((c.y-wy)/step))-config[1]);
    int yh=lower(config[3]-1,int(floor((c.y+wy)/step))-config[1]);
    if(yh<yl)return;
    double tilt=xy/yy,thick=width*sqrt(det/yy)/arc;
    double phi=.052+sqr(.30*(e[5]-a[8]));
    double bias=-.48*log(.018+clutter)-.5*log(phi);
    double lr=-log(rho);
    Obs v;v.gx=g.x;v.gy=g.y;v.hx=h.x;v.hy=h.y;v.ggx=gg.x;v.ggy=gg.y;v.hhx=hh.x;v.hhy=hh.y;
    v.ghx=gh.x;v.ghy=gh.y;v.l=l;v.r=r;v.noise=noise;v.night=night;v.logrho=lr;
    for(int y=yl;y<=yh;y++){
        double dy0=fmax(-wy,(y+config[1])*step-c.y),dy1=fmin(wy,(y+config[1]+1)*step-c.y);
        double left=c.x+fmin(tilt*dy0,tilt*dy1)-thick,right=c.x+fmax(tilt*dy0,tilt*dy1)+thick;
        int lo=upper(0,int(floor(left/step))-config[0]),hi=lower(config[2]-1,int(floor(right/step))-config[0]);
        if(hi<lo)continue;
        int z=config[4]+y*config[2];
        for(int k=sky.edges[z+lo];k<sky.edges[z+hi+1];k++){
            const float *p=sky.points+4*k;
            double dm=p[2]-a[7],phot=dm*dm/phi;
            if(phot>12.)continue;
            double dx=(p[0]-c.x)*arc,dy=(p[1]-c.y)*arc;
            double chi=dy*dy/yy+sqr(dx-tilt*dy)*yy/det;
            if(chi>width*width)continue;
            double mark=.39*clip(p[3],-8.,13.)-.45*phot;
            double score=mark+bias-.40*chi;
            if(score<=top[1].score)continue;
            v.row=sky.row[k];v.dx=dx;v.dy=dy;v.photo=mark;v.score=score;
            update(top,v);
        }
    }
}
VF V index_at(const Obs &o,double u,double v){
    return {u*o.gx+v*o.hx+u*u*o.ggx+v*v*o.hhx+u*v*o.ghx,
            u*o.gy+v*o.hy+u*u*o.ggy+v*v*o.hhy+u*v*o.ghy};
}
VF Best compare(const Obs &a,const Obs &b){
    Best out=empty();
    if(a.row<0||b.row<0||a.night==b.night)return out;
    double va=a.noise+sqr(a.l)+sqr(a.r),vb=b.noise+sqr(b.l)+sqr(b.r);
    double shared=(a.l*b.l+a.r*b.r)/va;
    double wb=1./fmax(double(b.noise),vb-va*shared*shared),wa=1./va;
    V oa{a.dx,a.dy},ob=sub({b.dx,b.dy},mul(oa,shared));
    double u=0.,v=0.,det=0.,chi=0.;
    for(int k=0;k<3;k++){
        V ga{a.gx+2.*u*a.ggx+v*a.ghx,a.gy+2.*u*a.ggy+v*a.ghy};
        V ha{a.hx+2.*v*a.hhx+u*a.ghx,a.hy+2.*v*a.hhy+u*a.ghy};
        V gb=sub({b.gx+2.*u*b.ggx+v*b.ghx,b.gy+2.*u*b.ggy+v*b.ghy},mul(ga,shared));
        V hb=sub({b.hx+2.*v*b.hhx+u*b.ghx,b.hy+2.*v*b.hhy+u*b.ghy},mul(ha,shared));
        V ma=index_at(a,u,v),mb=sub(index_at(b,u,v),mul(ma,shared));
        V ra=sub(oa,ma),rb=sub(ob,mb);
        double aa=1.+wa*dot(ga,ga)+wb*dot(gb,gb),bb=1.+wa*dot(ha,ha)+wb*dot(hb,hb);
        double ab=wa*dot(ga,ha)+wb*dot(gb,hb),ar=wa*dot(ra,ga)+wb*dot(rb,gb)-u;
        double br=wa*dot(ra,ha)+wb*dot(rb,hb)-v;
        det=aa*bb-ab*ab;
        if(!(det>0.))return out;
        double du=(ar*bb-br*ab)/det,dv=(br*aa-ar*ab)/det;
        u+=du;v+=dv;
        if(fabs(u)>4.||fabs(v)>4.)return out;
        if(fabs(du)+fabs(dv)<.0005)break;
    }
    V ma=index_at(a,u,v),mb=sub(index_at(b,u,v),mul(ma,shared));
    V ra=sub(oa,ma),rb=sub(ob,mb);
    chi=u*u+v*v+wa*dot(ra,ra)+wb*dot(rb,rb);
    if(chi>24.)return out;
    out.u=u;out.v=v;out.c=a.row;out.d=b.row;
    out.score=a.photo+b.photo-.49*chi-.5*log(det/(wa*wb))+a.logrho+b.logrho;
    return out;
}
VF void winner(Best *best,Best trial){
    if(trial.c<0)return;
    if(trial.score>best[0].score){best[1]=best[0];best[0]=trial;}
    else if(trial.score>best[1].score)best[1]=trial;
}
VF V gradient_g(const Obs &p,double u,double v){return {p.gx+2.*u*p.ggx+v*p.ghx,p.gy+2.*u*p.ggy+v*p.ghy};}
VF V gradient_h(const Obs &p,double u,double v){return {p.hx+2.*v*p.hhx+u*p.ghx,p.hy+2.*v*p.hhy+u*p.ghy};}
VF void witness(Best &best,const Obs *items,int n,const Obs &a,const Obs &b){
    if(best.c<0)return;
    double u=best.u,v=best.v;
    double va=a.noise+sqr(a.l)+sqr(a.r),vb=b.noise+sqr(b.l)+sqr(b.r);
    double ab=a.l*b.l+a.r*b.r,ratio=ab/va,wb=1./fmax(double(b.noise),vb-ratio*ab);
    V ga=gradient_g(a,u,v),ha=gradient_h(a,u,v),gb=gradient_g(b,u,v),hb=gradient_h(b,u,v);
    V dg=sub(gb,mul(ga,ratio)),dh=sub(hb,mul(ha,ratio));
    double gg=1.+dot(ga,ga)/va+wb*dot(dg,dg),hh=1.+dot(ha,ha)/va+wb*dot(dh,dh);
    double gh=dot(ga,ha)/va+wb*dot(dg,dh),det=gg*hh-gh*gh;
    if(!(det>0.))return;
    V ra=sub({a.dx,a.dy},index_at(a,u,v)),rb=sub({b.dx,b.dy},index_at(b,u,v));
    double extra=0.;
    for(int i=0;i<n;i++){
        const auto &p=items[i];
        if(p.row<0||p.night==a.night||p.night==b.night)continue;
        double pa=p.l*a.l+p.r*a.r,pb=p.l*b.l+p.r*b.r;
        double cb=wb*(pb-pa*ratio),ca=pa/va-cb*ratio;
        V r=sub(sub(sub({p.dx,p.dy},index_at(p,u,v)),mul(ra,ca)),mul(rb,cb));
        V g=sub(sub(gradient_g(p,u,v),mul(ga,ca)),mul(gb,cb));
        V h=sub(sub(gradient_h(p,u,v),mul(ha,ca)),mul(hb,cb));
        double noise=fmax(double(p.noise),p.noise+sqr(p.l)+sqr(p.r)-ca*pa-cb*pb)+.018;
        double xx=noise+(hh*g.x*g.x-2.*gh*g.x*h.x+gg*h.x*h.x)/det;
        double yy=noise+(hh*g.y*g.y-2.*gh*g.y*h.y+gg*h.y*h.y)/det;
        double xy=(hh*g.x*g.y-gh*(g.x*h.y+g.y*h.x)+gg*h.x*h.y)/det;
        double area2=xx*yy-xy*xy;
        if(!(area2>0.))continue;
        double chi=(yy*r.x*r.x-2.*xy*r.x*r.y+xx*r.y*r.y)/area2;
        if(chi>13.)continue;
        double support=p.logrho-1.837877-.5*log(area2)-.5*chi+.48*p.photo-2.7;
        if(support>extra){extra=support;best.fifth=p.row;}
    }
    best.score+=fmin(8.,extra);
}
VF void write(const Best &b,const double *m,int *ids,float *val){
    ids[0]=b.c;ids[1]=b.d;ids[2]=b.fifth;val[0]=b.score+.37*m[65];val[1]=m[60]+m[56]*b.u;
    val[2]=m[5]+m[57]*b.v;
}
}
#ifdef VISIT_GPU
extern "C" __global__ void visit_ranging(const double *model,const double *ex,const int *grid,const int *edges,
        const float *points,const float *rho,const int *row,int ne,int count,int *ids,float *val){
    using namespace visit_device;
    int lane=threadIdx.x,i=blockIdx.x;if(i>=count)return;
    Sky sky{ex,grid,edges,points,rho,row,ne};const double *m=model+68*i;
    __shared__ Obs records[2*lanes];
    __shared__ int warp_counts[2];
    __shared__ Best wins[lanes];
    Obs top[2]{};top[0].score=top[1].score=-1.e18f;top[0].row=top[1].row=-1;
    for(int e=lane;e<ne;e+=lanes)inspect(sky,m,e,top);
    int count_obs=(top[0].row>=0)+(top[1].row>=0),prefix=count_obs;
    for(int step=1;step<32;step*=2){
        int prev=__shfl_up_sync(0xffffffffu,prefix,step);
        if((lane%32)>=step)prefix+=prev;
    }
    if(lane%32==31)warp_counts[lane/32]=prefix;
    __syncthreads();
    int first=prefix-count_obs+(lane>=32?warp_counts[0]:0),n=warp_counts[0]+warp_counts[1];
    for(int j=0;j<count_obs;j++)records[first+j]=top[j];
    __syncthreads();
    Best best[2]={empty(),empty()};
    for(int a=lane;a<n;a+=lanes)for(int b=0;b<a;b++){
        auto trial=compare(records[a],records[b]);
        if(trial.c>=0&&trial.score+8.>best[1].score){
            witness(trial,records,n,records[a],records[b]);winner(best,trial);
        }
    }
    wins[lane]=best[0];__syncthreads();
    // Preserve four separate groups of combinations in each block.
    for(int s=8;s;s/=2){
        if((lane%16)<s&&wins[lane+s].score>wins[lane].score)wins[lane]=wins[lane+s];
        __syncthreads();
    }
    if(lane%16==0){int at=4*i+lane/16;write(wins[lane],m,ids+3*at,val+3*at);}
}
#else
extern "C" void visit_ranging_host(const double *model,const double *ex,const int *grid,const int *edges,
        const float *points,const float *rho,const int *row,int ne,int count,int *ids,float *val,int threads){
    using namespace visit_device;
    Sky sky{ex,grid,edges,points,rho,row,ne};
    std::atomic<int> next{0};std::vector<std::thread> workers;
    for(int t=0;t<threads;t++)workers.emplace_back([&]{
        std::vector<Obs> items;items.reserve(2*lanes);
        for(;;){
            int i=next.fetch_add(1);if(i>=count)break;
            const auto *m=model+68*i;items.clear();
            for(int lane=0;lane<lanes;lane++){
                Obs top[2]{};top[0].score=top[1].score=-1.e18f;top[0].row=top[1].row=-1;
                for(int e=lane;e<ne;e+=lanes)inspect(sky,m,e,top);
                for(int j=0;j<2;j++)if(top[j].row>=0)items.push_back(top[j]);
            }
            Best groups[4][2];for(auto &group:groups)group[0]=group[1]=empty();
            for(int a=0;a<int(items.size());a++)for(int b=0;b<a;b++){
                auto trial=compare(items[a],items[b]);auto *keep=groups[(a%lanes)/16];
                if(trial.c>=0&&trial.score+8.>keep[1].score){
                    witness(trial,items.data(),items.size(),items[a],items[b]);winner(keep,trial);
                }
            }
            for(int j=0;j<4;j++){int at=4*i+j;write(groups[j][0],m,ids+3*at,val+3*at);}
        }
    });
    for(auto &worker:workers)worker.join();
}
#endif
