#ifndef __CUDACC__
#include <cmath>
struct Id {int x=0,y=0,z=0;};
static Id blockIdx,blockDim,threadIdx;
#define __global__
#define __device__
#endif
#include "quartet_fit.hpp"
namespace raw_chord {
struct Pixel {float x,y,mag,sigma,shape;unsigned id;};
struct Pair {float pos[4],mag,quality;int a,b;float raw[4];int ea,eb;};
struct Model {double center[4],transform[4][4],poly[15][4],first[3],slope[2][2],last[3];};
struct Visit {double dx,dy,scale;int e,pad;};
struct Seed {int rows[4];float gamma,score;};
struct Peak {float x,y,rawx,rawy,quality;int row;};
__device__ inline int hi(int a,int b){return a>b?a:b;}
__device__ inline int lo(int a,int b){return a<b?a:b;}
__device__ inline double sq(double x){return x*x;}
template<int slots>
__device__ inline int peaks(const Pixel *points,const int *starts,const int *box,const float *density,
                  const Visit &probe,const double *v,double mx,double my,double mag,double color,Peak *out){
    double x=(mx-probe.dx)/probe.scale,y=(my-probe.dy)/probe.scale;
    double xx=v[0],xy=v[1],yy=v[2],det=xx*yy-xy*xy;
    if(!(det>1.e-6)||xx+yy>1.1e8||box[2]<=0)return 0;
    double area=sqrt(det);int b=hi(0,lo(79,int((mag-10.)*4.))),tier=0;
    if(area*density[b]>4.){
        tier=1;if(area*density[80+b]>8.)tier=2;
        if(area*density[80*tier+b]>50.)return 0;
    }
    double slope=xy/yy,half=3.12*sqrt(fmax(.001,xx-xy*slope)),height=3.12*sqrt(yy);
    int lower=hi(0,int(floor((y-height)/24.))-box[1]),upper=lo(box[3]-1,int(floor((y+height)/24.))-box[1]);
    int n=0,worst=0;
    for(int j=lower;j<=upper;j++){
        double bot=fmax(-height,(box[1]+j)*24.-y),top=fmin(height,(box[1]+j+1)*24.-y);
        double u=x+bot*slope,w=x+top*slope;
        int xl=hi(0,int(floor((fmin(u,w)-half)/24.))-box[0]);
        int xh=lo(box[2]-1,int(floor((fmax(u,w)+half)/24.))-box[0]);
        if(xh<xl)continue;
        const int cells=box[2]*box[3]+1;
        for(int ix=xl;ix<=(tier?xh:xl);ix++){
          int cell=j*box[2]+ix,start=starts[cell];
          int end=tier?start+starts[tier*cells+cell]:starts[j*box[2]+xh+1];
          for(int k=start;k<end;k++){
            const Pixel &p=points[k];if(p.id>>31||int((p.id>>29)&3)<tier)continue;
            double dm=p.mag-mag,var=.27*.27+color*color+sq(p.sigma);
            if(!(dm*dm<2.95*2.95*var))continue;
            double dx=p.x-x,dy=p.y-y,chi=(yy*dx*dx+xx*dy*dy-2.*xy*dx*dy)/det;
            if(!(chi<9.7))continue;
            float q=float(-.39*chi-.48*dm*dm/var-p.shape+.67*((p.id>>29)&3));
            if(n==slots&&q<=out[worst].quality)continue;
            int at=n<slots?n++:worst;
            out[at]={float(p.x*probe.scale+probe.dx),float(p.y*probe.scale+probe.dy),p.x,p.y,q,int(p.id&0x1fffffffu)};
            if(at==worst){for(int i=0;i<n;i++)if(out[i].quality<out[worst].quality)worst=i;}
            else if(q<out[worst].quality)worst=at;
          }
        }
    }
    return n;
}
}
extern "C" __global__ void raw_chord_query(const raw_chord::Pair *pairs,int n,const raw_chord::Model *model,
        const raw_chord::Visit *probes,const raw_chord::Pixel *p3,const int *s3,const int *b3,const float *d3,
        const raw_chord::Pixel *p4,const int *s4,const int *b4,const float *d4,
        const quartet_fit::Shutter *shutters,double gamma,double color3,double color4,raw_chord::Seed *out,float *score){
    using namespace raw_chord;
    int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);if(i>=n)return;
    constexpr int slots=3;
    for(int k=0;k<slots;k++)score[i*slots+k]=-1.e20f;
    const Pair &p=pairs[i];const Model &m=*model;
    double z[4]{},v[4];
    for(int k=0;k<4;k++)for(int j=0;j<4;j++)z[k]+=(p.pos[j]-m.center[j])*m.transform[j][k];
    for(int k=0;k<4;k++){
        v[k]=m.poly[0][k];for(int j=0;j<4;j++)v[k]+=z[j]*m.poly[j+1][k];
        int ii=5;for(int j=0;j<4;j++)for(int l=j;l<4;l++,ii++)v[k]+=z[j]*z[l]*m.poly[ii][k];
    }
    Peak middle[16],last[9];int used=0,worst=0;Seed best[slots];
    int cnt=peaks<16>(p3,s3,b3,d3,probes[0],m.first,v[0],v[1],p.mag,color3,middle);
    for(int j=0;j<cnt;j++){
        auto &q=middle[j];double dx=q.x-v[0],dy=q.y-v[1];
        double x=v[2]+dx*m.slope[0][0]+dy*m.slope[1][0],y=v[3]+dx*m.slope[0][1]+dy*m.slope[1][1];
        int count=peaks<9>(p4,s4,b4,d4,probes[1],m.last,x,y,p.mag,color4,last);
        for(int k=0;k<count;k++){
            auto &r=last[k];int ei[]={p.ea,p.eb,probes[0].e,probes[1].e},idx[]={p.a,p.b,q.row,r.row};
            float xx[]={p.raw[0],p.raw[2],q.rawx,r.rawx},yy[]={p.raw[1],p.raw[3],q.rawy,r.rawy};int order[]={0,1,2,3};
            for(int u=1;u<4;u++)for(int w=u;w>0 && shutters[ei[order[w]]].t<shutters[ei[order[w-1]]].t;w--){
                int save=order[w];order[w]=order[w-1];order[w-1]=save;
            }
            int a=order[0],b=order[1],c=order[2],d=order[3];
            float aa[]={xx[a],yy[a],xx[b]-xx[a],yy[b]-yy[a]},bb[]={xx[c],yy[c],xx[d]-xx[c],yy[d]-yy[c]};
            quartet_fit::Shutter e[]={shutters[ei[a]],shutters[ei[b]],shutters[ei[c]],shutters[ei[d]]};
            quartet_fit::State orbit;double rms;if(!quartet_fit::solve(aa,bb,e,gamma,orbit,rms))continue;
            float merit=float(p.quality+q.quality+r.quality-1.7*sq(rms/.22));
            if(used==slots&&merit<=best[worst].score)continue;
            int at=used<slots?used++:worst;
            best[at]={{idx[a],idx[b],idx[c],idx[d]},float(orbit.g),merit};
            if(at==worst){for(int h=0;h<used;h++)if(best[h].score<best[worst].score)worst=h;}
            else if(merit<best[worst].score)worst=at;
        }
    }
    for(int k=0;k<used;k++){out[i*slots+k]=best[k];score[i*slots+k]=best[k].score;}
}
#ifndef __CUDACC__
extern "C" void raw_chord_item(int i){blockIdx.x=i;blockDim.x=1;threadIdx.x=0;}
#endif
