// Runtime-compiled CUDA. The host fixture calls the same kernel entry points.
#ifndef __CUDACC__
#include <algorithm>
#include <cmath>
#include <cstdint>
using std::isfinite;
struct Coord {int x=0,y=0,z=0;};
static Coord blockIdx,blockDim,threadIdx;
#define __global__
#define __device__
static inline int atomicExch(int *p,int v){int old=*p;*p=v;return old;}
static inline int atomicAdd(int *p,int v){int old=*p;*p+=v;return old;}
#endif
#include "quartet_fit.hpp"
namespace qdevice {
struct Point {float x,y,mag,sigma,quality;int row;float rawx,rawy;int exposure;};
struct Field {float x0,width;int nx,first,start,count;};
struct Pair {int a,b;float quality;};
struct Transform {double center[4],linear[4][4],p[15][4];};
struct Plan {Transform model[2];quartet_fit::Shutter ex[4];double gamma,color;};
struct Node {float z[4];int next;};
struct Seed {int row[4];float gamma,score;};
struct Route {double center[4],offset[4],slope[4][4],radius[2];};
struct Tile {int x,y,nx,ny,start;};
__device__ inline int limit(int x,int lo,int hi){return x<lo?lo:x>hi?hi:x;}
__device__ inline double sq(double x){return x*x;}
__device__ inline int bin(double v){return int(floor(fmin(1.e8,fmax(-1.e8,v))*.25));}
__device__ inline unsigned hash(int x,int y){
    unsigned z=unsigned(x)*73856093u^unsigned(y)*19349663u;
    z^=z>>16;z*=0x7feb352du;return z^(z>>15);
}
__device__ inline void measured(const Point *p,const Pair &a,float *s){
    const auto &x=p[a.a],&y=p[a.b];s[0]=x.x;s[1]=x.y;s[2]=y.x-x.x;s[3]=y.y-x.y;
}
__device__ inline void transform(const float *pos,const Transform &m,float *out){
    double s[4]={},w[4];
    for(int i=0;i<4;i++){
        double d=pos[i]-m.center[i];
        for(int k=0;k<4;k++)s[k]+=d*m.linear[i][k];
    }
    for(int k=0;k<4;k++)w[k]=m.p[0][k];
    for(int i=0;i<4;i++)for(int k=0;k<4;k++)w[k]+=s[i]*m.p[1+i][k];
    int t=5;for(int i=0;i<4;i++)for(int j=i;j<4;j++,t++){
        double d=s[i]*s[j];for(int k=0;k<4;k++)w[k]+=d*m.p[t][k];
    }
    for(int k=0;k<4;k++)out[k]=float(w[k]);
}
}
extern "C" __global__ void exposure_pairs(const qdevice::Point* points,const qdevice::Field* fields,
             const int* starts,int a,int b,const double *gate,double color,int per,
             int *members,float *score){
    using namespace qdevice;
    int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    Field fa=fields[a],fb=fields[b];if(i>=fa.count)return;
    int ai=fa.first+i;
    const auto &p=points[ai];
    float best[24];int bestid[24],used=0,worst=0;
    double x=p.x-gate[0],y=p.y-gate[1];
    double mx=p.x+gate[2]+x*gate[4]+y*gate[6],my=p.y+gate[3]+x*gate[5]+y*gate[7];
    double xx=gate[8],slope=gate[9]/xx,sy=sqrt(fmax(.0001,gate[10]-gate[9]*slope));
    double reach=3.15*sqrt(xx);
    int lo=limit(int(floor((mx-reach-fb.x0)/fb.width)),0,fb.nx-1);
    int hi=limit(int(floor((mx+reach-fb.x0)/fb.width)),0,fb.nx-1);
    if(fb.count>0 && mx+reach>=fb.x0 && mx-reach<=fb.x0+fb.width*fb.nx){
        for(int ix=lo;ix<=hi;ix++){
            double xl=fmax(-reach,fb.x0+ix*fb.width-mx),xh=fmin(reach,fb.x0+(ix+1)*fb.width-mx);
            double yl=my+fmin(slope*xl,slope*xh)-3.15*sy,yh=my+fmax(slope*xl,slope*xh)+3.15*sy;
            int l=starts[fb.start+ix],h=starts[fb.start+ix+1],end=h;
            while(l<h){int mid=l+(h-l)/2;if(points[mid].y<yl)l=mid+1;else h=mid;}
            for(int j=l;j<end&&points[j].y<=yh;j++){
                const auto &v=points[j];double dx=v.x-mx,dy=v.y-my;
                double chi=dx*dx/xx+sq((dy-slope*dx)/sy);
                if(chi>3.15*3.15)continue;
                double var=sq(p.sigma)+sq(v.sigma)+.036+sq(.28*color),dm=sq(p.mag-v.mag)/var;
                if(dm>12.25)continue;
                float value=float(.54*(p.quality+v.quality)-.34*chi-.44*dm);
                if(used<per){bestid[used]=j;best[used]=value;if(value<best[worst])worst=used;used++;}
                else if(value>best[worst]){
                    bestid[worst]=j;best[worst]=value;worst=0;
                    for(int k=1;k<per;k++)if(best[k]<best[worst])worst=k;
                }
            }
        }
    }
    for(int k=0;k<per;k++){
        int s=i*per+k;members[2*s]=ai;members[2*s+1]=k<used?bestid[k]:-1;
        score[s]=k<used?best[k]:-1.e20f;
    }
}

extern "C" __global__ void companion_footprints(const qdevice::Point *points,const qdevice::Pair *a,
        const qdevice::Pair *b,int count,const qdevice::Route *routes,const int *visits,
        const qdevice::Tile *tiles,const unsigned char *grid,float *scores){
    using namespace qdevice;
    int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x),task=int(blockIdx.y),role=int(blockIdx.z);
    if(i>=count)return;
    const Pair &p=(role?b:a)[task*count+i];int idx=(2*task+role)*count+i;scores[idx]=-1.e20f;
    if(p.b<0)return;
    const Route &r=routes[2*task+role];float pos[4];measured(points,p,pos);
    for(int k=0;k<2;k++){
        int e=visits[4*task+(role?0:2)+k];
        const auto &t=tiles[e];if(t.nx==0)return;if(t.nx<0)continue;
        double x=r.offset[2*k],y=r.offset[2*k+1];
        for(int j=0;j<4;j++){
            double d=double(pos[j])-r.center[j];x+=d*r.slope[j][2*k];y+=d*r.slope[j][2*k+1];
        }
        int bx=int(floor(x/240.))-t.x,by=int(floor(y/240.))-t.y;
        int ix=limit(bx,0,t.nx-1),iy=limit(by,0,t.ny-1);
        double outside=sqrt(sq(double(bx)-ix)+sq(double(by)-iy));
        double distance=.25*grid[t.start+ix*t.ny+iy];
        if(240.*fmax(outside,distance-outside)>r.radius[k])return;
    }
    scores[idx]=p.quality;
}

extern "C" __global__ void chord_cells(const qdevice::Point* points,const qdevice::Pair *b,
            int count,const qdevice::Plan *plans,int *heads,int cells,qdevice::Node *node){
    using namespace qdevice;
    int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x),task=int(blockIdx.y);
    if(i>=count)return;
    int j=task*count+i;const Pair &p=b[j];if(p.b<0)return;
    float v[4];measured(points,p,v);
    transform(v,plans[task].model[1],node[j].z);
    unsigned k=hash(bin(node[j].z[0]),bin(node[j].z[1]))&unsigned(cells-1);
    node[j].next=atomicExch(&heads[task*cells+k],j);
}
extern "C" __global__ void chord_query(const qdevice::Point* p,const qdevice::Pair *a,const qdevice::Pair *b,
            int count,const qdevice::Plan *plans,const int *heads,int cells,const qdevice::Node *node,
            qdevice::Seed *output,int *used,int cap,const quartet_fit::Shutter *shutter,
            qdevice::Seed *best_rows,float *best_scores){
    using namespace qdevice;
    int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x),task=int(blockIdx.y);
    if(i>=count)return;
    int entry=task*count+i;best_scores[entry]=-1.e20f;
    const Pair &left=a[entry];if(left.b<0)return;
    const Plan &fit=plans[task];
    float w[4],z[4];measured(p,left,w);transform(w,fit.model[0],z);
    double mag=.5*(p[left.a].mag+p[left.b].mag);
    double var=.25*(sq(p[left.a].sigma)+sq(p[left.b].sigma))+.083+sq(.28*fit.color);
    int bx=bin(z[0]),by=bin(z[1]);Seed best{};best.score=-1.e20f;
    for(int x=bx-1;x<=bx+1;x++)for(int y=by-1;y<=by+1;y++){
        unsigned h=hash(x,y)&unsigned(cells-1);
        for(int j=heads[task*cells+h];j>=0;j=node[j].next){
            const Node &n=node[j];
            if(bin(n.z[0])!=x||bin(n.z[1])!=y)continue;
            double chi=0.;for(int k=0;k<4;k++)chi+=sq(z[k]-n.z[k]);if(chi>16.)continue;
            const Pair &right=b[j];
            double dm=mag-.5*(p[right.a].mag+p[right.b].mag);
            double phot=dm*dm/(var+.25*(sq(p[right.a].sigma)+sq(p[right.b].sigma)));
            if(phot>11.)continue;
            const auto &l0=p[left.a],&l1=p[left.b],&r0=p[right.a],&r1=p[right.b];
            float al[]={l0.rawx,l0.rawy,l1.rawx-l0.rawx,l1.rawy-l0.rawy};
            float br[]={r0.rawx,r0.rawy,r1.rawx-r0.rawx,r1.rawy-r0.rawy};
            quartet_fit::Shutter epochs[]={shutter[l0.exposure],shutter[l1.exposure],
                                           shutter[r0.exposure],shutter[r1.exposure]};
            quartet_fit::State state;double rms;
            if(!quartet_fit::solve(al,br,epochs,fit.gamma,state,rms))continue;
            int slot=atomicAdd(used+task,1);
            Seed seed={{l0.row,l1.row,r0.row,r1.row},float(state.g),
                       float(left.quality+right.quality-.32*chi-.48*phot-1.6*sq(rms/.22))};
            if(slot<cap)output[task*cap+slot]=seed;
            if(seed.score>best.score)best=seed;
        }
    }
    best_rows[entry]=best;best_scores[entry]=best.score;
}
#ifndef __CUDACC__
extern "C" void quartet_thread(int x,int y){blockIdx.x=x;blockIdx.y=y;blockDim.x=1;threadIdx.x=0;}
extern "C" void quartet_role(int z){blockIdx.z=z;}
#endif
