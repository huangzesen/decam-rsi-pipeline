// Tiled four-direction orbit comparison on the host or CUDA device.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CHORD_GPU
#define FN __device__ __forceinline__
#else
#include <cmath>
#define FN inline
#endif
namespace chord_device {
constexpr double arc=206264.80624709636,gravity=.01720209895,gr=1./44.;
constexpr int block=128;
struct V {double x,y;};
FN V add(V a,V b){return {a.x+b.x,a.y+b.y};}
FN V sub(V a,V b){return {a.x-b.x,a.y-b.y};}
FN V mul(V a,double t){return {a.x*t,a.y*t};}
FN double dot(V a,V b){return a.x*b.x+a.y*b.y;}
FN double sq(double x){return x*x;}
FN V vec(const float *a,int off){return {a[off],a[off+1]};}
FN double speed(double g){return gravity*g*sqrt(g);}
struct Normal {
    double aa=1.,ab=0.,bb=1.,ay=0.,by=0.,cost=0.;
    FN void measure(double r,double a,double b,double weight){
        aa+=a*a*weight;ab+=a*b*weight;bb+=b*b*weight;
        ay+=a*r*weight;by+=b*r*weight;cost+=r*r*weight;
    }
};
FN bool overlaps(const float *a,const float *group,double ex,double ey){
    double dt=group[0]-a[0],days=fabs(dt);
    if(days>4900.+group[1])return false;
    double g=a[12],phase=speed(g)*dt,dz=fabs(g-gr)+a[13];
    V velocity=vec(a,14),shift=mul(velocity,dt*(1.+phase*phase/3.));
    V obs=vec(group,2),D=vec(a,6),p=sub(add(add(vec(a,2),mul(D,g)),shift),mul(obs,g-gr));
    V axis=mul(sub(add(D,mul(vec(a,16),dt)),obs),a[13]);
    V drift=mul({ex,ey},speed(gr)*arc),margin=mul(sub(velocity,drift),group[1]+.1);
    double thick=24.+days*(a[18]+.035)+.86*a[19]*days*(g*arc+sqrt(dot(shift,shift)));
    double floor=sq(thick)+sq(dot(margin,{-ey,ex}));
    double xx=floor+sq(axis.x)+sq(margin.x)+sq(dz*group[4]);
    double yy=floor+sq(axis.y)+sq(margin.y)+sq(dz*group[5]);
    return p.x+3.7*sqrt(xx)>=group[6]&&p.x-3.7*sqrt(xx)<=group[7]
        &&p.y+3.7*sqrt(yy)>=group[8]&&p.y-3.7*sqrt(yy)<=group[9];
}
FN double compare(const float *a,const float *b,const int *na,const int *nb,
                  double ex,double ey,double &inferred){
    double span=double(b[0])-a[0],days=fabs(span);
    if(days<1.2*fmax(a[1],b[1])||days>4920.||fabs(a[10]-b[10])>1.10)return -1.e19;
    if(na[2]==nb[2]||na[2]==nb[3]||na[3]==nb[2]||na[3]==nb[3])return -1.e19;
    if(fabs(a[12]-b[12])>4.2*sqrt(sq(a[13])+sq(b[13]))+
       .8*fmax(a[19],b[19])*days*fmax(a[12],b[12]))return -1.e19;
    double phot=sq(a[10]-b[10])/(sq(.26)+sq(.27*(a[11]-b[11])));
    if(phot>16.)return -1.e19;
    V east{ex,ey},north{-ey,ex},delta=sub(vec(b,2),vec(a,2));
    double g0=.5*(a[12]+b[12]),g=g0,h=0.,norm=1.+.25*dot(add(vec(a,2),vec(b,2)),add(vec(a,2),vec(b,2)))/sq(arc);
    // The long displacement also has to be attainable from the first
    // measured sky rate throughout its remaining distance uncertainty.
    V k=sub(vec(b,6),vec(a,6)),shift=mul(vec(a,14),span);
    V mismatch=sub(add(delta,mul(k,a[12])),shift);
    V axis=mul(sub(k,mul(vec(a,16),span)),a[13]);
    double phase=speed(a[12])*days;
    double noise=17.+(a[18]+.032)*days+.80*a[19]*days*(a[12]*arc+sqrt(dot(shift,shift)))+
                 .7*sqrt(dot(shift,shift))*phase*phase;
    double den=sq(noise)+dot(axis,axis),proj=dot(axis,mismatch);
    if(dot(mismatch,mismatch)-proj*proj/den>18.*sq(noise))return -1.e19;
    V p[4],D[4];
    for(int j=0;j<4;j++){
        const float *s=j<2?a:b;double offset=(j%2?.5:-.5)*s[1];
        p[j]=add(vec(s,2),mul(vec(s,4),offset));D[j]=add(vec(s,6),mul(vec(s,8),offset));
    }
    double dt=span+.5*(a[1]-b[1]);
    if(fabs(dt)<1.)return -1.e19;
    double t[4]={-.5*dt,-.5*dt+a[1],.5*dt,.5*dt+b[1]};
    double gw=.44*g0,hw=.48*speed(g0),chi=0.,det=0.,energy=0.;
    for(int step=0;step<2;step++){
        double n2=step?sq(speed(g))/(norm*sqrt(norm)):0.;
        double radial=step?h+.25*dot(add(vec(a,14),vec(b,14)),add(vec(a,2),vec(b,2)))/sq(arc):0.;
        double tau[4],ivf[4];
        for(int j=0;j<4;j++){
            double z=n2*t[j]*t[j],u=radial*t[j];
            ivf[j]=1./(1.-.5*z*(1.-u)+z*(z-15.*u*u)/24.);
            tau[j]=t[j]*(1.-z/6.+z*u/4.+z*(z-45.*u*u)/120.)*ivf[j];
        }
        double base=tau[2]-tau[0],inv=1./base;Normal fit;
        V dp=sub(p[2],p[0]),ds=sub(mul(D[2],ivf[2]),mul(D[0],ivf[0]));
        for(int j=1;j<=3;j+=2){
            double f=(tau[j]-tau[0])*inv,dtj=j==1?a[1]:b[1];
            V local=sub(p[j],p[0]);
            V G=sub(sub(mul(D[j],ivf[j]),mul(D[0],ivf[0])),mul(ds,f));
            V H=sub(mul(local,tau[j]),mul(dp,tau[2]*f));
            V R=add(sub(local,mul(dp,f)),mul(G,g0));
            double angular=speed(g)*days;
            double wt=1./(.055+sq(.14)*(sq(f)+sq(1.-f))+
                          sq((step?.012:.90)*speed(g)*arc*dtj*angular*angular));
            fit.measure(R.x,G.x*gw,H.x*hw,wt);fit.measure(R.y,G.y*gw,H.y*hw,wt);
        }
        V U=mul(dp,inv),K=mul(ds,inv),Z=mul(sub(mul(p[2],tau[2]),mul(p[0],tau[0])),inv);
        V vel=add(U,mul(K,g0));double vc=speed(g)*arc/sqrt(sqrt(norm));
        fit.measure(dot(vel,east)-vc*(1.+1.5*(g0-g)/g),
                    gw*(dot(K,east)-1.5*vc/g),hw*dot(Z,east),1./sq(.30*vc+.2/fabs(dt)));
        fit.measure(dot(vel,north),gw*dot(K,north),hw*dot(Z,north),1./sq(.39*vc+.2/fabs(dt)));
        det=fit.aa*fit.bb-fit.ab*fit.ab;if(!(det>0.))return -1.e19;
        double u=(fit.ab*fit.by-fit.bb*fit.ay)/det,v=(fit.ab*fit.ay-fit.aa*fit.by)/det;
        chi=fmax(0.,fit.cost+u*fit.ay+v*fit.by);
        if(chi>(step?28.:40.)||fabs(u)>3.3||fabs(v)>3.4)return -1.e19;
        g=g0+gw*u;h=hw*v;
        if(!(g>1./2500.&&g<1./17.))return -1.e19;
        V w=add(add(U,mul(K,g)),mul(Z,h));
        energy=(dot(w,w)+sq(h*arc))*sqrt(norm)/sq(speed(g)*arc);
        if(energy>(step?2.4:3.25))return -1.e19;
    }
    inferred=g;
    return -.5*chi-.40*phot-.24*log(det)+.47*(a[20]+b[20])+.26*log1p(days/60.)-.75*fmax(0.,energy-1.7);
}
}
#ifdef CHORD_GPU
extern "C" __global__ void chord_groups(const float *arc,const float *group,int *mask,int groups,int first,int total,double ex,double ey){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=total)return;
    mask[k]=chord_device::overlaps(arc+24*(first+k/groups),group+10*(k%groups),ex,ey);
}
extern "C" __global__ void chord_winners(const float *arc,const int *row,const int *order,const long long *tasks,
           const int *base,const int *size,int groups,int first,int count,double ex,double ey,int *winners,float *values){
    using namespace chord_device;
    int k=blockIdx.x;
    if(k>=count)return;
    int i=first+int(tasks[k]/groups),s=int(tasks[k]%groups),lane=threadIdx.x;
    int j=lane<size[s]?order[base[s]+lane]:-1;
    double g=0.,v=-1.e19;
    if(j>i)v=compare(arc+24*i,arc+24*j,row+4*i,row+4*j,ex,ey,g);
    __shared__ float score[128],range[128];__shared__ int which[128];
    score[lane]=float(v);range[lane]=float(g);which[lane]=j;__syncthreads();
    // Each half of a spatially coherent group preserves one continuation.
    for(int step=32;step>0;step/=2){
        if((lane%64)<step&&score[lane+step]>score[lane]){
            score[lane]=score[lane+step];range[lane]=range[lane+step];which[lane]=which[lane+step];
        }
        __syncthreads();
    }
    if(lane==0||lane==64){
        int at=2*k+lane/64;
        winners[2*at]=i;winners[2*at+1]=which[lane];values[2*at]=score[lane];values[2*at+1]=range[lane];
    }
}
#else
extern "C" void chord_scores(const float *arc,const int *row,const int *pairs,int n,double ex,double ey,double *out){
    for(int i=0;i<n;i++){
        int a=pairs[2*i],b=pairs[2*i+1];double g=0.;
        double score=chord_device::compare(arc+24*a,arc+24*b,row+4*a,row+4*b,ex,ey,g);
        out[2*i]=score;out[2*i+1]=g;
    }
}
#endif
