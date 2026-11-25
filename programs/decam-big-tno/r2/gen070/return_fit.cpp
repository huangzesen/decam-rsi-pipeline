// Condition the short displacements at either end on a measured long chord.
// The same calculation is used by the CPU and the CUDA thread for one chord.
#ifdef __CUDACC__
#define RF __device__ inline
#else
#include <algorithm>
#include <cmath>
#define RF inline
#endif

namespace return_fit {
constexpr double arc=206264.80624709636, gaussian=.01720209895;
constexpr int adjacent=8;
struct Point {
    // t is in days. Directions and projected observer coordinates share
    // the arcsecond scale; inverse AU converts the latter to parallax.
    double t;
    float x,y,px,py,mag,error,lever,quality;
    int night,row;
};
struct Node {int center,edge[adjacent];float score[adjacent];};
struct Fit {int row[4];float g,h,merit;int witness;};
struct Block {
    float t,dt,x,y,wx,wy,px,py,dpx,dpy,mag,dmag;
    int first,count;
};
struct V {
    double x,y;
    RF V operator+(V b)const{return {x+b.x,y+b.y};}
    RF V operator-(V b)const{return {x-b.x,y-b.y};}
    RF V operator*(double k)const{return {x*k,y*k};}
    RF double dot(V b)const{return x*b.x+y*b.y;}
};
RF double sq(double x){return x*x;}
RF double bound(double x,double lo,double hi){return fmax(lo,fmin(hi,x));}
RF V xy(const Point &p){return {p.x,p.y};}
RF V par(const Point &p){return {p.px,p.py};}

// f and s are the Lagrange coefficients relative to the epoch halfway
// between the two distant measurements. Times are in days.
struct Model {
    double g,h,k,rad,v2,mid,ta,tb,ia,ib;
    RF void epoch(double t,double &tau,double &invf)const{
        double t2=t*t,z=k*t2,u=t*rad;
        double f=1.-.5*z*(1.-u)+z*((3.*v2-2.*k)*t2-15.*u*u)/24.;
        double s=t*(1.-z/6.+z*u/4.+z*((9.*v2-8.*k)*t2-45.*u*u)/120.);
        invf=1./f;tau=s*invf;
    }
    RF Model(const Point &a,const Point &b,double gamma,double hdot):
            g(gamma),h(hdot),mid(.5*(a.t+b.t)){
        V p=(xy(a)+xy(b)+(par(a)+par(b))*g)*(.5/arc);
        V v=(xy(b)-xy(a)+(par(b)-par(a))*g)*(1./((b.t-a.t)*arc))+p*h;
        double n=1.+p.dot(p);
        k=sq(gaussian)*g*g*g/(n*sqrt(n));rad=(p.dot(v)+h)/n;
        v2=(v.dot(v)+h*h)/n;
        epoch(a.t-mid,ta,ia);epoch(b.t-mid,tb,ib);
    }
    RF V residual(const Point &a,const Point &b,const Point &p)const{
        double t,inv;epoch(p.t-mid,t,inv);
        double w=(t-ta)/(tb-ta);
        V da=xy(p)-xy(a),db=xy(p)-xy(b);
        return da*((1.-w)*(1.+h*ta))+db*(w*(1.+h*tb))+
               (par(p)*inv-par(a)*((1.-w)*ia)-par(b)*(w*ib))*g;
    }
};

struct Cut {
    int id;
    V r,u,v;
    double f,var,merit,low,high;
};
struct Normal {
    double aa=1.,ab=0.,bb=1.,ar=0.,br=0.,rr=0.;
    RF void observe(V r,V a,V b,double wt){
        aa+=wt*a.dot(a);ab+=wt*a.dot(b);bb+=wt*b.dot(b);
        ar+=wt*a.dot(r);br+=wt*b.dot(r);rr+=wt*r.dot(r);
    }
    RF double solve(double &u,double &v,double &chi)const{
        double det=aa*bb-ab*ab;
        u=(ab*br-bb*ar)/det;v=(ab*ar-aa*br)/det;
        chi=fmax(0.,rr+u*ar+v*br);return det;
    }
};
RF bool physical(const Point &a,const Point &b,double g,double h,double &energy){
    if(!(g>1./4000.&&g<1./15.))return false;
    Model m(a,b,g,h);
    V vel=((xy(b)*(1.+m.tb*h)+par(b)*(g*m.ib))-
           (xy(a)*(1.+m.ta*h)+par(a)*(g*m.ia)))*(1./(arc*(m.tb-m.ta)));
    V pos=(xy(a)*(1.+h*m.ta)+par(a)*(g*m.ia))*(1./arc)-vel*m.ta;
    energy=(vel.dot(vel)+h*h)*sqrt(1.+pos.dot(pos))/(sq(gaussian)*g*g*g);
    return energy<2.19;
}
RF Fit empty(){
    Fit v={{-1,-1,-1,-1},0,0,-1.e20f,-1};
    return v;
}
RF bool nearby_block(const Point &a,const Block &b,double g,double vc,double scatter,bool wide){
    double t=b.t-a.t,dt=fabs(t)+b.dt;
    if(dt<27.||fabs(t)-b.dt>5050.||fabs(a.mag-b.mag)>b.dmag+1.10)return false;
    double drift=wide?.25*vc:vc;
    double px=(a.px-b.px)*g+1.5*drift*t,py=(a.py-b.py)*g;
    double wx=3.2*(scatter*(fabs(px)+g*b.dpx+1.5*drift*b.dt)+(wide?.78:.28)*vc*dt)+55.;
    double wy=3.2*(scatter*(fabs(py)+g*b.dpy)+(wide?.79:.40)*vc*dt)+55.;
    return fabs(a.x+g*a.px-drift*a.t-b.x)<=b.wx+wx && fabs(a.y+g*a.py-b.y)<=b.wy+wy;
}
RF void search(const Point *points,const Node &la,const Node &lb,double radius,Fit &best){
    const Point &a=points[la.center],&b=points[lb.center];
    double dt=b.t-a.t;
    if(a.row==b.row||a.night==b.night||fabs(dt)<27.||fabs(dt)>5050.)return;
    double vm=.030+sq(.28*(a.lever-b.lever))+sq(.72*a.error)+sq(.72*b.error);
    double dm=a.mag-b.mag,phot=dm*dm/vm;
    if(phot>13.)return;
    bool wide=radius<0.;radius=fabs(radius);
    V pos=(xy(a)+xy(b))*(.5/arc);
    double norm=sqrt(1.+pos.dot(pos)),g0=norm/radius,vc=gaussian*pow(radius,-1.5)*norm*arc;
    V measured=(xy(b)-xy(a))*(1./dt),reflex=(par(b)-par(a))*(1./dt);
    V v0=measured+reflex*g0;
    double drift=wide?.25*vc:vc;
    V derivative=reflex*g0-V{1.5*drift,0.};
    double sigma_g=(wide||radius>75.)?.39:.23;
    double wx=1./sq((wide?.74:(radius>75.?.55:.25))*vc),wy=1./sq((wide?.76:(radius>75.?.65:.38))*vc);
    double prec=1./sq(sigma_g)+sq(derivative.x)*wx+sq(derivative.y)*wy;
    double shift=-((v0.x-drift)*derivative.x*wx+v0.y*derivative.y*wy)/prec;
    double kin=sq(shift/sigma_g)+sq(v0.x-drift+derivative.x*shift)*wx+sq(v0.y+derivative.y*shift)*wy;
    if(kin>17.||shift<-.75||shift>.83)return;
    g0*=1.+shift;
    double gs=fmax(g0*.022,(norm/radius)/sqrt(prec));
    double circular=gaussian*pow(g0/norm,1.5);
    double hs=circular*((wide||radius>75.)?.60:.38),h0=-pos.dot(measured+reflex*g0)/arc;
    Model base(a,b,g0,h0),distance(a,b,g0+.02*gs,h0),radial(a,b,g0,h0+.02*hs);
    Cut cuts[2*adjacent];int n=0;
    Fit three=empty();
    for(int k=0;k<2*adjacent;k++){
        const Node &node=k<adjacent?la:lb;
        int id=node.edge[k%adjacent];
        if(id<0)continue;
        const Point &p=points[id];
        double local=p.t-(k<adjacent?a.t:b.t);
        if(fabs(local)>.52*fabs(dt)||p.row==a.row||p.row==b.row)continue;
        double f=(p.t-a.t)/dt;
        V r=base.residual(a,b,p);
        V u=(distance.residual(a,b,p)-r)*50.,v=(radial.residual(a,b,p)-r)*50.;
        double variance=sq(p.error)+sq(a.error*(1.-f))+sq(b.error*f)+sq(.055);
        // The Taylor expansion here is used only to find measurements for
        // an exact Kepler fit. Widen the triplet gate on long extrapolations.
        variance+=sq(.06+.00006*fabs(local)*sq(circular*dt)*arc*.01);
        Normal fit;fit.observe(r,u,v,1./variance);
        double z,w,chi;double det=fit.solve(z,w,chi);
        if(chi>20.||fabs(z)>3.8||fabs(w)>3.8)continue;
        double half=3.8*sqrt(fit.bb/det)+.07;
        cuts[n++]={id,r,u,v,f,variance, node.score[k%adjacent],z-half,z+half};
        if(p.night!=a.night&&p.night!=b.night){
            double g=g0+z*gs,h=h0+w*hs,energy;
            double merit=.63*(a.quality+b.quality)+.83*node.score[k%adjacent]
                    -.48*(phot+chi)-.44*kin-.32*log(det)+.28*log1p(fabs(local));
            if(merit>three.merit && physical(a,b,g,h,energy))
                three={{a.row,b.row,p.row,-1},float(g),float(h),float(merit),-1};
        }
    }
    for(int i=0;i<n;i++)for(int j=0;j<i;j++){
        const Cut &c=cuts[i],&d=cuts[j];if(c.low>d.high||d.low>c.high||c.id==d.id)continue;
        const Point &p=points[c.id],&q=points[d.id];
        int nights=2+(p.night!=a.night&&p.night!=b.night)+
                   (q.night!=a.night&&q.night!=b.night&&p.night!=q.night);
        if(nights<3||fabs(p.t-q.t)<.00015)continue;
        double shared=(1.-c.f)*(1.-d.f)*sq(a.error)+c.f*d.f*sq(b.error);
        double beta=shared/c.var,variance=fmax(sq(.07),d.var-beta*shared);
        Normal fit;
        fit.observe(c.r,c.u,c.v,1./c.var);
        fit.observe(d.r-c.r*beta,d.u-c.u*beta,d.v-c.v*beta,1./variance);
        double u,v,chi;double det=fit.solve(u,v,chi);
        if(chi>32.||fabs(u)>3.7||fabs(v)>3.6)continue;
        double g=g0+u*gs,h=h0+v*hs,energy;
        if(!physical(a,b,g,h,energy))continue;
        // Curvature of the admissible family can exceed the centroid noise.
        // Take a local Newton step at the inferred range and radial speed.
        Model fresh(a,b,g,h);
        V r=fresh.residual(a,b,p),s=fresh.residual(a,b,q);
        Normal step;step.ar=u;step.br=v;step.rr=sq(u)+sq(v);
        step.observe(r,c.u,c.v,1./c.var);
        step.observe(s-r*beta,d.u-c.u*beta,d.v-c.v*beta,1./variance);
        double du,dv;step.solve(du,dv,chi);
        g+=du*gs;h+=dv*hs;
        if(chi>20.||!physical(a,b,g,h,energy))continue;
        double merit=.37*(a.quality+b.quality)+.65*(c.merit+d.merit)-.45*phot
                    -.37*kin-.48*chi-.36*log(det)+6.8*(nights-3);
        Model solution(a,b,g,h);
        int witness=-1;double extra=0.;
        for(int k=0;k<n;k++){
            if(k==i||k==j)continue;
            const Cut &z=cuts[k];const Point &hit=points[z.id];
            if(hit.night==a.night||hit.night==b.night||hit.night==p.night||hit.night==q.night)continue;
            V err=solution.residual(a,b,hit);
            double xx=z.var+(fit.bb*sq(z.u.x)-2.*fit.ab*z.u.x*z.v.x+fit.aa*sq(z.v.x))/det;
            double yy=z.var+(fit.bb*sq(z.u.y)-2.*fit.ab*z.u.y*z.v.y+fit.aa*sq(z.v.y))/det;
            double xy=(fit.bb*z.u.x*z.u.y-fit.ab*(z.u.x*z.v.y+z.v.x*z.u.y)+fit.aa*z.v.x*z.v.y)/det;
            double D=xx*yy-xy*xy;if(!(D>0.))continue;
            double dev=(yy*sq(err.x)-2.*xy*err.x*err.y+xx*sq(err.y))/D;
            double odds=7.5+.35*z.merit-.5*dev-.5*log(fmax(.01,D));
            if(odds>extra){extra=fmin(13.,odds);witness=hit.row;}
        }
        merit+=extra;
        if(merit>best.merit)best={{a.row,b.row,p.row,q.row},float(g),float(h),float(merit),witness};
    }
    if(best.row[0]<0)best=three;
}
}

#ifdef __CUDACC__
extern "C" __global__ void return_blocks(const return_fit::Point *p,const return_fit::Node *nodes,
            const return_fit::Block *blocks,int nb,int first,int n,double radius,int *accept){
    using namespace return_fit;
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=n)return;
    const auto &a=p[nodes[first+k/nb].center];
    bool wide=radius<0.;radius=fabs(radius);
    accept[k]=nearby_block(a,blocks[k%nb],1./radius,gaussian*pow(radius,-1.5)*arc,(wide||radius>75.)?.4:.23,wide);
}
extern "C" __global__ void return_tiles(const return_fit::Point *p,const return_fit::Node *nodes,const int *order,
            const return_fit::Block *blocks,const long long *work,int nb,int first,int n,double radius,
            return_fit::Fit *out){
    using namespace return_fit;
    int tile=blockIdx.x,i=threadIdx.x;
    if(tile>=n)return;
    int k=int(work[tile]),a=first+k/nb;
    const auto &b=blocks[k%nb];
    auto best=empty();
    if(i<b.count){
        int j=order[b.first+i];if(j>a)search(p,nodes[a],nodes[j],radius,best);
    }
    __shared__ float merit[64];
    merit[i]=best.merit;
    __syncthreads();
    int lo=i&(~7),winner=lo;
    for(int u=lo+1;u<lo+8;u++)if(merit[u]>merit[winner])winner=u;
    if(i==winner)out[tile*8+(i/8)]=best;
}
#else
#include <thread>
#include <vector>
#include <atomic>
namespace return_fit {
struct Partner {int id;float merit,prior,g,sg;};
double local_support(const Point *points,const Node &node,V rate,V reflex,double g,double sg,double days,double vc){
    const auto &center=points[node.center];double best=-4.;
    for(int k=0;k<adjacent&&k<4&&node.edge[k]>=0;k++){
        const auto &p=points[node.edge[k]];
        double dt=p.t-center.t;
        if(fabs(dt)>.52*fabs(days)||fabs(dt)<.0001)continue;
        V local=(xy(p)-xy(center))*(1./dt);
        V slope=(par(p)-par(center))*(1./dt)-reflex;
        V offset=local-rate+slope*g;
        // Perspective changes a measured angular rate between apparitions.
        // The broad term is the unknown radial speed times the elapsed time.
        double phase=fabs(days)*vc/arc;
        double variance=(sq(p.error)+sq(center.error)+.006)/sq(dt)+
                        sq(.35+.64*phase*(vc+.7*sqrt(local.dot(local))));
        V u=slope*sg;
        double ud=u.dot(u),ur=u.dot(offset);
        double chi=(offset.dot(offset)-sq(ur)/(variance+ud))/variance;
        double info=.45*log1p(sq(.4*vc)/(variance+.04));
        best=fmax(best,.48*bound(node.score[k],-4.,12.)+info-.44*fmin(24.,chi));
    }
    return best;
}
}
extern "C" int return_partners(const return_fit::Point *points,const return_fit::Node *nodes,const int *order,
        const return_fit::Block *blocks,int nb,int first,int count,double radius,int cap,int threads,
        int *out,int *counts) {
    using namespace return_fit;
    std::vector<std::thread> pool;
    bool wide=radius<0.;radius=fabs(radius);
    double g=1./radius,vc=gaussian*pow(radius,-1.5)*arc;
    double scatter=(wide||radius>75.)?.4:.23,drift=wide?.25:1.;
    double wx=wide?.60:.085,wy=wide?.64:.165;
    std::atomic<int> next{0};
    for(int thread=0;thread<threads;thread++)pool.emplace_back([&]{
        std::vector<Partner> partners;
        for(;;){
            int i=next.fetch_add(1);if(i>=count)break;
            int root=i+first;const auto &na=nodes[root];const auto &a=points[na.center];
            partners.clear();
            for(int bi=0;bi<nb;bi++){
                const Block &z=blocks[bi];
                if(!nearby_block(a,z,g,vc,scatter,wide))continue;
                for(int k=z.first;k<z.first+z.count;k++){
                    int j=order[k];if(j<=root)continue;
                    const auto &nn=nodes[j];const auto &b=points[nn.center];
                    if(na.edge[0]<0&&nn.edge[0]<0)continue;
                    double days=b.t-a.t;if(fabs(days)<27.||fabs(days)>5050.)continue;
                    double vm=.030+sq(.28*(a.lever-b.lever))+sq(.72*a.error)+sq(.72*b.error);
                    double phot=sq(a.mag-b.mag)/vm;if(phot>13.)continue;
                    double rx=(b.x-a.x+(b.px-a.px)*g)/(days*vc)-drift;
                    double ry=(b.y-a.y+(b.py-a.py)*g)/(days*vc);
                    double sx=((b.px-a.px)*g/(days*vc)-1.5*drift)*scatter;
                    double sy=((b.py-a.py)*g/(days*vc))*scatter;
                    double xx=sq(sx)+wx,yy=sq(sy)+wy,cross=sx*sy;
                    double kinematic=(rx*rx*yy+ry*ry*xx-2.*rx*ry*cross)/(xx*yy-cross*cross);
                    if(kinematic>15.)continue;
                    double prec=1.+sx*sx/wx+sy*sy/wy;
                    double u=-(rx*sx/wx+ry*sy/wy)/prec;
                    float prior=.45*b.quality-.55*phot-.43*kinematic;
                    float first=prior+(nn.edge[0]>=0?.65*bound(nn.score[0],-4.,12.):-1.65);
                    partners.push_back({j,first,prior,float(g*(1.+scatter*u)),float(g*scatter/sqrt(prec))});
                }
            }
            auto better=[](const Partner &x,const Partner &y){return x.merit>y.merit;};
            int preselect=cap*3;
            if(int(partners.size())>preselect){
                std::nth_element(partners.begin(),partners.begin()+preselect,partners.end(),better);
                partners.resize(preselect);
            }
            for(auto &pick:partners){
                const auto &nn=nodes[pick.id];const auto &b=points[nn.center];double days=b.t-a.t;
                V v=(xy(b)-xy(a))*(1./days),s=(par(b)-par(a))*(1./days);
                double left=local_support(points,na,v,s,pick.g,pick.sg,days,vc);
                double right=local_support(points,nn,v,s,pick.g,pick.sg,days,vc);
                pick.merit=pick.prior+.90*fmax(left,right)+.66*fmin(left,right);
            }
            int nkeep=std::min(cap,int(partners.size()));
            if(int(partners.size())>cap)
                std::nth_element(partners.begin(),partners.begin()+cap,partners.end(),better);
            counts[i]=nkeep;
            for(int k=0;k<nkeep;k++){
                out[(i*cap+k)*2]=root;out[(i*cap+k)*2+1]=partners[k].id;
            }
        }
    });
    for(auto &w:pool)w.join();
    return 0;
}
extern "C" void return_chords_cpu(const return_fit::Point *p,const return_fit::Node *nodes,
            const int *pairs,int n,double radius,int threads,return_fit::Fit *out){
    std::vector<std::thread> pool;
    for(int j=0;j<threads;j++)pool.emplace_back([&,j]{
        for(int i=j;i<n;i+=threads){
            auto best=return_fit::empty();
            return_fit::search(p,nodes[pairs[2*i]],nodes[pairs[2*i+1]],radius,best);out[i]=best;
        }
    });
    for(auto &w:pool)w.join();
}
#endif
