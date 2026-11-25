// A pair fixes the angular endpoints of a two-dimensional orbit family.
// Sample its image in each exposure and bring the measured peaks back to
// common coordinates in inverse range and line-of-sight velocity.
#ifdef __CUDACC__
#define FN __device__ __forceinline__
#else
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#define FN inline
using std::sqrt;
using std::fabs;
using std::exp;
using std::log;
using std::floor;
using std::fmax;
using std::fmin;
#endif

namespace mesh {
constexpr double arc=206264.80624709636,mu=.0002959122082855911;
constexpr int nstate=7, per_view=6;
struct V {
    double x,y;
    FN V operator+(V b)const{return {x+b.x,y+b.y};}
    FN V operator-(V b)const{return {x-b.x,y-b.y};}
    FN V operator*(double z)const{return {x*z,y*z};}
};
FN double dot(V a,V b){return a.x*b.x+a.y*b.y;}
FN double sq(double x){return x*x;}
FN double clip(double x,double a,double b){return fmax(a,fmin(x,b));}
struct Image {
    double t,ox,oy,oz,depth,lever;
    float xmin,xmax,ymin,ymax,bin_width;
    int first,columns,night;
};
struct Point {float x,y,mag,quality;int row;};
struct Endpoint {double x,y,m,sn;int exposure,row;};
struct Edge {Endpoint a,b,c;double gamma,merit;int component,pad;};
struct State {V pos,vel;double g,h,k,rad,v2;};
struct Family {
    State state[nstate];double ref,span,g0,sg,sh,cross,mag,color,anchor_noise,merit;
    int a,b,c,na,nb,nc;
};
struct Wave {V center,range,radial,curve,bend,mixed;double width;};
struct Vote {
    int row,image;
    float u,v,aa,ab,bb,score;
};
struct Seed {int rows[4];float gamma,score;int witness=-1;};
FN void lagrange(double t,double k,double radial,double v2,double &f,double &s){
    double z=k*t*t,u=radial*t,vt=v2*t*t;
    f=1.-.5*z*(1.-u)+z*(3.*vt-2.*z-15.*u*u)/24.;
    s=t*(1.-z/6.+z*u/4.+z*(9.*vt-8.*z-45.*u*u)/120.);
}
FN V project(const State &p,const Image &e,double t){
    double f,s;
    lagrange(t,p.k,p.rad,p.v2,f,s);
    return (p.pos*f+p.vel*s-V{e.ox,e.oy}*p.g)*(arc/(f+s*p.h-e.oz*p.g));
}
FN State through(const Edge &p,const Image &a,const Image &b,double g,double h){
    double t=.5*(b.t-a.t);
    V obs_a={a.ox,a.oy},obs_b={b.ox,b.oy};
    V qa={p.a.x,p.a.y},qb={p.b.x,p.b.y};
    State s{};s.g=g;s.h=h;s.k=mu*g*g*g;
    for(int k=0;k<3;k++){
        double f1,s1,f2,s2;
        lagrange(-t,s.k,s.rad,s.v2,f1,s1);
        lagrange(t,s.k,s.rad,s.v2,f2,s2);
        V first=qa*(f1+s1*h-g*a.oz)+obs_a*g;
        V last=qb*(f2+s2*h-g*b.oz)+obs_b*g;
        double d=f1*s2-f2*s1;
        s.pos=(first*s2-last*s1)*(1./d);
        s.vel=(last*f1-first*f2)*(1./d);
        double n=1.+dot(s.pos,s.pos);
        s.k=mu*g*g*g/(n*sqrt(n));s.rad=(dot(s.pos,s.vel)+h)/n;
        s.v2=(dot(s.vel,s.vel)+h*h)/n;
    }
    return s;
}
FN bool make_family(const Edge &p,const Image *vis,V east,Family &f){
    const Image &a=vis[p.a.exposure],&b=vis[p.b.exposure];
    double dt=b.t-a.t;
    if(fabs(dt)<.45 || a.night==b.night || !(p.gamma>0.))return false;
    double norm=sqrt(1.+sq(.5*(p.a.x+p.b.x))+sq(.5*(p.a.y+p.b.y)));
    double g=p.gamma*norm, start=g;
    V across={-east.y,east.x};
    double sig=.2*g;
    double noise=sqrt(.095*.095+sq(1.05/fmax(4.5,p.a.sn))+
                                  sq(1.05/fmax(4.5,p.b.sn)));
    // Condition the distance on the measured transverse velocity. The
    // radial velocity is relative to the heliocentric radius vector.
    for(int k=0;k<3;k++){
        State s=through(p,a,b,g,0.);
        double vc=sqrt(s.k)*norm;
        State t=through(p,a,b,g*1.001,0.);
        V v=s.vel*(1./vc),slope=(t.vel*(1./(vc*1.001500375))-v)*1000.;
        double meas=noise/(arc*vc*fabs(dt));
        double xx=sq(.145)+sq(meas),yy=sq(.35)+sq(meas);
        double u=dot(v,east)-1.,w=dot(v,across);
        double du=dot(slope,east),dw=dot(slope,across);
        double a0=du*du/xx+dw*dw/yy+1./sq(.27);
        sig=g/sqrt(a0);
        g*=1.+clip(-(du*u/xx+dw*w/yy+log(g/start)/sq(.27))/a0,-.12,.12);
    }
    if(g<1./13000.||g>1./15.){
        if(!p.component)return false;
        g=clip(start,1./12500.,1./16.);
    }
    auto s=through(p,a,b,g,0.);
    double h=-dot(s.pos,s.vel),vc=sqrt(s.k)*norm;
    f.sg=fmin(.31*g,fmax(.009*g,2.4*sig));f.sh=.46*vc;
    if(p.component&&p.c.row<0){
        // The two angular positions constrain the minimum heliocentric
        // speed at each trial distance. Integrate the allowed radial-speed
        // interval without an inclination or a circular-speed constraint.
        constexpr int nodes=41;
        double values[nodes],mass[nodes],sum=0.;
        for(int i=0;i<nodes;i++){
            double test=start*exp(.088*(i-nodes/2));values[i]=test;mass[i]=0.;
            if(test<=1./13000.||test>=1./15.)continue;
            State orbit=through(p,a,b,test,0.);
            double rad=-dot(orbit.pos,orbit.vel);
            orbit=through(p,a,b,test,rad);
            double scale=orbit.k*(1.+dot(orbit.pos,orbit.pos));
            double speed=(dot(orbit.vel,orbit.vel)+rad*rad)/scale;
            double err=noise/(arc*fabs(dt)*sqrt(scale));
            double excess=fmax(0.,sqrt(speed)-sqrt(2.));
            double support=sqrt(fmax(.012,2.-speed));
            mass[i]=support*exp(-.5*sq(excess/(.045+err))-.08*sq(log(test/start)));sum+=mass[i];
        }
        if(sum<.06)return false;
        double lo=g,mid=g,hi=g,acc=0.;
        for(int i=0;i<nodes;i++){
            double next=acc+mass[i];
            if(acc<=.10*sum&&next>.10*sum)lo=values[i];
            if(acc<=.50*sum&&next>.50*sum)mid=values[i];
            if(acc<=.90*sum&&next>.90*sum)hi=values[i];
            acc=next;
        }
        g=mid;s=through(p,a,b,g,0.);vc=sqrt(s.k)*sqrt(1.+dot(s.pos,s.pos));h=-dot(s.pos,s.vel);
        f.sg=fmin(.47*g,fmax(.055*g,.58*(hi-lo)));
        double kinetic=(dot(s.vel,s.vel)+h*h)/sq(vc);
        f.sh=vc*clip(.82*sqrt(fmax(.06,2.-kinetic)),.26,1.08);
    }
    f.g0=g;f.ref=.5*(a.t+b.t);f.span=fabs(dt);
    f.a=p.a.row;f.b=p.b.row;f.c=p.c.row;f.na=a.night;f.nb=b.night;f.nc=-1;
    f.mag=.5*(p.a.m+p.b.m);f.color=.5*(a.lever+b.lever);
    f.anchor_noise=noise;f.merit=p.merit;f.cross=0.;
    if(f.c>=0){
        const auto &c=vis[p.c.exposure];f.nc=c.night;
        if(f.nc==f.na||f.nc==f.nb)return false;
        double t=c.t-f.ref;
        double se=sqrt(sq(.11)+sq(1.1/fmax(4.5,p.c.sn))+sq(noise)*(.5+sq(t/dt)));
        double base=p.gamma*norm,width=fmax(f.sg,.052*g),rad=f.sh;
        g=base;
        V obs={p.c.x*arc,p.c.y*arc};
        for(int iter=0;iter<7;iter++){
            auto state=through(p,a,b,g,h);
            V pred=project(state,c,t),err=obs-pred;
            V A=(project(through(p,a,b,g+.0002*width,h),c,t)-pred)*(5000./se);
            V B=(project(through(p,a,b,g,h+.0002*rad),c,t)-pred)*(5000./se);
            double aa=dot(A,A)+1.,ab=dot(A,B),bb=dot(B,B)+1.;
            double det=sq(A.x*B.y-A.y*B.x)+aa+bb-1.;
            double ax=dot(A,err)/se-(g-base)/width;
            double bx=dot(B,err)/se-h/rad;
            if(iter==6){
                if(dot(err,err)>sq(5.*se)||fabs(h)>1.65*vc)return false;
                f.sg=fmin(.27*g,2.6*width/sqrt(aa));
                double sigv=sqrt(aa/det);
                f.sh=2.6*rad*sigv;
                f.cross=-2.6*width*sigv*ab/aa;
                break;
            }
            g+=clip(width*(bb*ax-ab*bx)/det,-.12*g,.12*g);
            h+=rad*clip((aa*bx-ab*ax)/det,-.50,.50);
            if(!(g>1./13000.&&g<1./13.))return false;
        }
        f.g0=g;f.mag=(p.a.m+p.b.m+p.c.m)/3.;f.color=(a.lever+b.lever+c.lever)/3.;
    }
    double excursion=f.sg+fabs(f.cross);
    if(excursion>.82*g){f.sg*=.82*g/excursion;f.cross*=.82*g/excursion;}
    f.state[0]=through(p,a,b,g,h);
    f.state[1]=through(p,a,b,g+f.sg,h);f.state[2]=through(p,a,b,g-f.sg,h);
    f.state[3]=through(p,a,b,g+f.cross,h+f.sh);f.state[4]=through(p,a,b,g-f.cross,h-f.sh);
    f.state[5]=through(p,a,b,g+f.cross+f.sg,h+f.sh);
    f.state[6]=through(p,a,b,g-f.cross-f.sg,h-f.sh);
    return true;
}
FN bool image_wave(const Family &f,const Image &e,Wave &w){
    if(!e.columns||e.night==f.na||e.night==f.nb||e.night==f.nc||e.depth+.90<f.mag)return false;
    double dt=e.t-f.ref;
    if(fabs(dt)*sqrt(f.state[0].k)>.63)return false;
    w.center=project(f.state[0],e,dt);
    // Most survey shutters are away from the path. This loose bound uses
    // the distance derivative of the velocity fixed by the endpoints.
    double between=arc*(fabs(dt)*(sqrt(dot(f.state[1].vel-f.state[2].vel,
                        f.state[1].vel-f.state[2].vel))+.40*f.sh)
                        +3.5*f.sg+.00035);
    if(w.center.x<e.xmin-between||w.center.x>e.xmax+between||
       w.center.y<e.ymin-between||w.center.y>e.ymax+between)return false;
    V p=project(f.state[1],e,dt),m=project(f.state[2],e,dt);
    w.range=(p-m)*.5;w.curve=(p+m)*.5-w.center;
    p=project(f.state[3],e,dt);m=project(f.state[4],e,dt);
    w.radial=(p-m)*.5;w.bend=(p+m)*.5-w.center;
    p=project(f.state[5],e,dt);m=project(f.state[6],e,dt);
    w.mixed=(p+m)*.5-w.center-w.curve-w.bend;
    w.width=sqrt(sq(.23)+sq(f.anchor_noise)*(.5+sq(dt/f.span)));
    return true;
}
FN V locus(const Wave &w,double u,double v){
    return w.center+w.range*u+w.curve*(u*u)+w.radial*v+w.bend*(v*v)+w.mixed*(u*v);
}
FN int lower_y(const Point *points,int lo,int hi,double y){
    while(lo<hi){int m=lo+(hi-lo)/2;if(points[m].y<y)lo=m+1;else hi=m;}
    return lo;
}
FN int measure_view(const Family &f,const Image &e,int ei,const int *cols,
                    const Point *points,Vote *out,int &examined){
    Wave w;if(!image_wave(f,e,w))return 0;
    // Bound the narrow coordinate at each sorted catalogue column.
    double xx=sq(w.range.x)+sq(w.radial.x)+sq(w.width);
    double xy=w.range.x*w.range.y+w.radial.x*w.radial.y;
    double yy=sq(w.range.y)+sq(w.radial.y)+sq(w.width);
    double var=sq(w.range.x*w.radial.y-w.range.y*w.radial.x)
              +sq(w.width)*(xx+yy-sq(w.width));
    double sy=sqrt(var/xx),slope=xy/xx;
    double cx=fabs(w.curve.x)+fabs(w.bend.x)+fabs(w.mixed.x);
    double cy=fabs(w.curve.y-slope*w.curve.x)+fabs(w.bend.y-slope*w.bend.x)+fabs(w.mixed.y-slope*w.mixed.x);
    double wx=1.37*sqrt(xx)+1.65*cx+2.*w.width;
    double wy=1.37*sqrt(yy)+1.65*(fabs(w.curve.y)+fabs(w.bend.y)+fabs(w.mixed.y))+2.*w.width;
    if(w.center.x+wx<e.xmin||w.center.x-wx>e.xmax||
       w.center.y+wy<e.ymin||w.center.y-wy>e.ymax)return 0;
    // Wide, nearly round regions have poor localization and thousands
    // of alternatives. More informative visits still vote for this pair.
    if(sqrt(var)>380000.||w.width>120.)return 0;
    int l=(int)fmax(0.,floor((w.center.x-wx-e.xmin)/e.bin_width));
    int r=(int)fmin(e.columns-1.,floor((w.center.x+wx-e.xmin)/e.bin_width));
    double photvar=.045+sq(.29*(f.color-e.lever))+.033;
    int count=0;
    for(int col=l;col<=r;col++){
        double left=fmax(w.center.x-wx,e.xmin+e.bin_width*col);
        double right=fmin(w.center.x+wx,e.xmin+e.bin_width*(col+1));
        double y=w.center.y+slope*(.5*(left+right)-w.center.x);
        double height=1.4*sy+1.7*cy+.5*fabs(slope)*(right-left)+2.*w.width;
        double low=fmax(w.center.y-wy,y-height),high=fmin(w.center.y+wy,y+height);
        if(high<e.ymin||low>e.ymax)continue;
        int end=cols[e.first+col+1],i=lower_y(points,cols[e.first+col],end,low);
        for(;i<end&&points[i].y<=high;i++){
            const auto &p=points[i];++examined;
            if(sq(p.mag-f.mag)>9.*photvar)continue;
            V target={p.x,p.y};double u=0.,v=0.,aa=1.,ab=0.,bb=1.;
            for(int k=0;k<3;k++){
                V d=target-locus(w,u,v);
                V A=w.range+w.curve*(2.*u)+w.mixed*v,B=w.radial+w.bend*(2.*v)+w.mixed*u;
                double regular=sq(w.width);
                aa=dot(A,A)+regular;ab=dot(A,B);bb=dot(B,B)+regular;
                double determinant=sq(A.x*B.y-A.y*B.x)+regular*(aa+bb-regular);
                double a=dot(A,d)-regular*u,b=dot(B,d)-regular*v;
                u+=clip((bb*a-ab*b)/determinant,-1.25,1.25);
                v+=clip((aa*b-ab*a)/determinant,-1.25,1.25);
            }
            V err=target-locus(w,u,v);
            double residual=dot(err,err)/sq(w.width);
            if(fabs(u)>1.26||fabs(v)>1.30||residual>10.)continue;
            // Pack the unregularized local likelihood at the inferred
            // position. The weak prior is applied only once in the join.
            V A=w.range+w.curve*(2.*u)+w.mixed*v,B=w.radial+w.bend*(2.*v)+w.mixed*u;
            double length=sqrt(dot(A,A)+1.e-18);
            aa=length/w.width;ab=dot(A,B)/(length*length);
            bb=fabs(A.x*B.y-A.y*B.x)/(length*w.width);
            double sc=.52*p.quality-.5*sq(p.mag-f.mag)/photvar-.53*residual
                     -.50*(sq(u)+sq(v))-.34*log(1.+sqrt(var)/80.);
            Vote vote={p.row,ei,(float)u,(float)v,(float)aa,(float)ab,(float)bb,(float)sc};
            if(count<per_view){out[count++]=vote;continue;}
            // Keep returns from different regions of the orbital family.
            int worst=0;double low_score=1.e30;
            for(int j=0;j<count;j++){
                double overlap=exp(-7.*(sq(u-out[j].u)+.55*sq(v-out[j].v)));
                double score=out[j].score-.85*overlap;
                if(score<low_score){worst=j;low_score=score;}
            }
            if(sc>out[worst].score)out[worst]=vote;
        }
    }
    return count;
}
} // namespace mesh

namespace mesh {
struct Hit {Vote vote;int pair;};
}
#ifdef __CUDACC__
extern "C" __global__ void orbital_returns(const mesh::Family *family,const mesh::Image *vis,
        const int *cols,const mesh::Point *points,int ne,int count,mesh::Hit *hits,int *total){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=ne*count)return;
    int i=k/ne,e=k%ne;
    if(family[i].span<=0.)return;
    mesh::Vote votes[mesh::per_view];int examined=0;
    int n=mesh::measure_view(family[i],vis[e],e,cols,points,votes,examined);
    if(n){
        int at=atomicAdd(total,n);
        for(int j=0;j<n;j++)hits[at+j]={votes[j],i};
    }
}
#else
#include <atomic>
#include <thread>
namespace mesh {
struct Join {int a,b;double u,v,merit;};
bool agreement(const Vote &a,const Vote &b,Join &out){
    // Eliminate inverse range from two strip measurements. Use conditional
    // slopes instead of subtracting almost equal normal-matrix determinants.
    double ia=1./(sq(a.aa)+.001),ib=1./(sq(b.aa)+.001);
    double sigma=ia+ib+.00000004;
    double d=a.u-b.u+a.ab*a.v-b.ab*b.v,s=a.ab-b.ab;
    double inv=1.+sq(a.bb)+sq(b.bb)+s*s/sigma;
    double v=(sq(a.bb)*a.v+sq(b.bb)*b.v+s*d/sigma)/inv;
    double chi=sq(a.bb*(v-a.v))+sq(b.bb*(v-b.v))+sq(d-s*v)/sigma+sq(v);
    if(chi>19.||fabs(v)>1.45)return false;
    double u=((a.u+a.ab*(a.v-v))*ib+(b.u+b.ab*(b.v-v))*ia)/(ia+ib);
    if(fabs(u)>1.37)return false;
    out.u=u;out.v=v;
    out.merit=a.score+b.score-.44*chi-.5*sq(u)+
         .33*log(1.+fmin(1.e8,(1./ia+1./ib)*inv));
    return true;
}
double discrepancy(const Vote &a,double u,double v){
    double x=a.aa*(u-a.u+a.ab*(v-a.v)),y=a.bb*(v-a.v);
    return x*x+y*y;
}
void push(Seed *out,int &count,int cap,Seed value){
    if(count<cap){out[count++]=value;return;}
    int worst=0;
    for(int i=1;i<count;i++)if(out[i].score<out[worst].score)worst=i;
    if(value.score>out[worst].score)out[worst]=value;
}
int match(const Family &f,const Image *vis,std::vector<Vote> &all,Seed *out,int cap){
    if(all.empty())return 0;
    // Device blocks finish in arbitrary order. Canonicalize the ordered
    // night pairs before reducing the independent evidence.
    auto order=[](const Vote &a,const Vote &b){
        return a.image!=b.image?a.image<b.image:a.row<b.row;
    };
    if(!std::is_sorted(all.begin(),all.end(),order))std::sort(all.begin(),all.end(),order);
    // Retain a spread in observing epoch when a very crowded family
    // produces many votes. The return catalogue is already magnitude-limited.
    if(all.size()>200){
        std::partial_sort(all.begin(),all.begin()+200,all.end(),
                          [](auto &a,auto &b){return a.score>b.score;});
        all.resize(200);
    }
    if(f.c>=0){
        int count=0;
        for(const auto &a:all){
            double bonus=0.,u=a.u,v=a.v;int witness=-1;
            for(const auto &b:all){
                if(vis[a.image].night==vis[b.image].night)continue;
                Join j;
                if(!agreement(a,b,j))continue;
                double support=3.8+.43*b.score-.25*(discrepancy(a,j.u,j.v)+discrepancy(b,j.u,j.v));
                if(support>bonus){bonus=support;witness=b.row;u=j.u;v=j.v;}
            }
            double merit=.58*f.merit+a.score+bonus;
            push(out,count,cap,{{f.a,f.b,f.c,a.row},float(f.g0+f.sg*u+f.cross*v),float(merit),witness});
        }
        return count;
    }
    std::vector<Join> pool;
    for(int i=0;i<int(all.size());i++)for(int j=0;j<i;j++){
        const auto &a=all[i],&b=all[j];
        if(vis[a.image].night==vis[b.image].night)continue;
        Join q{i,j,0.,0.,0.};
        if(!agreement(a,b,q))continue;
        pool.push_back(q);
    }
    int count=0;
    if(pool.size()>32){
        std::partial_sort(pool.begin(),pool.begin()+32,pool.end(),
                          [](auto &a,auto &b){return a.merit>b.merit;});
        pool.resize(32);
    }
    for(auto &j:pool){
        auto &a=all[j.a],&b=all[j.b];
        double spare=0.;int witness=-1;
        for(auto &v:all){
            if(vis[v.image].night==vis[a.image].night||vis[v.image].night==vis[b.image].night)continue;
            double chi=discrepancy(v,j.u,j.v);
            double support=4.7+.42*v.score-.33*chi;
            if(chi<15.&&support>spare){spare=support;witness=v.row;}
        }
        double q=j.merit+spare+.46*clip(f.merit,-5.,12.);
        Seed seed{{f.a,f.b,a.row,b.row},float(f.g0+f.sg*j.u),float(q),witness};
        push(out,count,cap,seed);
    }
    return count;
}
}
extern "C" int mesh_family_bytes(){return sizeof(mesh::Family);}
extern "C" void mesh_families(const mesh::Edge *edge,int n,const mesh::Image *vis,double east,
                              double north,mesh::Family *out){
    for(int i=0;i<n;i++){
        out[i]=mesh::Family{};
        if(!make_family(edge[i],vis,{east,north},out[i]))out[i].span=0.;
    }
}
extern "C" int mesh_host(const mesh::Family *family,int count,const mesh::Image *vis,int ne,
                         const int *cols,const mesh::Point *points,mesh::Seed *out,int slots,
                         int *sizes,int threads){
    try{
        std::atomic<int> next{0};std::vector<std::thread> team;
        auto run=[&]{
            std::vector<mesh::Vote> all;
            for(int i=next++;i<count;i=next++){
                all.clear();
                if(family[i].span>0.){
                    for(int e=0;e<ne;e++){
                        mesh::Vote tmp[mesh::per_view];int seen=0;
                        int n=measure_view(family[i],vis[e],e,cols,points,tmp,seen);
                        all.insert(all.end(),tmp,tmp+n);
                    }
                }
                sizes[i]=mesh::match(family[i],vis,all,out+i*slots,slots);
            }
        };
        for(int k=1;k<threads;k++)team.emplace_back(run);
        run();for(auto &t:team)t.join();
        return 0;
    }catch(...){return -1;}
}
extern "C" int mesh_matches(const mesh::Family *family,int count,const mesh::Image *vis,
        const mesh::Hit *hits,int n,mesh::Seed *out,int slots,int *sizes){
    try{
        std::vector<std::vector<mesh::Vote>> all(count);
        for(int i=0;i<n;i++)all.at(hits[i].pair).push_back(hits[i].vote);
        for(int i=0;i<count;i++)sizes[i]=mesh::match(family[i],vis,all[i],out+i*slots,slots);
        return 0;
    }catch(...){return -1;}
}
#endif
