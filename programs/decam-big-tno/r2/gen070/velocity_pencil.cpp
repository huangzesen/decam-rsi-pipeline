// Transverse velocities inferred from a common angular position and many
// independent single detections. For fixed inverse depth and radial rate
// each observation supplies two linear constraints on the transverse rate.
#ifndef __CUDACC_RTC__
#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <vector>
#define VP inline
#else
#define VP __device__ inline
#endif

namespace pencil {
constexpr double arc=206264.80624709636, gaussian=.01720209895;
struct Point { float x,y,mag,var,q; int row,ex; };
struct Visit { double t,ox,oy,oz,lever,depth; int night,pad; };
struct Edge {
    float x,y,gx,gy,hx,hy,xx,yy,ghx,ghy;
    float dt,noise,logrho,mark,mag,lever,variance;
    int row,night;
};
struct Origin {float g,vc,h,noise,mark,inorm; int row;};
struct Fit { int row[4]; float g,h,merit; int witness; };
struct V {double x,y;};
struct Transfer {
    double factor;
    V mean;
    VP V velocity(const Point &p)const{return {(p.x-mean.x)*factor,(p.y-mean.y)*factor};}
};
struct Track {
    const Point &p;const Visit &start;
    double ux,uy,hz,nominal,vc,norm;
    VP Track(const Point &a,const Visit &s,double radius,double pole_y,double pole_z,bool isotropic=false):p(a),start(s){
        norm=sqrt(1.+p.x*p.x+p.y*p.y);
        nominal=norm/radius;vc=gaussian*pow(nominal,1.5);
        double x=p.x+nominal*(s.ox-p.x*s.oz),y=p.y+nominal*(s.oy-p.y*s.oz);
        double ex=pole_y-y*pole_z,ey=pole_z*x,ez=-pole_y*x;
        double inv=1./sqrt((ex*ex+ey*ey+ez*ez)*sqrt(1.+x*x+y*y));
        hz=ez*inv;ux=(ex-p.x*ez)*inv;uy=(ey-p.y*ez)*inv;
        if(isotropic)hz=ux=uy=0.;
    }
    VP Transfer equation(const Visit &e,double d,double dh)const{
        double g=nominal*(1.+d),speed=vc*(1.+d)*sqrt(1.+d);
        double hx=p.x+g*(start.ox-p.x*start.oz),hy=p.y+g*(start.oy-p.y*start.oz);
        double n2=1.+hx*hx+hy*hy;
        double k=gaussian*gaussian*g*g*g/(n2*sqrt(n2));
        double h=hz*speed+dh*vc,vx=ux*speed+p.x*h,vy=uy*speed+p.y*h;
        double radial=(h+hx*vx+hy*vy)/n2,v2=(vx*vx+vy*vy+h*h)/n2;
        double dt=e.t-start.t;
        double z=k*dt*dt,rd=radial*dt;
        double f=1.-.5*z*(1.-rd)+z*((3.*v2-2.*k)*dt*dt-15.*rd*rd)/24.;
        double t=dt*(1.-z/6.+z*rd/4.+z*((9.*v2-8.*k)*dt*dt-45.*rd*rd)/120.);
        double den=f+t*h-g*e.oz;
        return {den/(t*vc),{(f*hx+t*vx-g*e.ox)/den,(f*hy+t*vy-g*e.oy)/den}};
    }
};
VP double square(double x){return x*x;}
VP double product(V a,V b){return a.x*b.x+a.y*b.y;}
VP V at(const Edge &b,double g,double h){
    return {b.x+g*(b.gx+g*b.xx)+h*(b.hx+g*b.ghx),
            b.y+g*(b.gy+g*b.yy)+h*(b.hy+g*b.ghy)};
}
struct Posterior {
    double a,b,c,det,g,h,chi;
};
// Posterior in fractional inverse depth and radial rate. Starting with
// the second direction conditions the velocity prior before testing a third.
VP Posterior solve(double a,double b,double c,double x,double y,double rr){
    double det=fmax(1.e-15,a*c-b*b);
    double g=(b*y-c*x)/det,h=(b*x-a*y)/det;
    return {a,b,c,det,g,h,fmax(0.,rr+g*x+h*y)};
}
struct Normal {
    double a,b,c,x,y,rr;
    VP void add(V d,V g,V h,double w){
        a+=w*product(g,g);b+=w*product(g,h);c+=w*product(h,h);
        x+=w*product(g,d);y+=w*product(h,d);rr+=w*product(d,d);
    }
    VP Posterior fit()const{return solve(a,b,c,x,y,rr);}
};
VP Normal prior(const Edge &b,double sw,double sh,double sx=.24,double sy=.33){
    // A broad disk in transverse velocity. The radial coordinate and
    // the line-of-sight depth are fitted continuously within a family.
    Normal n{1./square(sw),0.,1./square(sh),0.,0.,0.};
    n.add({b.x/sx,b.y/sy},{b.gx/sx,b.gy/sy},{b.hx/sx,b.hy/sy},1.);
    return n;
}
VP double velocity_variance(const Origin &a,const Edge &b,const Edge &c){
    return square(b.noise)+square(c.noise)+square(a.noise*(1./b.dt-1./c.dt));
}
VP V diff(V a,V b){return {a.x-b.x,a.y-b.y};}
VP double photometry(const Edge &b,const Edge &c){
    return square(b.mag-c.mag)/(b.variance+c.variance+square(.27*(b.lever-c.lever)));
}
VP void equation(Normal &n,const Edge &b,const Edge &c,double dg,double h,double weight){
    V g{c.gx-b.gx+2.*dg*(c.xx-b.xx)+h*(c.ghx-b.ghx),
        c.gy-b.gy+2.*dg*(c.yy-b.yy)+h*(c.ghy-b.ghy)};
    V v{c.hx-b.hx+dg*(c.ghx-b.ghx),c.hy-b.hy+dg*(c.ghy-b.ghy)};
    V d=diff(at(c,dg,h),at(b,dg,h));
    n.add({d.x-dg*g.x-h*v.x,d.y-dg*g.y-h*v.y},g,v,weight);
}
struct Trial {Normal n;Posterior p;double merit,variance;int id;};
VP bool triple(const Origin &a,const Edge &b,const Edge &c,const Normal &base,
               double dg,double radial,Trial &out){
    double photo=photometry(b,c);
    if(b.night==c.night||photo>15.)return false;
    double v=velocity_variance(a,b,c);
    // A low-order solar propagation supplies the proposals. Its radial
    // truncation error grows with fractional orbital period.
    double curvature=.00018+.0008*fmax(fabs(b.dt),fabs(c.dt))*a.vc;
    double noise=v+square(curvature);
    Normal n=base;
    equation(n,b,c,dg,radial,1./noise);
    auto p=n.fit();
    if(p.chi>37.||fabs(p.g)>.65||fabs(p.h)>.95)return false;
    if(fabs(p.g-dg)>.025||fabs(p.h-radial)>.15){
        n=base;equation(n,b,c,p.g,p.h,1./noise);p=n.fit();
        if(p.chi>30.||fabs(p.g)>.62||fabs(p.h)>.88)return false;
    }
    // The angular Jacobian converts the density in velocity to the
    // probability of accidentally seeing this particular third source.
    double jac=2.*log(fmax(.01,a.vc*arc*fabs(c.dt)))+log(noise);
    double contrast=-.5*log(p.det/base.fit().det)-jac-c.logrho;
    double merit=c.mark-.38*photo-.5*p.chi+contrast;
    out={n,p,merit,noise,-1};
    return true;
}
VP bool disjoint(const Edge &b,const Edge &c,const Edge &d){
    return b.night!=d.night&&c.night!=d.night;
}
VP double fourth(const Origin &a,const Edge &b,const Edge &c,const Edge &d,const Trial &fit,
                 double &dg,double &h){
    if(!disjoint(b,c,d))return -1.e20;
    const auto &p=fit.p;
    V r=diff(at(d,p.g,p.h),at(b,p.g,p.h));
    V u{d.gx-b.gx+2.*p.g*(d.xx-b.xx)+p.h*(d.ghx-b.ghx),
        d.gy-b.gy+2.*p.g*(d.yy-b.yy)+p.h*(d.ghy-b.ghy)};
    V w{d.hx-b.hx+p.g*(d.ghx-b.ghx),d.hy-b.hy+p.g*(d.ghy-b.ghy)};
    double noise=velocity_variance(a,b,d)+square(.00023+.001*fmax(fabs(b.dt),fabs(d.dt))*a.vc);
    // Shared astrometric errors in the common anchor and in B correlate
    // the two inferred velocities. Condition their residuals together.
    double shared=square(b.noise)+square(a.noise)*(1./c.dt-1./b.dt)*(1./d.dt-1./b.dt);
    double z=shared/fit.variance;
    V rc=diff(at(c,p.g,p.h),at(b,p.g,p.h));
    r.x-=z*rc.x;r.y-=z*rc.y;
    u.x-=z*(c.gx-b.gx+2.*p.g*(c.xx-b.xx)+p.h*(c.ghx-b.ghx));
    u.y-=z*(c.gy-b.gy+2.*p.g*(c.yy-b.yy)+p.h*(c.ghy-b.ghy));
    w.x-=z*(c.hx-b.hx+p.g*(c.ghx-b.ghx));
    w.y-=z*(c.hy-b.hy+p.g*(c.ghy-b.ghy));
    noise=fmax(square(.00020),noise-z*shared);
    double gu=(p.c*u.x-p.b*w.x)/p.det, hu=(p.a*w.x-p.b*u.x)/p.det;
    double gv=(p.c*u.y-p.b*w.y)/p.det, hv=(p.a*w.y-p.b*u.y)/p.det;
    double xx=noise+gu*u.x+hu*w.x,xy=gu*u.y+hu*w.y,yy=noise+gv*u.y+hv*w.y;
    if(r.x*r.x>19.*xx||r.y*r.y>19.*yy)return -1.e20;
    double det=fmax(noise*noise,xx*yy-xy*xy);
    double rx=(yy*r.x-xy*r.y)/det,ry=(xx*r.y-xy*r.x)/det;
    double chi=rx*r.x+ry*r.y;
    if(chi>19.)return -1.e20;
    dg=p.g-gu*rx-gv*ry;h=p.h-hu*rx-hv*ry;
    if(fabs(dg)>.64||fabs(h)>.92)return -1.e20;
    double photo=fmin(photometry(b,d),photometry(c,d));
    return d.mark-.45*photo-.5*chi-.5*log(det)-d.logrho-2.*log(fmax(.01,arc*a.vc*fabs(d.dt)));
}
VP Fit empty_fit(){return {{-1,-1,-1,-1},0.f,0.f,-1.e20f,-1};}
VP Fit proposal(const Origin &a,const Edge &b,const Edge &c,int d,double dg,double h,double score,double dt4=0.){
    // Convert the line-of-sight coordinate and its time derivative to
    // the epoch used for differential corrections. Include solar curvature.
    double mean=(b.dt+c.dt+dt4)/(d>=0?4.:3.);
    double g=a.g*(1.+dg),actual_h=a.vc*(h+a.h*(1.+dg)*sqrt(1.+dg));
    double frequency=gaussian*pow(g*a.inorm,1.5),phase=frequency*mean;
    double f=cos(phase),s=sin(phase)/frequency,z=f+s*actual_h;
    return {{a.row,b.row,c.row,d},float(g/z),float((f*actual_h-frequency*sin(phase))/z),float(score),-1};
}
// B is an independent epoch. Retain some triangles before asking whether
// a fourth measurement in the same foreground catalogue corroborates them.
VP void compare(const Origin &a,const Edge *edges,int n,int ib,int family,Fit *out){
    Fit best=empty_fit(),quad=empty_fit(),leverage=empty_fit();
    if(ib>=n){out[0]=best;out[1]=quad;out[2]=leverage;return;}
    const Edge &b=edges[ib];
    double sw=family==0?.19:.29,sh=family==0?.20:.36;
    double sx=family==2?.73:.24,sy=family==2?.73:.33;
    Normal base=prior(b,sw,sh,sx,sy);auto initial=base.fit();
    Trial winners[6];int nw=0;
    double bscore=a.mark+b.mark-.5*log(initial.det*sw*sw*sh*sh)-b.logrho
           -2.*log(fmax(.1,a.vc*arc*fabs(b.dt)))-log(sx*sy);
    for(int k=0;k<n;++k){
        if(k==ib)continue;
        Trial trial;
        if(!triple(a,b,edges[k],base,initial.g,initial.h,trial))continue;
        trial.id=k;
        const Edge &c=edges[k];
        double span=fmax(fabs(b.dt),fmax(fabs(c.dt),fabs(c.dt-b.dt)));
        double information=.28*log(trial.p.det/initial.det)+.38*log1p(span/120.);
        double priority=bscore+trial.merit+information;
        if(priority>leverage.merit)leverage=proposal(a,b,c,-1,trial.p.g,trial.p.h,priority);
        if(nw==6 && trial.merit<winners[5].merit)continue;
        int j=0;while(j<nw&&winners[j].merit>=trial.merit)++j;
        int upto=nw<6?nw:5;
        for(int t=upto;t>j;--t)winners[t]=winners[t-1];
        if(j<6){winners[j]=trial;if(nw<6)++nw;}
    }
    for(int i=0;i<nw;++i){
        const auto &v=winners[i];const Edge &c=edges[v.id];
        double score=bscore+v.merit;
        if(score>best.merit)best=proposal(a,b,c,-1,v.p.g,v.p.h,score);
        for(int k=0;k<n;++k){
            double dg=0.,h=0.;const Edge &d=edges[k];
            double add=fourth(a,b,c,d,v,dg,h);
            if(score+add>quad.merit&&add>-1.e10){
                quad=proposal(a,b,c,d.row,dg,h,score+add,d.dt);
            }
        }
    }
    out[0]=best;out[1]=quad;out[2]=leverage;
}
}
#include "pencil_forecast.hpp"
#ifdef __CUDACC_RTC__
extern "C" __global__ void velocity_pencils(const pencil::Origin *a,const pencil::Edge *data,
             const int *counts,int stride,int nb,int n,int family,pencil::Fit *out){
    int i=blockIdx.x,j=threadIdx.x+blockIdx.y*blockDim.x;
    if(i<n&&j<nb)pencil::compare(a[i],data+i*stride,counts[i],j,family,out+3*(i*nb+j));
}
#else
extern "C" void pencil_host(const pencil::Origin *a,const pencil::Edge *data,const int *counts,
            int stride,int nb,int n,int family,int threads,pencil::Fit *out){
    std::atomic<int> next{0};std::vector<std::thread> workers;
    for(int i=0;i<threads;i++)workers.emplace_back([&]{
        for(int j;(j=next.fetch_add(1))<n;)
            for(int b=0;b<nb;b++)pencil::compare(a[j],data+j*stride,counts[j],b,family,out+3*(j*nb+b));
    });
    for(auto &worker:workers)worker.join();
}
#include "pencil_catalog.hpp"
#endif
