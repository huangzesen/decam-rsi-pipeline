// At fixed inverse range and radial speed the two positions and transverse
// velocities are linear parameters once the Lagrange coefficients are known.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {
constexpr double mu=.0002959122082855911*1.0013418,arc=206264.80624709636;
constexpr double pi=3.14159265358979323846, twopi=2.*pi;
struct Observation {double x,y,t,ox,oy,oz,sigma;};
struct State {double x,y,g,vx,vy,h;};
struct Family {double weight,radius,spread,mean,along,north,radial,latitude;};
double sq(double x){return x*x;}
using Vec=std::array<double,3>;
Vec cross(Vec a,Vec b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
Vec unit(Vec a){double inv=1./std::max(1.e-12,std::sqrt(dot(a,a)));return {a[0]*inv,a[1]*inv,a[2]*inv};}

struct Motion {
    double r,inva,n,c1,s0;
    Motion(const State &o){
        r=std::sqrt(1.+sq(o.x)+sq(o.y))/o.g;
        inva=std::max(1.e-6,2./r-(sq(o.vx)+sq(o.vy)+sq(o.h))/(sq(o.g)*mu));
        n=std::sqrt(mu*inva*inva*inva);c1=1.-r*inva;
        s0=(o.x*o.vx+o.y*o.vy+o.h)*std::sqrt(inva/mu)/sq(o.g);
    }
    void coefficients(double dt,double &f,double &s)const{
        double m=n*dt,e=m/std::max(.025,1.-c1);
        for(int j=0;j<7;j++){
            double sn=std::sin(e),cs=std::cos(e);
            double d=(e-c1*sn+s0*(1.-cs)-m)/std::max(.00002,1.-c1*cs+s0*sn);
            e-=std::clamp(d,-.8,.8);if(std::abs(d)<2.e-12)break;
        }
        f=1.-2.*sq(std::sin(.5*e))/(r*inva);
        double d=std::abs(e)<.004?e*e*e*(-1./6.+e*e/120.):std::sin(e)-e;
        s=dt+d/n;
    }
};

double profile(const Observation *data,int n,unsigned mask,State o,const Family &fam,Vec pole){
    if(!(o.g>.000007 && o.g<.092))return -1e20;
    double xx[16],yy[16],weight[16],tau[16];
    double chi=0.,lognorm=0.,W=0.,S=0.,vlog=0.,r=0.;int count=0;
    for(int it=0;it<3;it++){
        Motion orbit(o);r=orbit.r;
        double swt=0.,swx=0.,swy=0.;W=0.;lognorm=0.;count=0;
        for(int i=0;i<n;i++)if(mask&(1u<<i)){
            auto &p=data[i];double f,s;
            // First light-time iteration uses the locally propagated radius.
            double t=p.t,delay=std::sqrt(sq(o.x+t*o.vx-o.g*p.ox)+sq(o.y+t*o.vy-o.g*p.oy)+
                               sq(1.+t*o.h-o.g*p.oz))/(173.1446326846693*o.g);
            orbit.coefficients(t-delay,f,s);
            double fx=f*o.x+s*o.vx-o.g*p.ox,fy=f*o.y+s*o.vy-o.g*p.oy,fz=f+s*o.h-o.g*p.oz;
            delay=std::sqrt(fx*fx+fy*fy+fz*fz)/(173.1446326846693*o.g);
            orbit.coefficients(t-delay,f,s);
            if(f<.45)return -1.e20;
            double z=s/f,g=o.g/f,den=1.+z*o.h-g*p.oz;
            if(den<.5)return -1.e20;
            tau[i]=z;double w=1./sq(p.sigma*den);weight[i]=w;
            xx[i]=(p.x*den+g*p.ox)*arc;yy[i]=(p.y*den+g*p.oy)*arc;
            W+=w;swt+=w*z;swx+=w*xx[i];swy+=w*yy[i];
            lognorm-=std::log(twopi*sq(p.sigma));count++;
        }
        double tbar=swt/W,xbar=swx/W,ybar=swy/W,cx=0.,cy=0.;
        S=0.;
        for(int i=0;i<n;i++)if(mask&(1u<<i)){
            double dt=tau[i]-tbar,w=weight[i];S+=w*dt*dt;
            cx+=w*dt*(xx[i]-xbar);cy+=w*dt*(yy[i]-ybar);
        }
        if(!(S>1.e-12))return -1.e20;
        double vx=cx/S,vy=cy/S;
        Vec dir=unit({o.x,o.y,1.}),east=unit(cross(pole,dir)),north=cross(dir,east);
        double om=std::sqrt(mu/r)*o.g*arc;
        std::array<Vec,3> basis{east,north,dir};
        double var[3]={sq(om*fam.along),sq(om*fam.north),sq(om*fam.radial)};
        double cov[3][3]{};
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)
            cov[i][j]+=var[k]*basis[k][i]*basis[k][j];
        double hz=o.h*arc-om*fam.mean*east[2],iz=1./cov[2][2];
        double mx=om*fam.mean*east[0]+cov[0][2]*iz*hz,my=om*fam.mean*east[1]+cov[1][2]*iz*hz;
        double d1=cov[0][0]-sq(cov[0][2])*iz+1./S;
        double d2=cov[1][1]-sq(cov[1][2])*iz+1./S,off=cov[0][1]-cov[0][2]*cov[1][2]*iz;
        double det=d1*d2-off*off;
        if(!(det>0.))return -1.e20;
        double ex=vx-mx,ey=vy-my,ux=(d2*ex-off*ey)/det,uy=(d1*ey-off*ex)/det;
        vlog=std::log(arc)-.5*(sq(hz)*iz+std::log(twopi*cov[2][2]))-
            std::log(twopi)-.5*(std::log(det)+ex*ux+ey*uy);
        if(it==2){
            chi=0.;
            for(int i=0;i<n;i++)if(mask&(1u<<i)){
                chi+=weight[i]*(sq(xx[i]-xbar-(tau[i]-tbar)*vx)+sq(yy[i]-ybar-(tau[i]-tbar)*vy));
            }
            break;
        }
        // Conditional mean of the angular rates for the next two-body fit.
        vx-=ux/S;vy-=uy/S;
        o.vx=vx/arc;o.vy=vy/arc;o.x=(xbar-tbar*vx)/arc;o.y=(ybar-tbar*vy)/arc;
    }
    if(!(r>12.&&r<140000.&&std::isfinite(chi)))return -1e20;
    double logr=fam.spread>0.?
        -.5*sq(std::log(r/fam.radius)/fam.spread)-std::log(std::sqrt(twopi)*fam.spread):
        -std::log(std::log(140000./12.));
    double speed=std::sqrt((sq(o.h)+sq(o.vx)+sq(o.vy))*r/(mu*sq(o.g)));
    double lat=dot(unit({o.x,o.y,1.}),pole);
    double disk=fam.latitude<1.?std::log(std::sqrt(2./pi)/fam.latitude)-.5*sq(lat/fam.latitude):0.;
    double dof=2.*count-4.;
    double ast=std::lgamma(4.+.5*dof)-std::lgamma(4.)-.5*dof*std::log(4.)-
               (4.+.5*dof)*std::log1p(chi/8.);
    return lognorm+2.*std::log(twopi)-std::log(W*S)+vlog+ast+logr+disk+
           std::log(fam.weight/o.g)-.5*sq(std::max(0.,speed-1.41)/.06);
}
}

extern "C" int ranging_nodes(const Observation *data,int n,const unsigned *masks,int nm,
    const State *initial,const Family *family,const double *pole,int nodes,const double *gh,double *values){
    if(n>16||n<3||nodes<=0)return -1;
    for(int k=0;k<nm;k++)for(int j=0;j<nodes;j++){
        int i=k*nodes+j;State o=*initial;o.g=gh[2*i];o.h=gh[2*i+1];
        values[i]=profile(data,n,masks[k],o,*family,{pole[0],pole[1],pole[2]});
    }
    return 0;
}
