// Evidence for a sparse assignment, including its 3- and 4-night rivals.
// At fixed distance and radial speed the two transverse coordinates and
// their rates have a small conditional least-squares fit. Integrate the
// remaining two variables around each subset's own solution.
namespace ranging_posterior {
constexpr int longest=8;
struct Observation {double t,ox,oy,oz,x,y,sigma,mag,lever,sn,density;};
struct Subset {
    const Observation *all;
    int ids[longest],n;
    ExactState seed;
    Vec2 east;
    double z0;
    struct Profile {
        double res[2*longest+4]{};
        double logvolume=0.,chi=0.;
        bool valid=false;
    };
    Profile profile(double u,double w)const{
        Profile out;
        double g=z0*std::exp(u),norm=std::sqrt(1.+sq(seed.x)+sq(seed.y));
        double distance=norm/g;
        if(!(distance>10.&&distance<12500.)||std::abs(w)>3.5)return out;
        double vc=std::sqrt(mu*g*g*g/norm);
        double broad=std::clamp(std::log(distance/75.)/1.5,0.,1.);
        double vel_sigma=(.37+.46*broad)*vc/as,mean_speed=(1.-.77*broad)*vc;
        double h=w*vc;
        ExactState s=seed;s.g=g;s.h=h;
        double det=0.,C[longest],T[longest],X[longest],Y[longest];
        // Fix an origin close to the measured position to preserve precision
        // in the high-leverage short-arc normal equations.
        for(int it=0;it<3;++it){
            Kepler orbit(s);
            double aa=0.,ab=0.,bb=1./sq(vel_sigma);
            double xa=0.,ya=0.,xb=mean_speed*east.x/as*bb,yb=mean_speed*east.y/as*bb;
            for(int i=0;i<n;++i){
                const auto &o=all[ids[i]];
                double f,st,px,py;
                Sample p{-1,o.x,o.y,o.t,o.ox,o.oy,o.oz};
                orbit.apparent(p,px,py,f,st);
                double den=f+st*h-g*o.oz;
                if(!(den>.2&&std::isfinite(st)))return out;
                double c=C[i]=f/(den*o.sigma),t=T[i]=st/(den*o.sigma);
                X[i]=(o.x*den+g*o.ox-f*seed.x)/(as*den*o.sigma);
                Y[i]=(o.y*den+g*o.oy-f*seed.y)/(as*den*o.sigma);
                aa+=c*c;ab+=c*t;bb+=t*t;
                xa+=X[i]*c;ya+=Y[i]*c;xb+=X[i]*t;yb+=Y[i]*t;
            }
            det=aa*bb-ab*ab;
            if(!(det>1.e-8))return out;
            double ax=(bb*xa-ab*xb)/det,ay=(bb*ya-ab*yb)/det;
            double vx=(aa*xb-ab*xa)/det,vy=(aa*yb-ab*ya)/det;
            s.x=seed.x+ax*as;s.y=seed.y+ay*as;s.vx=vx*as;s.vy=vy*as;
            if(it==2){
                for(int i=0;i<n;++i){
                    out.res[2*i]=C[i]*ax+T[i]*vx-X[i];
                    out.res[2*i+1]=C[i]*ay+T[i]*vy-Y[i];
                }
            }
        }
        // The broad component permits genuine scattered and distant orbits.
        double distance_width=.53+1.3*broad;
        double radial_width=.43+.30*broad;
        double vr=(s.h+s.x*s.vx+s.y*s.vy)/(vc*norm);
        out.res[2*n]=std::log(distance/43.)*(1.-.87*broad)/distance_width;
        out.res[2*n+1]=(s.vx-mean_speed*east.x)/(as*vel_sigma);
        out.res[2*n+2]=(s.vy-mean_speed*east.y)/(as*vel_sigma);
        out.res[2*n+3]=vr/radial_width;
        for(int k=0;k<2*n+4;++k)out.chi+=sq(out.res[k]);
        // Nuisance integral in (sky arcseconds, arcseconds/day). The two
        // log-range and radial-coordinate priors use the same units as the
        // quadrature. Constants cancel only between equal memberships.
        out.logvolume=-std::log(det*sq(vel_sigma)*distance_width*radial_width);
        double excess=std::max(0.,energy(s)-1.993);
        out.logvolume-=std::min(65.,excess/.035);
        out.valid=std::isfinite(out.chi+out.logvolume);return out;
    }
    double integral()const{
        double u=0.,h=seed.h/std::sqrt(mu*std::pow(z0,3)/std::sqrt(1.+sq(seed.x)+sq(seed.y)));
        double aa=0.,ab=0.,bb=0.,det=0.;
        for(int it=0;it<5;++it){
            auto p=profile(u,h);if(!p.valid)return -1.e8;
            constexpr double step=.000021;
            auto dg=profile(u+step,h),dh=profile(u,h+step);
            if(!dg.valid||!dh.valid)return -1.e8;
            aa=.02;ab=0.;bb=.02;double a=0.,b=0.;
            for(int k=0;k<2*n+4;++k){
                double x=(dg.res[k]-p.res[k])/step,y=(dh.res[k]-p.res[k])/step;
                aa+=x*x;ab+=x*y;bb+=y*y;a-=x*p.res[k];b-=y*p.res[k];
            }
            det=aa*bb-ab*ab;if(!(det>0.))return -1.e8;
            double du=std::clamp((bb*a-ab*b)/det,-.30,.30);
            double dw=std::clamp((aa*b-ab*a)/det,-.5,.5);
            if(it==4||aa*du*du+2.*ab*du*dw+bb*dw*dw<.001)break;
            for(double shrink:{1.,.5,.2,.06}){
                auto trial=profile(u+shrink*du,h+shrink*dw);
                if(trial.valid && trial.chi<p.chi){
                    u+=shrink*du;h+=shrink*dw;break;
                }
            }
        }
        // In a short arc, range and radial rate can be almost perfectly
        // correlated. The Cholesky factors preserve that narrow direction.
        double lu=std::sqrt(bb/det),lh=-ab/(lu*det),lv=1./std::sqrt(bb);
        constexpr double nodes[]={-2.8569700139,-1.35562617997,0.,1.35562617997,2.8569700139};
        constexpr double wt[]={.01125741133,.22207592201,.53333333333,.22207592201,.01125741133};
        double highest=-1.e100,sum=0.;
        for(int i=0;i<5;i++)for(int j=0;j<5;j++){
            double v=nodes[i],z=nodes[j];
            auto p=profile(u+lu*v,h+lh*v+lv*z);
            if(!p.valid)continue;
            double term=p.logvolume-.5*p.chi+.5*(v*v+z*z)+std::log(wt[i]*wt[j]);
            if(term>highest){sum=sum*std::exp(highest-term)+1.;highest=term;}
            else sum+=std::exp(term-highest);
        }
        if(sum==0.)return -1.e8;
        double logz=highest+std::log(sum)+std::log(6.28318530718)-.5*std::log(det);
        // Marginalize band slope and mean magnitude within each membership.
        // The likelihood ratio uses the differential magnitude density of
        // foreground-selected detections, measured at the actual position.
        double m0=all[ids[0]].mag;
        double a=0.,b=0.,c=1./sq(.30),x=0.,y=0.,yy=0.;
        for(int i=0;i<n;i++){
            const auto &o=all[ids[i]];
            double variance=sq(.17)+sq(1.086/o.sn),w=1./variance,dm=o.mag-m0+1.1/sq(o.sn);
            a+=w;b+=w*o.lever;c+=w*sq(o.lever);x+=w*dm;y+=w*dm*o.lever;yy+=w*dm*dm;
            logz-=std::log(6.28318530718*sq(o.sigma)*o.density)+.5*std::log(6.28318530718*variance);
        }
        double disc=a*c-b*b;
        double m=(c*x-b*y)/disc,col=(a*y-b*x)/disc;
        logz+=-.5*std::max(0.,yy-m*x-col*y)+.5*std::log(6.28318530718/disc)-std::log(.30);
        logz+=.46*std::log(10.)*std::clamp(m+m0-23.,-6.,4.)-2.*std::log(3600.);
        return std::isfinite(logz)?logz:-1.e8;
    }
};
inline double add_logs(double a,double b){
    return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}
}

extern "C" double sparse_membership_evidence(const ranging_posterior::Observation *p,int n,
                                    const double *x,double ref,const double *east,double *full){
    using namespace ranging_posterior;
    if(n<5||n>longest||!(x[2]>1./10000.&&x[2]<.092))return -1.e8;
    ExactState state{x[0],x[1],x[2],x[3],x[4],x[5],ref};
    Subset test{p,{},0,state,{east[0],east[1]},x[2]};
    double log_true=-1.e8,log_false=0.;
    for(int mask=0;mask<(1<<n);++mask){
        test.n=0;
        for(int j=0;j<n;++j)if(mask&(1<<j))test.ids[test.n++]=j;
        if(test.n<3)continue;
        double integral=test.integral();
        if(integral<-1.e7)continue;
        if(test.n==n)*full=integral;
        double log_weight=integral;
        for(int j=0;j<n;++j)if(!(mask&(1<<j))){
            double missing=.44+.29*std::exp(-std::max(0.,p[j].sn-4.5)/2.);
            log_weight+=std::log(missing);
        }
        // Marginalize possible memberships. Competing partial orbits account
        // for a genuine short arc accidentally completed by other peaks.
        if(test.n>=5)log_true=add_logs(log_true,log_weight);
        else log_false=add_logs(log_false,log_weight);
    }
    return log_true-log_false;
}
