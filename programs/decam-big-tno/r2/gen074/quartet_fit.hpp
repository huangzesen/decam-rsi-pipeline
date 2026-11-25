// Four positions give two independent astrometric residuals after ranging.
// The same small least-squares calculation runs before the host or device cap.
#ifdef __CUDACC__
#define QINLINE __device__ inline
#else
#define QINLINE inline
#endif
namespace quartet_fit {
constexpr double arc=206264.80624709636,mu=2.96309e-4;
struct Shutter {double t,ox,oy,oz;};
struct State {double x,y,g,vx,vy,h,ref;};
QINLINE double sq(double x){return x*x;}
QINLINE void lagrange(const State &s,double t,double &f,double &u){
    double r=sqrt(1.+sq(s.x)+sq(s.y))/s.g;
    double alpha=fmax(1.e-5,2./r-(sq(s.vx)+sq(s.vy)+sq(s.h))/(mu*sq(s.g)));
    double n=sqrt(mu*alpha*alpha*alpha);
    double sin0=(s.x*s.vx+s.y*s.vy+s.h)/sq(s.g)*sqrt(alpha/mu);
    double cos0=1.-r*alpha,M=n*t,E=M;
    for(int j=0;j<6;j++){
        double step=(E-cos0*sin(E)+sin0*(1.-cos(E))-M)/(1.-cos0*cos(E)+sin0*sin(E));
        E-=step;if(fabs(step)<2.e-12)break;
    }
    f=1.-2.*sq(sin(.5*E))/(r*alpha);u=t+(sin(E)-E)/n;
}
template<int N>
QINLINE bool positions(const double (&xy)[2][N],const Shutter *e,double initial,State &o,double &rms){
    o={0.,0.,initial,0.,0.,0.,0.};
    double first=e[0].t,last=first;
    for(int i=0;i<N;i++){
        o.ref+=e[i].t/N;o.x+=xy[0][i]/N;o.y+=xy[1][i]/N;
        first=fmin(first,e[i].t);last=fmax(last,e[i].t);
    }
    // Use a mid-arc emission epoch. The differential light time enters the
    // final iterations; the shared delay changes only the state epoch.
    double span=last-first;
    if(span<.5)return false;
    double f[N],u[N],tt[N],dt[N];
    double zdepth=sqrt(1.+sq(o.x)+sq(o.y))/o.g;
    for(int i=0;i<N;i++)dt[i]=e[i].t-o.ref;
    for(int iter=0;iter<8;iter++){
        double mean=0.,den=0.,aa=sq(.22/(arc*.48*initial));
        double bb=sq(.22/arc)/(.32*mu*initial*initial*initial),ab=0.,ag=aa*initial,bh=0.;
        double intercept[2][3],rate[2][3];
        for(int i=0;i<N;i++){
            if(!iter){
                double n=sqrt(mu/(zdepth*zdepth*zdepth));
                f[i]=cos(n*dt[i]);u[i]=sin(n*dt[i])/n;
            }else lagrange(o,dt[i],f[i],u[i]);
            tt[i]=u[i]/f[i];mean+=tt[i]/N;
        }
        for(int i=0;i<N;i++)den+=sq(tt[i]-mean);
        for(int k=0;k<2;k++){
            double resid[3][N];
            for(int j=0;j<3;j++){
                double avg=0.,vel=0.;
                for(int i=0;i<N;i++){
                    double obs=k?e[i].oy:e[i].ox;
                    double val=j==0?xy[k][i]:j==1?(obs-xy[k][i]*e[i].oz)/f[i]:xy[k][i]*tt[i];
                    resid[j][i]=val;avg+=val/N;vel+=(tt[i]-mean)*val;
                }
                vel/=den;intercept[k][j]=avg-mean*vel;rate[k][j]=vel;
                for(int i=0;i<N;i++)resid[j][i]-=avg+(tt[i]-mean)*vel;
            }
            for(int i=0;i<N;i++){
                double p=resid[1][i],v=resid[2][i],y=resid[0][i];
                aa+=p*p;ab+=p*v;bb+=v*v;ag-=p*y;bh-=v*y;
            }
        }
        double det=aa*bb-ab*ab, g=(ag*bb-bh*ab)/det, h=(bh*aa-ag*ab)/det;
        if(!(g>1./14000. && g<1./13.)||fabs(h)>2.5*sqrt(mu*g*g*g))return false;
        double change=fabs(g-o.g)/g+fabs(h-o.h)/sqrt(mu*g*g*g);
        o.g=g;o.h=h;
        o.x=intercept[0][0]+g*intercept[0][1]+h*intercept[0][2];
        o.y=intercept[1][0]+g*intercept[1][1]+h*intercept[1][2];
        o.vx=rate[0][0]+g*rate[0][1]+h*rate[0][2];
        o.vy=rate[1][0]+g*rate[1][1]+h*rate[1][2];
        for(int i=0;i<N;i++){
            double dz=(f[i]+u[i]*h)/g-e[i].oz;
            dt[i]=e[i].t-o.ref-(dz*sqrt(1.+sq(xy[0][i])+sq(xy[1][i]))-zdepth)/173.1446327;
        }
        if(iter>1&&change<1.e-6)break;
    }
    double energy=(sq(o.vx)+sq(o.vy)+sq(o.h))*sqrt(1.+sq(o.x)+sq(o.y))/(mu*o.g*o.g*o.g);
    if(!(energy<2.015))return false;
    double err=0.;
    for(int i=0;i<N;i++){
        lagrange(o,dt[i],f[i],u[i]);
        double z=f[i]+u[i]*o.h-o.g*e[i].oz;
        double dx=((f[i]*o.x+u[i]*o.vx-o.g*e[i].ox)/z-xy[0][i])*arc;
        double dy=((f[i]*o.y+u[i]*o.vy-o.g*e[i].oy)/z-xy[1][i])*arc;
        err+=dx*dx+dy*dy;
    }
    rms=sqrt(err/N);o.ref-=zdepth/173.1446327;
    return rms<.57;
}
QINLINE bool solve(const float *a,const float *b,const Shutter *e,double initial,State &o,double &rms){
    double xy[2][4]={{a[0]/arc,(double(a[0])+a[2])/arc,b[0]/arc,(double(b[0])+b[2])/arc},
                     {a[1]/arc,(double(a[1])+a[3])/arc,b[1]/arc,(double(b[1])+b[3])/arc}};
    return positions(xy,e,initial,o,rms);
}

struct Forecast {
    double mean,st2,model[2][2],rate[2][2],gg,gh,hh,r0,arc_days,variance;
};
QINLINE void angle(const State &o,const Shutter &e,double r0,double &x,double &y,
                   double &f,double &u){
    double delay=r0/173.1446327;
    for(int j=0;j<2;j++){
        lagrange(o,e.t-o.ref-delay,f,u);
        double z=f+u*o.h-o.g*e.oz;
        x=(o.x*f+u*o.vx-o.g*e.ox)/z;y=(o.y*f+u*o.vy-o.g*e.oy)/z;
        delay=z*sqrt(1.+x*x+y*y)/(o.g*173.1446327);
    }
}
template<int N=4>
QINLINE Forecast uncertainty(const State &o,const Shutter *e,double rms){
    Forecast m{};m.r0=sqrt(1.+sq(o.x)+sq(o.y))/o.g;
    m.arc_days=fmax(1.,e[N-1].t-e[0].t);m.variance=sq(fmax(.19,fmin(.35,rms+.06)));
    double tt[N],der[2][2][N];
    for(int i=0;i<N;i++){
        double x,y,f,u;angle(o,e[i],m.r0,x,y,f,u);tt[i]=u/f;m.mean+=tt[i]/N;
        der[0][0][i]=(e[i].ox-x*e[i].oz)/f;der[1][0][i]=(e[i].oy-y*e[i].oz)/f;
        der[0][1][i]=tt[i]*x;der[1][1][i]=tt[i]*y;
    }
    for(int i=0;i<N;i++)m.st2+=sq(tt[i]-m.mean);
    double aa=m.variance/sq(arc*.29*o.g),ab=0.,bb=m.variance/(sq(arc)*.23*mu*o.g*o.g*o.g);
    for(int k=0;k<2;k++){
        for(int j=0;j<2;j++){
            for(int i=0;i<N;i++){
                m.model[k][j]+=der[k][j][i]/N;
                m.rate[k][j]+=(tt[i]-m.mean)*der[k][j][i]/m.st2;
            }
            for(int i=0;i<N;i++)der[k][j][i]-=m.model[k][j]+(tt[i]-m.mean)*m.rate[k][j];
        }
        for(int i=0;i<N;i++){aa+=sq(der[k][0][i]);ab+=der[k][0][i]*der[k][1][i];bb+=sq(der[k][1][i]);}
    }
    double det=fmax(1.e-28,aa*bb-ab*ab);
    m.gg=m.variance*bb/det;m.gh=-m.variance*ab/det;m.hh=m.variance*aa/det;return m;
}
QINLINE void ellipse(const State &o,const Forecast &m,const Shutter &e,double x,double y,
                    double f,double u,double &xx,double &xy,double &yy){
    double t=u/f,tau=t-m.mean;
    double p=(e.ox-x*e.oz)/f-m.model[0][0]-tau*m.rate[0][0];
    double q=(e.oy-y*e.oz)/f-m.model[1][0]-tau*m.rate[1][0];
    double h=x*t-m.model[0][1]-tau*m.rate[0][1],k=y*t-m.model[1][1]-tau*m.rate[1][1];
    double floor=m.variance*(1.35+tau*tau/m.st2);
    double scale=sq(f/(f+u*o.h-o.g*e.oz));
    xx=.07+scale*(floor+p*p*m.gg+2.*p*h*m.gh+h*h*m.hh);
    xy=scale*(p*q*m.gg+(h*q+k*p)*m.gh+h*k*m.hh);
    yy=.07+scale*(floor+q*q*m.gg+2.*q*k*m.gh+k*k*m.hh);
}
QINLINE void coarse(const State &o,const Forecast &m,const Shutter &e,double &x,double &y,double &pad){
    double t=e.t-o.ref-m.r0/173.1446327;
    double kt=mu*sq(t)/(m.r0*m.r0*m.r0);
    double rv=(o.x*o.vx+o.y*o.vy+o.h)/(1.+sq(o.x)+sq(o.y));
    double f=1.-.5*kt*(1.-t*rv),u=t*(1.-kt/6.+kt*t*rv/4.);
    double z=f+u*o.h-o.g*e.oz;
    x=arc*(o.x*f+u*o.vx-o.g*e.ox)/z;y=arc*(o.y*f+u*o.vy-o.g*e.oy)/z;
    pad=40.+fmin(15000.,.3*arc*kt*kt*(.03+fabs(t)*sqrt(mu/(m.r0*m.r0*m.r0))));
}
}
