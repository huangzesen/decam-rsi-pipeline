// Approximate marginal likelihood for the bounded native verification queue.
// The final reports use the full nonlinear orbit Jacobian and footprint.
namespace cadence {
double evidence(const Witness &w,const Certified &c,const double *depth,const double *color,
                const double *east){
    int order[64];int n=0;
    for(int i=0;i<c.count;i++){
        int row=c.row[i],e=w.objects[row].exposure,night=w.images[e].e.night;
        int j=0;for(;j<n;j++){
            int k=order[j],r=c.row[k];
            if(w.images[w.objects[r].exposure].e.night!=night)continue;
            double s1=w.sigma?w.sigma[row]:.18,s2=w.sigma?w.sigma[r]:.18;
            if(sq(c.err[i]/s1)+2.*std::log(s1)<sq(c.err[k]/s2)+2.*std::log(s2))order[j]=i;
            break;
        }
        if(j==n)order[n++]=i;
    }
    const double g=c.x[2],vc=std::sqrt(mu*g*g*g);
    double sw=0.,st=0.,stt=4.,gg=2.7,hh=6.,gh=0.,rhs[2][4]{};
    double weights=0.,lev=0.,mag=0.,ll=0.,lm=0.,mm=0.,logp=0.;
    for(int j=0;j<n;j++){
        int k=order[j],row=c.row[k];const auto &p=w.objects[row];int e=p.exposure;
        const auto &f=w.images[e];const auto &z=f.e;
        double signal=std::clamp(5.*std::pow(10.,.4*(depth[e]-p.mag)),4.5,60.);
        double sig=w.sigma?std::clamp(double(w.sigma[row]),.065,.38):std::hypot(.085,.98/signal);
        double weight=1./sq(sig),dt=z.t-c.ref,t=dt*vc/as;
        sw+=weight;st+=weight*t;stt+=weight*t*t;
        for(int a=0;a<2;a++){
            double theta=a?p.y:p.x,obs=a?z.oy:z.ox;
            double A=(obs-theta*z.oz)*g/as,B=theta*t;
            gg+=weight*A*A;hh+=weight*B*B;gh+=weight*A*B;
            auto &h=rhs[a];
            h[0]+=weight*A;h[1]+=weight*A*t;h[2]+=weight*B;h[3]+=weight*B*t;
        }
        double ms=std::hypot(.17,1.09/signal),wt=1./sq(ms),m=p.mag-23.,l=color[e];
        weights+=wt;lev+=wt*l;mag+=wt*m;ll+=wt*l*l;lm+=wt*l*m;mm+=wt*m*m;
        int bin=std::clamp(int((p.mag-10.)*4.),1,78);
        logp+=-std::log(6.2831853*sq(sig)*std::max(1.e-8,f.density[bin]))
              -.5*sq(c.err[k]/sig)-std::log(2.5066283*ms);
    }
    double det=sw*stt-st*st;
    if(!(det>0.&&weights>0.))return -50.;
    for(const auto &h:rhs){
        double ga=(stt*h[0]-st*h[1])/det,gb=(sw*h[1]-st*h[0])/det;
        gg-=ga*h[0]+gb*h[1];gh-=ga*h[2]+gb*h[3];
        hh-=(stt*sq(h[2])-2.*st*h[2]*h[3]+sw*sq(h[3]))/det;
    }
    double precision=ll-lev*lev/weights+1./sq(.26),slope=lm-lev*mag/weights;
    double chi=mm-mag*mag/weights-slope*slope/precision;
    logp+=-.5*std::max(0.,chi)+.5*std::log(6.2831853/(weights*precision*sq(.26)));
    logp+=3.*std::log(6.2831853)-std::log(det)-.5*std::log(std::max(1.,gg*hh-gh*gh));
    double u=(c.x[3]-c.x[0]*c.x[5])/vc,v=(c.x[4]-c.x[1]*c.x[5])/vc;
    double along=u*east[0]+v*east[1],cross=v*east[0]-u*east[1];
    double radial=(c.x[3]*c.x[0]+c.x[4]*c.x[1]+c.x[5])/vc;
    double hot=-2.5-.5*(sq((along-1.)/.48)+sq(cross/.58)+sq(radial/.55));
    double disk=-.5*(sq((along-1.)/.17)+sq(cross/.21)+sq(radial/.25));
    double radius=std::sqrt(1.+sq(c.x[0])+sq(c.x[1]))/g;
    double velocity=std::max(disk,hot);
    if(radius>95.)velocity=std::max(velocity-1.2,-1.8-.5*(sq(u)+sq(v)+sq(radial))/sq(.68));
    return std::clamp(logp+velocity-2.*std::log(3600.)+
             std::log(.18+std::exp(-.5*sq(std::log(radius/44.)/.29)))+.97*mag/weights,-65.,100.);
}
}
