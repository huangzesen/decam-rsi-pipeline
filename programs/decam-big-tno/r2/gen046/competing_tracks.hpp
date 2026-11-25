// Independent catalogue detections can favour a stationary or a stellar
// proper-motion trajectory through one of the reported positions.
namespace competitors {
struct Evidence {float fixed,follow,linear,control,stationary,offset;int repeats,epochs;};
struct Hit {double x,y,years,phot,weight;int night;};
struct Likelihood {double value=0.;int row=-1;};
Likelihood peak(const original::Sky &s,const original::Frame &f,double x,double y,
                double mag,double sm,double sig,int skip=-1){
    Likelihood best;
    if(!f.nx||x<f.xmin-2*as||x>f.xmax+2*as||y<f.ymin-2*as||y>f.ymax+2*as)return best;
    int mb=std::clamp(int((mag-10.)*4.),1,78);
    s.nearby(f,x,y,1.65*as,1.65*as,[&](int i){
        if(i==skip||s.flag[i]&1)return;
        double dm=s.mag[i]-mag;
        double var=sq(sig)+sq(1.05/std::max(4.5,double(s.sn[i])));
        double r2=(sq(s.xy[2*i]-x)+sq(s.xy[2*i+1]-y))/sq(as);
        if(r2>16.*var||!(std::abs(dm)<1.25))return;
        double ms=std::hypot(sm,1.09/std::max(4.5,double(s.sn[i])));
        double odds=std::exp(-.5*(r2/var+sq(dm/ms)))/(15.75*var*ms*std::max(.0001,f.rho[mb]));
        if(odds>best.value){best.value=odds;best.row=i;}
    });
    return best;
}
double linear(const original::Sky &s,double x,double y,int member,double mag,double col,
              const std::vector<char> &clear,int &epochs,double &fixed){
    int a=s.ex[member],n0=s.images[a].exposure.night;
    double t0=s.images[a].exposure.t;
    std::vector<Hit> hits;
    for(int e=0;e<int(s.images.size());e++){
        const auto &f=s.images[e];if(!f.nx||!clear[e]||f.exposure.night==n0)continue;
        double years=(f.exposure.t-t0)/365.25;
        double rad=std::min(9.,.85+.90*std::abs(years));
        if(x<f.xmin-rad*as||x>f.xmax+rad*as||y<f.ymin-rad*as||y>f.ymax+rad*as)continue;
        double lever=s.lever[e]-s.lever[a],sm=std::hypot(.24,.40*lever);
        int mb=std::clamp(int((mag-10.)*4),1,78);
        s.nearby(f,x,y,rad*as,rad*as,[&](int i){
            double dx=(s.xy[2*i]-x)/as,dy=(s.xy[2*i+1]-y)/as;
            double dm=s.mag[i]-mag-col*lever;
            if(s.flag[i]&1||dx*dx+dy*dy>rad*rad||!(std::abs(dm)<2.9*sm))return;
            double sig=std::hypot(.12,std::hypot(1.05/s.sn[i],1.05/s.sn[member]));
            hits.push_back({dx,dy,years,-.5*sq(dm/sm)-std::log(std::max(1.e-6,15.75*sm*sq(sig)*f.rho[mb])),
                            .5/sq(sig),f.exposure.night});
        });
    }
    if(hits.size()<2)return 0.;
    // Fit a proper motion from each long-baseline companion. Independent
    // nights must agree in flux as well as in both position coordinates.
    std::vector<Hit> trials=hits;
    std::sort(trials.begin(),trials.end(),[](auto &a,auto &b){return a.phot>b.phot;});
    if(trials.size()>18)trials.resize(18);
    trials.push_back({0.,0.,1.,0.,0.,-1});
    double highest=0.;epochs=0;
    std::sort(hits.begin(),hits.end(),[](auto &a,auto &b){return a.night<b.night;});
    for(auto &trial:trials){
        bool still=trial.night<0;
        if(!still&&std::abs(trial.years)<.36)continue;
        double vx=trial.x/trial.years,vy=trial.y/trial.years,sp=std::hypot(vx,vy);
        if(!still&&(sp<.035||sp>1.0))continue;
        double total=0.,best=0.,baseline=0.;int count=0,night=-1;
        for(auto &r:hits){
            if(r.night!=night){total+=best;count+=best>1.8;best=0.;night=r.night;}
            double dx=r.x-vx*r.years,dy=r.y-vy*r.years;
            double z=r.phot-r.weight*(dx*dx+dy*dy);
            if(z>1.8)baseline=std::max(baseline,std::abs(r.years));
            best=std::max(best,std::min(5.5,z-1.8));
        }
        total+=best;count+=best>1.8;
        if(count<2||baseline<.06)continue;
        if(still){fixed=std::min(20.,total);continue;}
        // The velocity was selected from an aperture that grows with time.
        // Charge the fitted two-dimensional motion its searched area.
        double penalty=std::log1p(sq(.35*baseline/.21))+std::log1p(sq(sp/.12));
        if(total-penalty>highest){highest=total-penalty;epochs=count;}
    }
    return std::min(20.,highest);
}
void assess(const original::Sky &s,const int *rows,int n,double g,Evidence *out){
    if(n<5||n>100||!(g>0.))return;
    std::vector<Sample> v;ExactState model;model.g=g;
    for(int i=0;i<n;i++){
        if(rows[i]<0||rows[i]>=s.size)return;
        auto m=s.sample(rows[i]);v.push_back(m);model.ref+=m.t/n;model.x+=m.x/n;model.y+=m.y/n;
    }
    std::vector<double> error;
    if(!measured_fit(v,model,error)||*std::max_element(error.begin(),error.end())>1.2)return;
    double msum=0.,wt=0.,ls=0.,lm=0.,ll=0.;
    for(int i=0;i<n;i++){
        int r=rows[i];if(!std::isfinite(s.mag[r]))continue;
        double w=1./(sq(.17)+sq(1.09/s.sn[r])),l=s.lever[s.ex[r]];
        msum+=w*s.mag[r];ls+=w*l;lm+=w*l*s.mag[r];ll+=w*l*l;wt+=w;
    }
    if(wt<1.)return;
    double color=std::clamp((lm-ls*msum/wt)/(ll-ls*ls/wt+1./sq(.27)),-.85,.85);
    Kepler orbit(model);std::vector<Vec2> path(s.images.size());
    for(int e=0;e<int(path.size());e++){
        const auto &e0=s.images[e].exposure;
        Sample m{-1,0.,0.,e0.t,e0.ox,e0.oy,e0.oz};double f,t;
        orbit.apparent(m,path[e].x,path[e].y,f,t);
    }
    std::vector<char> clear(path.size());
    for(int i=0;i<n;i++){
        int row=rows[i],a=s.ex[row];auto &base=s.images[a].exposure;
        double x=s.xy[2*row],y=s.xy[2*row+1],mag=s.mag[row];
        if(!std::isfinite(mag))continue;
        double sig=std::hypot(.11,1.05/std::max(4.5,double(s.sn[row])));
        std::vector<double> favors_fixed,favors_moving;
        for(int e=0;e<int(path.size());e++){
            auto &f=s.images[e];Vec2 step=path[e]-path[a];double dist=std::hypot(step.x,step.y)/as;
            // Protect annual parallactic returns and slowly moving objects.
            clear[e]=dist>8.+.95*std::abs(f.exposure.t-base.t)/365.25;
            if(e==a||!f.nx||f.exposure.night!=base.night||dist<.21)continue;
            double l=s.lever[e]-s.lever[a],m=mag+color*l;
            double sm=std::hypot(.16,std::hypot(.18*l,1.09/s.sn[row]));
            auto stop=peak(s,f,x,y,m,std::hypot(sm,.26*l),sig);
            auto move=peak(s,f,x+step.x,y+step.y,m,sm,sig);
            if(std::max(move.value,stop.value)<1.)continue;
            double ratio=std::log1p(.70*stop.value)-std::log1p(.70*move.value);
            ratio*=std::min(1.,sq(dist/.65));
            if(ratio>0)favors_fixed.push_back(ratio);
            if(ratio<0)favors_moving.push_back(-ratio);
        }
        auto strength=[](std::vector<double> &v){
            std::sort(v.begin(),v.end(),std::greater<double>());
            return v.empty()?0.:std::min(9.,v[0]+(v.size()>1?.52*v[1]:0.));
        };
        auto &r=out[i];r.epochs=0;r.fixed=strength(favors_fixed);r.follow=strength(favors_moving);
        r.repeats=int(favors_fixed.size()+favors_moving.size());
        double stationary=0.;
        r.linear=linear(s,x,y,row,mag,color,clear,r.epochs,stationary);
        r.stationary=stationary;
        // A displaced aperture in the same sky and at the same brightness
        // measures chance coincidences and blends with unrelated field stars.
        unsigned h=uint32_t(row)*2654435761u;
        double angle=(h/4294967296.)*6.28318531;
        double radius=as*(22.+(h%19));int dummy=0;stationary=0.;
        r.control=linear(s,x+std::cos(angle)*radius,y+std::sin(angle)*radius,row,mag,color,clear,dummy,stationary);
        r.offset=stationary;
    }
}
}
extern "C" int competing_tracks(const original::Sky *s,const int *rows,const int *ends,const double *range,
                        int n,int threads,double seconds,competitors::Evidence *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int k=0;k<threads;k++)workers.emplace_back([&]{
            // The final ranking runs after discovery workers have been stopped.
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=n)break;
                competitors::assess(*s,rows+ends[i],ends[i+1]-ends[i],range[i],out+ends[i]);
            }
        });
        for(auto &w:workers)w.join();
        return std::min(n,next.load());
    }catch(...){return -1;}
}
