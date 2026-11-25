// A four-or-more-night fit must predict the withheld observations against
// every unused visit. Nearby offsets measure the same opportunity for chance
// matches, including footprint boundaries, stars and correlated backgrounds.
namespace original {
struct Validation {
    int count,rows[48],pad;
    double x[6],ref;
};
struct NightTest {
    int night,visits,controls,exceeded;
    double odds,expected,empirical,residual,area,core;
};
struct Prediction {
    int e;
    double x,y,xx,xy,yy,det,mag,sm,maximum;
};
static bool held_core(const Sky &sky,const Validation &track,int omit,ExactState &fit,
                      std::vector<Sample> &other,double V[6][6],double &color,double &mag,double &lever){
    std::vector<Sample> core;std::vector<double> m;
    for(int i=0;i<track.count;i++){
        auto v=sky.sample(track.rows[i]);
        if(sky.images[sky.ex[v.row]].exposure.night==omit){other.push_back(v);continue;}
        core.push_back(v);m.push_back(sky.mag[v.row]);
    }
    if(core.size()<4||other.empty())return false;
    std::vector<double> err;
    fit={track.x[0],track.x[1],track.x[2],track.x[3],track.x[4],track.x[5],track.ref};
    if(!measured_fit(core,fit,err,8)||*std::max_element(err.begin(),err.end())>.80)return false;
    Kepler orbit(fit);double normal[6][6]{};
    normal[0][0]=normal[1][1]=1.e-10;normal[2][2]=5.;
    normal[3][3]=normal[4][4]=3.;normal[5][5]=9.;
    std::sort(m.begin(),m.end());
    double mid=.5*(m[(m.size()-1)/2]+m[m.size()/2]),sum=0.;
    mag=lever=0.;
    for(size_t j=0;j<core.size();j++){
        auto &v=core[j];double x,y,f,s,J[2][6]{};
        orbit.apparent(v,x,y,f,s);design(fit,v,x,y,f,s,J);
        double inv=1./(.008+sq(.20)/v.weight+.22*sq(err[j]));
        for(int a=0;a<6;a++)for(int b=0;b<6;b++)normal[a][b]+=inv*(J[0][a]*J[0][b]+J[1][a]*J[1][b]);
        double w=1./(.035+sq(1.09/sky.sn[v.row])+sq(.35*(sky.mag[v.row]-mid)));
        sum+=w;mag+=w*sky.mag[v.row];lever+=w*sky.lever[sky.ex[v.row]];
    }
    mag/=sum;lever/=sum;
    double aa=1./sq(.27),ay=0.;
    for(auto &v:core){
        double l=sky.lever[sky.ex[v.row]]-lever,dm=sky.mag[v.row]-mag;
        double w=1./(.035+sq(1.09/sky.sn[v.row])+sq(.35*dm));
        aa+=w*l*l;ay+=w*l*dm;
    }
    color=std::clamp(ay/aa,-.5,.55);
    return invert(normal,V);
}
static void contrast(const Sky &sky,const Validation &track,int omit,NightTest &out,int trials){
    double V[6][6],color,mag,lever;ExactState fit;std::vector<Sample> held;
    if(!held_core(sky,track,omit,fit,held,V,color,mag,lever))return;
    std::vector<int> blocked;
    bool transient=!sky.selected.empty()&&std::all_of(held.begin(),held.end(),
                                      [&](auto &v){return sky.selected[v.row]!=0;});
    for(int i=0;i<track.count;i++){
        int night=sky.images[sky.ex[track.rows[i]]].exposure.night;
        if(night!=omit)blocked.push_back(night);
    }
    std::sort(blocked.begin(),blocked.end());blocked.erase(std::unique(blocked.begin(),blocked.end()),blocked.end());
    if(blocked.size()<4)return;
    Kepler orbit(fit);double mug=mu*std::pow(fit.g,3);
    std::vector<Prediction> probes;
    double scored=-1.e8,actual_chi=0.,actual_area=0.;
    for(int e=0;e<int(sky.images.size());e++){
        auto &im=sky.images[e];const auto &z=im.exposure;
        if(!im.nx||std::binary_search(blocked.begin(),blocked.end(),z.night))continue;
        double dt=z.t-fit.ref,ph=mug*dt*dt,f=1.-.5*ph*(1.-dt*fit.h),s=dt*(1.-ph/6.);
        double den=f+s*fit.h-fit.g*z.oz;
        double x=(f*fit.x+s*fit.vx-fit.g*z.ox)/den,y=(f*fit.y+s*fit.vy-fit.g*z.oy)/den;
        double pad=.0025+.10*ph*ph;
        if(x<im.xmin-pad||x>im.xmax+pad||y<im.ymin-pad||y>im.ymax+pad)continue;
        Sample v{-1,0.,0.,z.t,z.ox,z.oy,z.oz};orbit.apparent(v,x,y,f,s);
        if(x<im.xmin-.0012||x>im.xmax+.0012||y<im.ymin-.0012||y>im.ymax+.0012)continue;
        double J[2][6]{},xx,xy,yy;design(fit,v,x,y,f,s,J);projected(J,V,xx,xy,yy);
        double det=xx*yy-xy*xy;
        if(!(det>0.)||xx+yy>sq(38.)||std::sqrt(det)>230.)continue;
        double m=mag+(sky.lever[e]-lever)*color;
        int mb=std::clamp(int((m-10.)*4.),0,79);
        double sm=std::hypot(.23,.14*std::abs(sky.lever[e]-lever));
        double rho=std::max(1.e-8,(transient?im.transient:im.rho)[mb]);
        double peak=-std::log(15.74961*sm*std::sqrt(det)*rho);
        probes.push_back({e,x,y,xx,xy,yy,det,m,sm,peak});
        for(auto &p:held)if(sky.ex[p.row]==e){
            double dx=(p.x-x)/as,dy=(p.y-y)/as;
            double chi=(yy*dx*dx-2.*xy*dx*dy+xx*dy*dy)/det;
            double sflux=std::hypot(sm,1.09/sky.sn[p.row]);
            double test=peak+std::log(sm/sflux)-.5*(chi+sq((sky.mag[p.row]-m)/sflux));
            if(test>scored){scored=test;actual_chi=chi;actual_area=std::sqrt(det);}
        }
    }
    if(scored< -40.||probes.empty())return;
    // Independent offsets preserve local repeats between exposures. Use a
    // control radius larger than the held-out positional uncertainty.
    double outer=std::clamp(4.*std::sqrt(actual_area),43.,115.);
    double analytic=0.,expected=0.,atleast=0.;
    std::vector<bool> above(trials);
    std::vector<std::pair<double,double>> offsets;
    for(int k=0;k<trials;k++){
        double angle=2.399963229728653*(k+1)+.27*omit;
        double off=as*outer*std::sqrt(1.+3.*(k+.5)/trials);
        offsets.emplace_back(off*std::cos(angle),off*std::sin(angle));
    }
    int effective=0;
    for(auto &p:probes){
        double delta=2.*(p.maximum-scored);
        if(delta<=0.)continue;
        const auto &im=sky.images[p.e];
        int mb=std::clamp(int((p.mag-10.)*4.),0,79);
        double rho=std::max(1.e-9,(transient?im.transient:im.rho)[mb]);
        double limit=std::min(delta,16.);
        double area=4.1887902*std::pow(limit,1.5)*p.sm*std::sqrt(p.det);
        // Empty tiles or chip gaps should not count as an opportunity. The
        // actual source distribution at the offsets estimates their area.
        int coverage=0;double local=0.;
        double reach=std::max(5.,std::min(22.,std::sqrt(limit*std::max(p.xx,p.yy))));
        double wx=as*std::max(reach,std::sqrt(limit*p.xx)),wy=as*std::max(reach,std::sqrt(limit*p.yy));
        for(int k=0;k<trials;k++){
            double cx=p.x+offsets[k].first,cy=p.y+offsets[k].second;
            int hits=0;bool in=false;
            // A broader control aperture estimates the intensity even when
            // the narrow orbit ellipse has no threshold-crossing events.
            sky.nearby(im,cx,cy,wx,wy,[&](int row){
                double dx=(sky.xy[2*row]-cx)/as,dy=(sky.xy[2*row+1]-cy)/as;
                if(std::abs(dx)<reach&&std::abs(dy)<reach){
                    in=true;
                    if((!transient||sky.selected[row])&&std::abs(sky.mag[row]-p.mag)<.62&&!(sky.flag[row]&1))hits++;
                }
                if(above[k]||sky.flag[row]&1||(transient&&!sky.selected[row]))return;
                double chi=(p.yy*dx*dx-2*p.xy*dx*dy+p.xx*dy*dy)/p.det;
                double sm=std::hypot(p.sm,1.09/sky.sn[row]);
                if(chi+sq((sky.mag[row]-p.mag)/sm)-2.*std::log(p.sm/sm)<=delta)above[k]=true;
            });
            coverage+=in;local+=hits;
        }
        double spread=4.*reach*reach*1.24*trials;
        // Use an independently measured nearby magnitude-density correction.
        double ratio=(local+5.)/(spread*rho+5.);
        double active=std::min(1.,.35+2.5*coverage/trials);
        if(coverage)effective++;
        analytic+=area*rho*active;
        expected+=area*rho*std::clamp(ratio,.12,6.)*active;
    }
    if(!effective)return;
    for(bool b:above)atleast+=b;
    double prob=1.-std::exp(-std::min(30.,expected));
    // Smooth the measured tail with the local density calculation. The
    // direct positional controls can reveal coherent catalogue structure.
    double chance=std::clamp((atleast+6.*prob)/(trials+6.),1.e-8,1.-1.e-6);
    out={omit,effective,trials,int(atleast),scored,expected,chance,actual_chi,actual_area,analytic};
}
}

extern "C" int offset_predictions(const original::Sky *sky,const original::Validation *jobs,int n,
                                  int trials,int threads,double seconds,original::NightTest *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int j=0;j<threads;j++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int k=next.fetch_add(1);if(k>=n)break;
                auto &job=jobs[k];std::vector<int> nights;
                for(int i=0;i<job.count;i++)nights.push_back(sky->images[sky->ex[job.rows[i]]].exposure.night);
                std::sort(nights.begin(),nights.end());nights.erase(std::unique(nights.begin(),nights.end()),nights.end());
                if(nights.size()<5||nights.size()>14)continue;
                for(size_t i=0;i<nights.size()&&Clock::now()<end;i++)
                    original::contrast(*sky,job,nights[i],out[k*14+i],trials);
            }
        });
        for(auto &t:pool)t.join();return std::min(n,next.load());
    }catch(...){return -1;}
}
