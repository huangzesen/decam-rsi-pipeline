// Conditional on a measured position, compare the slow orbit with a
// foreground asteroid. A second independent exposure must confirm the
// faster interpretation. Rates and apertures use the actual observer.
namespace {
namespace local_motion {
struct Request {double state[6],ref;int row,group;};
struct Result {float fast,slow;int partner,confirm,slow_row,slow_e;};
struct Peer {int row,ex;double gamma,score,rate;};
struct Near {int row=-1;double odds=-30.;};

struct Blend {float star,excess,reference;int nights;};

void blend(const original::Sky &sky,const Request &q,double moving,Blend &answer){
    int row=q.row;
    if(row<0||row>=sky.size||!std::isfinite(moving))return;
    int base=sky.ex[row];
    const auto &epoch=sky.images[base].exposure;
    Vec2 point{sky.xy[2*row],sky.xy[2*row+1]};
    double observed=sky.mag[row];
    if(!std::isfinite(observed)||sky.flag[row]&1)return;
    ExactState state{q.state[0],q.state[1],q.state[2],q.state[3],q.state[4],q.state[5],q.ref};
    if(!(state.g>1./9400.&&state.g<1./11.))return;
    Kepler orbit(state);
    struct Observation {int night;double mag,ds;};
    std::vector<Observation> obs;
    for(size_t e=0;e<sky.images.size();e++){
        const auto &im=sky.images[e];const auto &v=im.exposure;
        if(std::abs(v.t-epoch.t)<2.||sky.lever[e]!=sky.lever[base]||!im.nx||
           point.x<im.xmin||point.x>im.xmax||point.y<im.ymin||point.y>im.ymax)continue;
        int match=-1;double dist=.58*.58;
        sky.nearby(im,point.x,point.y,.58*as,.58*as,[&](int r){
            if(sky.flag[r]&1||!std::isfinite(sky.mag[r])||sky.sn[r]<4.7)return;
            double d=sq((sky.xy[2*r]-point.x)/as)+sq((sky.xy[2*r+1]-point.y)/as);
            if(d<dist){dist=d;match=r;}
        });
        if(match<0)continue;
        Sample sample{-1,0.,0.,v.t,v.ox,v.oy,v.oz};
        double x,y,f,s;orbit.apparent(sample,x,y,f,s);
        if(sq((x-point.x)/as)+sq((y-point.y)/as)<sq(4.5))continue;
        obs.push_back({v.night,sky.mag[match],dist});
    }
    if(obs.size()<2)return;
    std::sort(obs.begin(),obs.end(),[](auto &a,auto &b){return a.night!=b.night?a.night<b.night:a.ds<b.ds;});
    std::vector<double> mags,offsets;
    for(size_t j=0;j<obs.size();j++){
        if(j&&obs[j].night==obs[j-1].night)continue;
        mags.push_back(obs[j].mag);offsets.push_back(std::sqrt(obs[j].ds));
    }
    if(mags.size()<2)return;
    auto median=[](std::vector<double> &a){
        std::sort(a.begin(),a.end());
        return .5*(a[a.size()/2]+a[(a.size()-1)/2]);
    };
    double mr=median(mags),separation=median(offsets);
    for(auto &m:mags)m=std::abs(m-mr);
    double scatter=1.4826*median(mags);
    double sig=std::hypot(std::hypot(.11,1.086/std::max(4.5,double(sky.sn[row]))),scatter);
    if(scatter>.6)return;
    // Fluxes of an unresolved pair add in the same image. The median
    // reference includes only other nights with a spatially separated orbit.
    double ratio=std::pow(10.,std::clamp(.4*(mr-moving),-5.,5.));
    double blendmag=mr-2.5*std::log10(1.+ratio);
    double sb=std::hypot(sig,.23*ratio/(1.+ratio));
    double star_chi=sq((observed-mr)/sig);
    double blend_chi=sq((observed-blendmag)/sb);
    double delta=.5*(blend_chi-star_chi)+std::log(sb/sig);
    double precision=std::clamp((.58-separation)/.32,0.,1.)*std::min(1.,.45*mags.size());
    answer={float(precision*std::clamp(delta,-16.,16.)),
            float((mr-observed)/sig),float(mr),int(mags.size())};
}

Near nearby(const original::Sky &sky,int e,Vec2 p,double sigma,double m,double phot){
    const auto &im=sky.images[e];Near best;
    if(!im.nx||p.x<im.xmin-3.*sigma*as||p.x>im.xmax+3.*sigma*as||
       p.y<im.ymin-3.*sigma*as||p.y>im.ymax+3.*sigma*as)return best;
    double rad=std::min(11.,3.65*sigma);
    sky.nearby(im,p.x,p.y,rad*as,rad*as,[&](int j){
        if(sky.flag[j]&1||!std::isfinite(sky.mag[j]))return;
        double dd=sq((sky.xy[2*j]-p.x)/as)+sq((sky.xy[2*j+1]-p.y)/as);
        if(dd>rad*rad)return;
        double dm=sky.mag[j]-m;
        if(std::abs(dm)>3.5*phot)return;
        int mb=std::clamp(int((sky.mag[j]-10.)*4.),0,79);
        double lr=-std::log(15.7496*sq(sigma)*phot*std::max(1.e-8,im.rho[mb]))
                 -.5*(dd/sq(sigma)+sq(dm/phot));
        if(lr>best.odds)best={j,lr};
    });
    return best;
}
void inspect(const original::Sky &sky,const Request &q,const double *east,Result &out,
             const std::vector<int> &dates,const std::unordered_set<int> &members){
    out={0.,0.,-1,-1,-1,-1};
    int row=q.row;if(row<0||row>=sky.size)return;
    int base=sky.ex[row];const auto &from=sky.images[base].exposure;
    if(!std::isfinite(sky.mag[row]))return;
    ExactState orbit0{q.state[0],q.state[1],q.state[2],q.state[3],q.state[4],q.state[5],q.ref};
    if(!(orbit0.g>1./9400.&&orbit0.g<1./11.))return;
    Kepler orbit(orbit0);
    Vec2 p{sky.xy[2*row],sky.xy[2*row+1]},dir{east[0],east[1]};
    auto predict=[&](int e){
        auto &v=sky.images[e].exposure;
        Sample sn{-1,0.,0.,v.t,v.ox,v.oy,v.oz};
        double x,y,f,s;orbit.apparent(sn,x,y,f,s);
        return Vec2{x,y};
    };
    Vec2 fitted=predict(base),offset=p-fitted;
    if(dot(offset,offset)>sq(1.3*as))return;
    double m=sky.mag[row],sn=sky.sn[row],slow_support=0.;
    std::vector<Peer> peers;std::vector<int> adjacent;
    for(int e:dates){
        auto &im=sky.images[e];double dt=im.exposure.t-from.t;
        if(std::abs(dt)>.0&&std::abs(dt)<1.75&&im.nx&&from.night!=im.exposure.night)
            adjacent.push_back(e);
    }
    for(int e:dates){
        const auto &im=sky.images[e];auto &to=im.exposure;double dt=to.t-from.t;
        if(!im.nx||std::abs(dt)<.0020||std::abs(dt)>1.72)continue;
        Vec2 slow=predict(e)+offset;
        double move=std::sqrt(dot(slow-p,slow-p))/as;
        if(slow.x>im.xmin-.001&&slow.x<im.xmax+.001&&slow.y>im.ymin-.001&&slow.y<im.ymax+.001){
            double pm=std::hypot(.26,.20*std::abs(sky.lever[base]-sky.lever[e]));
            auto at=nearby(sky,e,slow,std::hypot(.19,.95/std::max(4.5,sn)),m,pm);
            if(at.row>=0&&move>.48){
                // Against a static centroid in another visit of this night.
                double pos=std::min(1.,sq(move/.95));
                double support=std::clamp(at.odds-2.,0.,9.)*pos;
                slow_support=std::max(slow_support,support);
                if(!members.count(at.row)&&support>out.slow){
                    out.slow=support;out.slow_row=at.row;out.slow_e=e;
                }
            }
        }
        if(std::abs(dt)>.34&&adjacent.size()<2)continue;
        // In each range slice project circular prograde motion, with a
        // dispersion in all velocity components. Use the predicted position
        // and the measured centre to set the orbit of the fast alternative.
        Three obs{from.ox,from.oy,from.oz},ray{p.x,p.y,1.};ray=ray*(1./ray.length());
        double depth=ray.product(obs),impact=obs.product(obs)-depth*depth;
        std::vector<Peer> local;
        for(double r:{1.65,2.05,2.48,2.95,3.65,5.25,8.0}){
            double range=std::sqrt(r*r-impact)-depth,g=1./(from.oz+ray.z*range);
            if(!(range>.2&&g>0.))continue;
            Vec2 origin=p*(1.-g*from.oz)+Vec2{from.ox,from.oy}*g;
            double vc=std::sqrt(mu/r)*g;
            double f=1.-.5*mu*dt*dt/(r*r*r);
            Vec2 predicted=(origin*f+dir*(vc*dt)-Vec2{to.ox,to.oy}*g)*(1./(f-g*to.oz));
            double sigma=std::hypot(.34,.28*std::abs(vc*dt)/as/(f-g*to.oz));
            double rad=2.65*sigma*as;
            if(predicted.x+rad<im.xmin||predicted.x-rad>im.xmax||
               predicted.y+rad<im.ymin||predicted.y-rad>im.ymax)continue;
            auto best=nearby(sky,e,predicted,sigma,m,std::hypot(.25,.23*std::abs(sky.lever[base]-sky.lever[e])));
            if(best.row<0)continue;
            Vec2 point{sky.xy[2*best.row],sky.xy[2*best.row+1]};
            double separation=std::sqrt(dot(point-slow,point-slow))/as;
            double motion=std::sqrt(dot(point-p,point-p))/as;
            if(separation<1.1||motion<1.35||motion/std::abs(24.*dt)<4.)continue;
            local.push_back({best.row,e,g,best.odds-1.6,vc});
        }
        std::sort(local.begin(),local.end(),[](auto &a,auto &b){return a.score>b.score;});
        for(size_t k=0;k<std::min(size_t(4),local.size());k++)peers.push_back(local[k]);
    }
    for(auto &peer:peers){
        auto &second=sky.images[peer.ex].exposure;
        double dt=second.t-from.t,g=peer.gamma;
        Vec2 b{sky.xy[2*peer.row],sky.xy[2*peer.row+1]};
        Vec2 origin=p*(1.-g*from.oz)+Vec2{from.ox,from.oy}*g;
        double grav=mu*std::pow(g/std::sqrt(1.+dot(origin,origin)),3);
        double f=1.-.5*grav*dt*dt;
        Vec2 rate=(b*(f-g*second.oz)+Vec2{second.ox,second.oy}*g-origin*f)*(1./dt);
        for(int e:dates){
            if(e==base||e==peer.ex)continue;
            auto &to=sky.images[e].exposure;double t=to.t-from.t,frac=t/dt;
            if(std::min(std::abs(t),std::abs(t-dt))<.002||std::abs(t)>1.75||std::abs(frac)>32.)continue;
            f=1.-.5*grav*t*t;
            Vec2 pred=(origin*f+rate*t-Vec2{to.ox,to.oy}*g)*(1./(f-g*to.oz));
            double disp=std::min(dot(pred-p,pred-p),dot(pred-b,pred-b));
            if(disp<sq(1.1*as))continue;
            double sig=std::hypot(.21*std::hypot(1.-frac,frac),.24+.9*std::abs(t*(t-dt)));
            if(sig>3.1)continue;
            auto third=nearby(sky,e,pred,sig,.5*(m+sky.mag[peer.row]),.33);
            if(third.row<0||members.count(third.row)||members.count(peer.row))continue;
            double score=peer.score+third.odds-slow_support;
            if(score>out.fast){
                out.fast=score;out.partner=peer.row;out.confirm=third.row;
            }
        }
    }
}
}
}

extern "C" int foreground_motion(const original::Sky *s,const local_motion::Request *q,int n,
        const double *east,int threads,double seconds,local_motion::Result *out){
    try{
        if(!s||n==0)return 0;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::vector<std::vector<int>> dates(s->images.size());
        for(size_t i=0;i<dates.size();i++)for(size_t j=0;j<dates.size();j++)
            if(i!=j&&std::abs(s->images[i].exposure.t-s->images[j].exposure.t)<1.76)dates[i].push_back(j);
        int ng=0;for(int i=0;i<n;i++)ng=std::max(ng,q[i].group+1);
        std::vector<std::unordered_set<int>> members(ng);
        for(int i=0;i<n;i++)members[q[i].group].insert(q[i].row);
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int j=0;j<threads;j++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=n)break;
                int row=q[i].row;if(row<0||row>=s->size)continue;
                local_motion::inspect(*s,q[i],east,out[i],dates[s->ex[row]],members[q[i].group]);
            }
        });
        for(auto &t:pool)t.join();return std::min(n,next.load());
    }catch(...){return -1;}
}

extern "C" int stationary_flux(const original::Sky *sky,const local_motion::Request *q,const double *mag,
        int n,int threads,double seconds,local_motion::Blend *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int j=0;j<threads;j++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=n)break;
                local_motion::blend(*sky,q[i],mag[i],out[i]);
            }
        });
        for(auto &t:pool)t.join();return std::min(n,next.load());
    }catch(...){return -1;}
}
