// Fit a local stellar proper motion using only independent reference nights.
// Positional/flux agreement in the member visit can then favour that star
// over a peak from a moving solar-system body.
namespace stellar_tracks {
struct Point {double years,x,y,dm,position,flux;int night;};
struct Model {double vx,vy,x,y,merit;};
struct Mark {float score,flux,motion;int count;};

static Mark measure(const original::Sky &sky,const reference_flux::Query &q,const float *range,
                    const uint8_t *separate){
    Mark out{};
    const auto &home=sky.images[q.e].exposure;
    std::vector<Point> points;
    std::vector<Model> proposals;
    for(size_t i=0;i<sky.images.size();i++){
        const auto &im=sky.images[i];const auto &e=im.exposure;
        double dt=e.t-home.t;
        if(std::abs(dt)<3.||!im.nx)continue;
        double t=dt/365.25,width=.90+std::min(8.,.85*std::abs(t));
        if(q.x<im.xmin-width*as||q.x>im.xmax+width*as||
           q.y<im.ymin-width*as||q.y>im.ymax+width*as)continue;
        // Exclude every reference in which the solar-system object could
        // still occupy the aperture, including its return near opposition.
        double dz=e.oz-home.oz;
        double par=std::hypot(e.ox-home.ox-q.x*dz,e.oy-home.oy-q.y*dz)/
                   (1.+q.x*q.x+q.y*q.y);
        double lo=std::max(10.,.78*range[0]),hi=1.30*range[1]+1.;
        if(!(separate&&separate[i])&&(par-std::abs(dt)*std::sqrt(2.*mu/lo))/hi<(width+1.3)*as){
            // A slow body could return to the same place a year later.
            // For the belt use the same guard. References close together
            // within another season can still resolve a common star.
            continue;
        }
        double lever=sky.lever[i]-sky.lever[q.e];
        sky.nearby(im,q.x,q.y,width*as,width*as,[&](int row){
            if(sky.flag[row]&1)return;
            double m=sky.mag[row]-q.mag;
            if(!std::isfinite(m)||m< -2.2||m>2.3)return;
            double x=(sky.xy[2*row]-q.x)/as,y=(sky.xy[2*row+1]-q.y)/as;
            if(x*x+y*y>width*width)return;
            double v=sq(.095)+sq(.95/std::max(4.5f,sky.sn[row]));
            double f=sq(.14)+sq(1.09/std::max(4.5f,sky.sn[row]))+sq(.29*lever);
            points.push_back({t,x,y,m,v,f,e.night});
        });
    }
    if(points.empty())return out;
    std::sort(points.begin(),points.end(),[](auto &a,auto &b){return a.night<b.night;});
    // Fit reference positions without counting a burst of visits as
    // separate epochs. Start from the candidate star at the origin and
    // from angular rates measured between the query and remote centroids.
    proposals.push_back({0.,0.,0.,0.,0.});
    double origin_variance=sq(.085)+sq(1.02/q.sn);
    for(auto &p:points){
        if(std::abs(p.years)<.18)continue;
        double shrink=sq(p.years)/(sq(p.years)+(p.position+origin_variance)/sq(.27));
        double vx=p.x/p.years*shrink,vy=p.y/p.years*shrink;
        if(vx*vx+vy*vy>.82*.82)continue;
        double cost=(vx*vx+vy*vy)/sq(.33)+.34*sq(p.dm)/(.15+p.flux);
        proposals.push_back({vx,vy,0.,0.,-cost});
    }
    std::sort(proposals.begin(),proposals.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    std::vector<Model> tested;std::vector<const Point*> visits;
    double best=0.;
    for(auto model:proposals){
        bool duplicate=false;
        for(auto &m:tested)if(sq(m.vx-model.vx)+sq(m.vy-model.vy)<sq(.033)){duplicate=true;break;}
        if(duplicate)continue;
        tested.push_back(model);if(tested.size()>22)break;
        for(int iter=0;iter<3;iter++){
            visits.clear();
            double rr=0.,rt=0.,tt=1./sq(.35),rx=0.,ry=0.,tx=0.,ty=0.;
            int night=-100000;
            double top=0.;
            for(auto &p:points){
                if(p.night!=night){night=p.night;top=0.;}
                double v=p.position+(iter==0?origin_variance+.04:.025);
                double d2=sq(p.x-model.x-model.vx*p.years)+sq(p.y-model.y-model.vy*p.years);
                double chi=d2/v;
                if(chi>12.)continue;
                double worth=std::exp(-.5*chi-.22*sq(p.dm)/(.10+p.flux))/std::sqrt(v*p.flux);
                if(worth<=top)continue;
                if(top>0.)visits.back()=&p;else visits.push_back(&p);
                top=worth;
            }
            if(visits.empty())break;
            std::vector<double> phot;phot.reserve(visits.size());
            for(auto pp:visits){
                auto &p=*pp;
                double w=1./(p.position+.035);
                rr+=w;rt+=w*p.years;tt+=w*sq(p.years);
                rx+=w*p.x;ry+=w*p.y;tx+=w*p.years*p.x;ty+=w*p.years*p.y;
                phot.push_back(p.dm);
            }
            double det=rr*tt-rt*rt;if(!(det>0.))break;
            model.vx=(rr*tx-rt*rx)/det;model.vy=(rr*ty-rt*ry)/det;
            model.x=(tt*rx-rt*tx)/det;model.y=(tt*ry-rt*ty)/det;
            if(iter<2)continue;
            std::sort(phot.begin(),phot.end());
            double dm=.5*(phot[(phot.size()-1)/2]+phot[phot.size()/2]);
            double magvar=.04+sq(1.09/q.sn),centroid=origin_variance+tt/det;
            double num=0.;
            for(auto pp:visits){
                auto &p=*pp;
                double residual=sq(p.x-model.x-model.vx*p.years)+sq(p.y-model.y-model.vy*p.years);
                double photchi=sq(p.dm-dm)/(p.flux+.025);
                double weight=std::exp(-.35*residual/(p.position+.035)-.38*photchi);
                num+=weight;magvar+=p.flux*weight/sq(std::max(1.,double(visits.size())));
            }
            double poschi=(sq(model.x)+sq(model.y))/centroid;
            // With uncertain reference astrometry the predicted stellar
            // centroid has a lower density even when the residual is zero.
            double strength=std::log1p(num)*std::exp(-.5*poschi-.5*sq(dm)/magvar)/
                            std::sqrt(centroid*centroid*magvar);
            if(strength>best){
                best=strength;out.score=std::log1p(.065*strength);
                out.flux=dm;out.motion=std::hypot(model.vx,model.vy);out.count=visits.size();
            }
        }
    }
    return out;
}
}

extern "C" int conditioned_stellar_tracks(const original::Sky *sky,const reference_flux::Query *q,
            const float *ranges,int n,const uint8_t *mask,int apertures,int threads,double seconds,
            stellar_tracks::Mark *out){
    try{
        std::atomic<int> next{0};std::vector<std::thread> workers;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        for(int i=0;i<threads;i++)workers.emplace_back([&]{
            while(Clock::now()<end){
                int at=next.fetch_add(24);if(at>=n)break;
                for(int j=at;j<std::min(n,at+24);j++)
                    out[j]=stellar_tracks::measure(*sky,q[j],ranges+j*2,
                                              mask?mask+(j/apertures)*sky->images.size():nullptr);
            }
        });
        for(auto &t:workers)t.join();return std::min(n,next.load());
    }catch(...){return -1;}
}

extern "C" int reference_stellar_tracks(const original::Sky *sky,const reference_flux::Query *q,
            const float *ranges,int n,int threads,double seconds,stellar_tracks::Mark *out){
    return conditioned_stellar_tracks(sky,q,ranges,n,nullptr,1,threads,seconds,out);
}
