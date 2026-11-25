// Conditional exposure-pair likelihood. Control anchors are catalogue
// sources in the first image with the same flux and foreground class.
namespace repeat_controls {
using original::Sky;
using original::Frame;
using encounter::V;
struct Candidate {double radius;int n,rows[32];int padding;};
struct Evidence {float odds,positive,missing;int tested,found,controls;};
V pos(const Sky &s,int row){return {s.xy[2*row],s.xy[2*row+1]};}
double brightness(double m,double depth){
    double sn=5.*std::exp(std::clamp(.921034*(depth-m),-8.,6.));
    return .965*.5*std::erfc((4.7-sn)/std::hypot(1.6,.25*sn));
}
double margin(const Sky &s,int row){
    return std::min({double(s.xp[row]),2048.-s.xp[row],double(s.yp[row]),4096.-s.yp[row]});
}
struct Observation {double chi=1000.,error=0.;bool covered=false;};
Observation observe(const Sky &s,int e,V p,double mag,double variance,double sm){
    Observation out;const auto &im=s.images[e];
    double close=1.e6;
    s.nearby(im,p.x,p.y,20.*as,20.*as,[&](int j){
        V d=(pos(s,j)-p)*(1./as);double r2=dot(d,d);
        if(r2>400.)return;
        if(margin(s,j)>8.+3.96*std::sqrt(r2)&&!(s.flag[j]&1)){
            out.covered=true;
            if(r2<close&&s.ferr[j]>0.&&s.sn[j]<36.){
                close=r2;
                out.error=s.mag[j]+2.5*std::log10(std::max(1.,double(s.sn[j]))/5.);
            }
        }
        if(r2>3.24||(s.flag[j]&1)||s.sharp[j]<=.03||s.sharp[j]>2.55)return;
        double var=variance+sq(std::min(2.4,s.seeing[e])/std::max(4.5f,s.sn[j]));
        double chi=r2/var+sq(s.mag[j]-mag)/(sm*sm+sq(1.09/std::max(4.5f,s.sn[j])));
        if(chi<out.chi){out.chi=chi;if(r2<.8)out.covered=true;}
    });
    return out;
}
void measure(const Sky &s,const double *depth,const std::vector<std::vector<int>> &peers,
             const Candidate &c,Evidence &out){
    std::vector<Sample> sample;
    double mean=0.,tm=0.,mx=0.,my=0.;
    for(int i=0;i<c.n;i++){
        int row=c.rows[i];if(row<0||row>=s.size||!std::isfinite(s.mag[row]))continue;
        auto p=s.sample(row);sample.push_back(p);mean+=s.mag[row];tm+=p.t;mx+=p.x;my+=p.y;
    }
    if(sample.size()<5)return;
    int n=sample.size();tm/=n;mx/=n;my/=n;mean/=n;
    double g=std::sqrt(1.+mx*mx+my*my)/std::max(12.,c.radius);
    ExactState state{mx,my,g,0.,0.,0.,tm};std::vector<double> res;
    bool fitted=c.radius<4000.&&measured_fit(sample,state,res,5);
    V rate{};double tt=0.;
    for(auto &p:sample){
        double dt=p.t-tm;V bary{p.x+g*(p.ox-p.x*p.oz),p.y+g*(p.oy-p.y*p.oz)};
        rate=rate+bary*dt;tt+=dt*dt;
    }
    rate=rate*(1./std::max(.001,tt));
    std::vector<double> mags;
    for(auto &p:sample)mags.push_back(s.mag[p.row]);
    std::sort(mags.begin(),mags.end());mean=mags[mags.size()/2];
    // Estimate a colour from independent dates. Within one night a common
    // rotational phase makes flux differences more precise than the mean.
    double sl=0.,mm=0.,ll=0.;for(auto &p:sample)sl+=s.lever[s.ex[p.row]];
    sl/=n;
    for(auto &p:sample){
        double l=s.lever[s.ex[p.row]]-sl;
        mm+=l*std::clamp(s.mag[p.row]-mean,-.6,.6);ll+=l*l;
    }
    double color=mm/(ll+.65);
    std::vector<std::pair<double,int>> anchors;
    for(auto &p:sample){
        int j=p.row,e=s.ex[j];if(peers[e].empty())continue;
        double expected=mean+(s.lever[e]-sl)*color;
        double quality=std::log(std::min(24.,double(s.sn[j])))-std::min(4.,sq((s.mag[j]-expected)/.36));
        anchors.push_back({quality,j});
    }
    std::sort(anchors.begin(),anchors.end(),std::greater<std::pair<double,int>>());
    std::vector<int> tested_nights;
    for(auto a:anchors){
        int row=a.second,source=s.ex[row];
        const auto &one=s.images[source].exposure;
        if(std::find(tested_nights.begin(),tested_nights.end(),one.night)!=tested_nights.end())continue;
        V at=pos(s,row);
        int best=-1;double quality=-1.e8;V delta{};
        for(int other:peers[source]){
            const auto &two=s.images[other];double dt=two.exposure.t-one.t;
            double dz=two.exposure.oz-one.oz;
            V move=(rate*dt-V{two.exposure.ox-one.ox-at.x*dz,two.exposure.oy-one.oy-at.y*dz}*g)*
                   (1./(1.-g*two.exposure.oz));
            if(fitted){
                Kepler orbit(state);double ax,ay,bx,by,f,v;
                orbit.apparent(s.sample(row),ax,ay,f,v);
                const auto &e=two.exposure;
                Sample moment{-1,0,0,e.t,e.ox,e.oy,e.oz};
                orbit.apparent(moment,bx,by,f,v);move={bx-ax,by-ay};
            }
            V p=at+move;
            if(!two.nx||p.x<two.xmin||p.x>two.xmax||p.y<two.ymin||p.y>two.ymax)continue;
            double l=s.lever[other]-s.lever[source],pred=s.mag[row]+l*color;
            double visible=brightness(pred,depth[other]);
            double criterion=std::log(.015+visible)+.28*std::log1p(std::min(36.,dot(move,move)/sq(.40*as)))
                            -.19*std::abs(l);
            if(criterion>quality){quality=criterion;best=other;delta=move;}
        }
        if(best<0)continue;
        double l=s.lever[best]-s.lever[source],dt=std::abs(one.t-s.images[best].exposure.t);
        double sm=std::hypot(std::hypot(.09,1.09/s.sn[row]),.17*l);
        double var=sq(.092)+sq(std::min(2.4,s.seeing[source])/s.sn[row])+
                   sq(.055*std::min(2.,dt/.1));
        double mag=s.mag[row]+color*l;
        auto hit=observe(s,best,at+delta,mag,var,sm);
        if(!hit.covered)continue;
        double depth_b=depth[best];
        if(hit.error>0.&&std::isfinite(hit.error))depth_b=.72*hit.error+.28*depth_b;
        double vis=brightness(mag+.035,depth_b);
        if(vis<.12)continue;
        // Compare to the distribution of a bounded matched-filter score on
        // actual nearby first-exposure sources. Control stars and transients
        // have different chances of repeating at an unresolved displacement.
        struct Control {double distance;int row;};
        std::vector<Control> controls;
        auto &im=s.images[source];
        s.nearby(im,at.x,at.y,260.*as,260.*as,[&](int j){
            if(j==row||(s.flag[j]&1)||!std::isfinite(s.mag[j])||s.sharp[j]<=.03||s.sharp[j]>2.55)return;
            double dm=s.mag[j]-s.mag[row];if(std::abs(dm)>1.15)return;
            V change=(pos(s,j)-at)*(1./as);double r2=dot(change,change);
            if(r2<25.||r2>260.*260.)return;
            double cost=sq(dm/.37)+r2/sq(220.);
            if(s.population)cost+=3.8*std::abs(int(s.population[j])-int(s.population[row]));
            cost+=.30*sq(std::log(std::max(4.5f,s.sn[j])/std::max(4.5f,s.sn[row]))/.40);
            controls.push_back({cost,j});
        });
        if(controls.size()<6)continue;
        if(controls.size()>80){
            std::nth_element(controls.begin(),controls.begin()+80,controls.end(),[](auto &a,auto &b){return a.distance<b.distance;});
            controls.resize(80);
        }
        double level=std::clamp(hit.chi,1.15,10.);
        double total=0.,wins=0.;
        for(auto v:controls){
            double weight=std::exp(-.5*v.distance);
            auto z=observe(s,best,pos(s,v.row)+delta,s.mag[v.row]+l*color,var,sm);
            if(z.covered){total+=weight;if(z.chi<=level)wins+=weight;}
        }
        if(total<3.)continue;
        // Use the same event for both hypotheses, including its probability
        // for a true measurement in two angular coordinates and one flux.
        double u=std::sqrt(level/2.);
        double good=vis*(std::erf(u)-1.128379167*u*std::exp(-u*u));
        double chance=(.45+wins)/(total+2.);
        bool found=hit.chi<=10.;
        double signal=chance+(1.-chance)*good;
        double odds=found?std::log(signal/chance):std::log((1.-signal)/(1.-chance));
        odds=std::clamp(odds,-2.5,3.8);
        out.positive+=std::max(0.,odds);out.missing+=std::min(0.,odds);
        out.odds+=odds;out.tested++;out.found+=found;out.controls+=int(total);
        tested_nights.push_back(one.night);
        if(out.tested>=9)break;
    }
}
}

extern "C" int repeat_control_odds(const original::Sky *sky,const double *depth,
        const repeat_controls::Candidate *inputs,int n,int threads,double seconds,repeat_controls::Evidence *out){
    try{
        std::vector<std::vector<int>> peers(sky->images.size());
        for(size_t e=0;e<peers.size();e++)for(size_t b=0;b<peers.size();b++)if(e!=b&&
            sky->images[e].exposure.night==sky->images[b].exposure.night&&sky->images[b].nx&&
            std::abs(sky->images[e].exposure.t-sky->images[b].exposure.t)>.00018)peers[e].push_back(b);
        std::atomic<int> next{0};std::vector<std::thread> pool;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        for(int t=0;t<threads;t++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int k=next.fetch_add(1);if(k>=n)break;
                if(inputs[k].n>=5&&inputs[k].n<=32)repeat_controls::measure(*sky,depth,peers,inputs[k],out[k]);
            }
        });
        for(auto &t:pool)t.join();
        return std::min(n,next.load());
    }catch(...){return -1;}
}
