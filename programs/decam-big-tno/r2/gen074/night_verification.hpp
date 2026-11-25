// Verify the five actual shutter measurements, including raw-catalogue
// returns that have no entry in the filtered orbit index.
namespace night_verify {
struct Trial {int rows[5];float gamma,score;};
bool fit(const original::Sky &sky,const Trial &trial,original::Recovery &out){
    std::vector<Sample> obs;
    std::vector<double> mags,err;
    std::set<int> nights;
    ExactState o{};
    o.g=trial.gamma;
    for(int i:trial.rows){
        if(i<0||i>=sky.size)return false;
        auto p=sky.sample(i);
        int n=sky.images[sky.ex[i]].exposure.night;
        if(!nights.insert(n).second)return false;
        obs.push_back(p);mags.push_back(sky.mag[i]);
        o.x+=.2*p.x;o.y+=.2*p.y;o.ref+=.2*p.t;
    }
    if(!measured_fit(obs,o,err,9)||energy(o)>2.00)return false;
    auto rms=[](auto &res){double v=0.;for(double e:res)v+=e*e;return std::sqrt(v/res.size());};
    if(rms(err)>.33||*std::max_element(err.begin(),err.end())>.68)return false;
    auto &r=out.orbit;
    std::sort(mags.begin(),mags.end());double mr=mags[2];
    // A precise measured fifth position must also be a plausible source
    // of the object's flux on this visit. A faint underlying star is common.
    double source=0.;
    for(auto &p:obs){
        int i=p.row;
        const auto &im=sky.images[sky.ex[i]];
        int mb=std::clamp(int((mr-10.)*4.),0,79);
        double dm=sky.mag[i]-mr;
        double w=sq(.27)+sq(1.086/std::max(4.5,double(sky.sn[i])));
        source+=std::clamp(-std::log(std::max(1.e-7,im.rho[mb])/.001)
                           -.43*std::min(15.,dm*dm/w),-6.,4.);
    }
    auto old=obs;auto olderr=err;auto seed=o;
    Kepler kep(o);
    for(int e=0;e<int(sky.images.size()) && obs.size()<56;e++){
        const auto &im=sky.images[e];
        if(!im.nx||std::any_of(old.begin(),old.end(),[&](auto &p){return sky.ex[p.row]==e;}))continue;
        double x,y,f,u;
        auto &v=im.exposure;
        Sample p{-1,0,0,v.t,v.ox,v.oy,v.oz};
        kep.apparent(p,x,y,f,u);
        if(x<im.xmin-as||x>im.xmax+as||y<im.ymin-as||y>im.ymax+as)continue;
        int best=-1;double score=7.;
        sky.nearby(im,x,y,.86*as,.86*as,[&](int i){
            if(sky.flag[i]&1)return;
            double cost=(sq(sky.xy[2*i]-x)+sq(sky.xy[2*i+1]-y))/sq(.28*as)
                         +sq((sky.mag[i]-mr)/.37);
            if(sky.population && !sky.population[i])cost+=1.6;
            if(cost<score){score=cost;best=i;}
        });
        if(best>=0)obs.push_back(sky.sample(best));
    }
    if(obs.size()>5 && (!measured_fit(obs,o,err,6)||rms(err)>.33||
                         *std::max_element(err.begin(),err.end())>.69||energy(o)>2.00)){
        o=seed;obs.swap(old);err.swap(olderr);
    }
    nights.clear();
    for(size_t j=0;j<obs.size();j++){
        r.row[j]=obs[j].row;r.err[j]=err[j];
        nights.insert(sky.images[sky.ex[obs[j].row]].exposure.night);
    }
    r.nights=nights.size();r.count=obs.size();r.ref=o.ref;
    double par[]={o.x,o.y,o.g,o.vx,o.vy,o.h};
    std::copy_n(par,6,r.x);
    r.merit=2.4*std::min(15,r.nights)-3.*sq(rms(err)/.24)+.68*source+
            .45*std::min(5,r.count-r.nights);
    out.seed=4;out.extra=r.count-4;
    out.logbf=std::clamp(5.+.15*source-1.5*sq(rms(err)/.25),.0,9.);
    return true;
}
}
extern "C" int night_candidate_fit(const original::Sky *sky,const night_verify::Trial *trial,
                                   int n,double seconds,int cap,original::Recovery *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        int count=0;
        for(int i=0;i<n && count<cap&&!expired(end);i++){
            original::Recovery report{};
            if(night_verify::fit(*sky,trial[i],report))out[count++]=report;
        }
        return count;
    }catch(...){return -1;}
}
