// Forced measurement of a moving component underneath a persistent
// foreground source, using only the catalogue peaks in independent visits.
namespace blended_sources {
using original::Sky;
struct Component {int row;float x,y,mag,sn,logbf;};
struct Result {Certified orbit;Component component;double merit;int core,pad;};
struct Ref {double t,x,y,flux,sn,color;int night;};

static double median(std::vector<double> a){
    auto middle=a.begin()+a.size()/2;
    std::nth_element(a.begin(),middle,a.end());double m=*middle;
    if(a.size()%2==0)m=.5*(m+*std::max_element(a.begin(),middle));
    return m;
}

bool resolve(const Sky &sky,int row,Component &c){
    c.row=-1;
    int e=sky.ex[row];const auto &now=sky.images[e].exposure;
    double x=sky.xy[2*row],y=sky.xy[2*row+1],mag=sky.mag[row];
    if(!std::isfinite(mag)||sky.sn[row]<5.2||(sky.flag[row]&1))return false;
    const double radius=std::clamp(1.55*sky.seeing[e],1.15,2.2);
    std::vector<Ref> refs;
    for(size_t j=0;j<sky.images.size();++j){
        const auto &im=sky.images[j];const auto &visit=im.exposure;
        if(!im.nx || std::abs(visit.t-now.t)<3. || x<im.xmin-radius*as||x>im.xmax+radius*as||
           y<im.ymin-radius*as||y>im.ymax+radius*as)continue;
        int best=-1;double separation=radius*radius;
        sky.nearby(im,x,y,radius*as,radius*as,[&](int r){
            double d2=(sq(sky.xy[2*r]-x)+sq(sky.xy[2*r+1]-y))/sq(as);
            double dm=sky.mag[r]-mag;
            if(!(d2<separation&&dm>-.90&&dm<3.25)||sky.flag[r]&1)return;
            separation=d2;best=r;
        });
        if(best<0)continue;
        refs.push_back({(visit.t-now.t)/365.25,(sky.xy[2*best]-x)/as,(sky.xy[2*best+1]-y)/as,
                        std::exp(-.921034*(sky.mag[best]-mag)),sky.sn[best],sky.lever[j]-sky.lever[e],visit.night});
    }
    if(refs.size()<3)return false;
    // Multiple visits in a burst share the reference star. Fit one measured
    // centroid and flux per night, preferring the same passband.
    std::sort(refs.begin(),refs.end(),[](auto &a,auto &b){
        if(a.night!=b.night)return a.night<b.night;
        return std::abs(a.color)-.015*std::min(30.,a.sn)<std::abs(b.color)-.015*std::min(30.,b.sn);
    });
    refs.erase(std::unique(refs.begin(),refs.end(),[](auto &a,auto &b){return a.night==b.night;}),refs.end());
    if(refs.size()<3)return false;
    std::vector<double> xs,ys;
    for(auto &v:refs){xs.push_back(v.x);ys.push_back(v.y);}
    double sx=median(xs),sy=median(ys),vx=0.,vy=0.,posvar=.10;
    for(int it=0;it<3;it++){
        double A=0.,B=0.,D=1./sq(.15),xx=0.,xt=0.,yy=0.,yt=0.;
        for(auto &v:refs){
            double d=std::hypot(v.x-sx-vx*v.t,v.y-sy-vy*v.t);
            double w=1./(.014+sq(1.12/v.sn));
            w*=sq(std::min(1.,.33/std::max(.001,d)));
            A+=w;B+=w*v.t;D+=w*v.t*v.t;xx+=w*v.x;xt+=w*v.x*v.t;yy+=w*v.y;yt+=w*v.y*v.t;
        }
        double det=A*D-B*B;if(!(det>0.))return false;
        sx=(D*xx-B*xt)/det;sy=(D*yy-B*yt)/det;
        vx=(A*xt-B*xx)/det;vy=(A*yt-B*yy)/det;posvar=D/det+.003;
    }
    if(sx*sx+sy*sy>sq(radius))return false;
    std::vector<Ref> stars;
    for(auto &v:refs)if(std::hypot(v.x-sx-vx*v.t,v.y-sy-vy*v.t)<.44)stars.push_back(v);
    if(stars.size()<3)return false;
    double m0=0.,color=0.,vflux=0.;
    std::vector<double> vals;
    for(auto &v:stars)vals.push_back(-1.085736*std::log(v.flux));
    m0=median(vals);
    int same=0;double color_min=9.,color_max=-9.;
    for(auto &v:stars){same+=std::abs(v.color)<.025;color_min=std::min(color_min,v.color);color_max=std::max(color_max,v.color);}
    if(!same && !(color_min<=0.&&color_max>=0.) && stars.size()<6)return false;
    double mad=.15;
    for(int it=0;it<3;it++){
        double a=0.,b=0.,d=1./sq(.35),u=0.,z=0.;
        for(size_t j=0;j<stars.size();++j){
            auto &v=stars[j];double m=vals[j]+1.2/sq(v.sn);
            double diff=m-m0-color*v.color;
            double w=1./(.008+sq(1.09/v.sn)+sq(.10*v.color));
            w*=sq(std::min(1.,std::max(.15,1.7*mad)/std::max(.001,std::abs(diff))));
            a+=w;b+=w*v.color;d+=w*v.color*v.color;u+=w*m;z+=w*m*v.color;
        }
        double det=a*d-b*b;if(!(det>0.))return false;
        m0=(d*u-b*z)/det;color=(a*z-b*u)/det;vflux=d/det;
        std::vector<double> diff;
        for(size_t j=0;j<stars.size();j++)diff.push_back(std::abs(vals[j]-m0-color*stars[j].color));
        mad=1.4826*median(diff);
    }
    if(mad>.36)return false;
    // A single-PSF amplitude is a matched-filter measurement. For Gaussian
    // seeing, each component contributes exp(-separation^2/(4 sigma^2)).
    // The derivative of that amplitude vanishes at the measured centroid.
    double sigma=std::clamp(sky.seeing[e]/2.355,.28,1.12);
    double r2=sx*sx+sy*sy;
    double steady=std::exp(-.921034*m0-r2/(4.*sq(sigma)));
    double frac=1.-steady;if(!(frac>.095&&frac<.84))return false;
    double fvar=sq(1./sky.sn[row])+sq(steady)*(.921*.921*(vflux+sq(std::max(.075,mad)))+
                   posvar*r2/(4.*std::pow(sigma,4)));
    double signal=frac/std::sqrt(fvar);
    if(signal<2.9)return false;
    double ox=-sx*steady/frac,oy=-sy*steady/frac;
    if(sq(ox-sx)+sq(oy-sy)>sq(1.40*sky.seeing[e]))return false;
    double moving=frac*std::exp((ox*ox+oy*oy)/(4.*sq(sigma)));
    double noise=std::sqrt((.003+sq(1.05/sky.sn[row])+steady*steady*posvar)/sq(frac)
                      +(ox*ox+oy*oy)*fvar/sq(frac)+sq(.10));
    if(noise>.72)return false;
    double stat=std::min(24.,sq(signal))+std::min(9.,r2/(posvar+sq(.09)+sq(1.05/sky.sn[row])));
    c={row,float(x+ox*as),float(y+oy*as),float(mag-1.085736*std::log(moving)),
       float(std::clamp(1./std::max(.05,noise),1.45,std::min(35.,signal))),float(.5*stat)};
    return true;
}

struct Worker {
    const Sky &sky;const double *depth;
    std::unordered_map<int,Component> cache;
    std::vector<Result> found;
    std::array<long long,5> stats{};
    const Component &component(int row){
        if(cache.size()>90000)cache.clear();
        auto it=cache.try_emplace(row);
        if(it.second){++stats[2];if(resolve(sky,row,it.first->second))++stats[3];}
        return it.first->second;
    }
    void core(const original::Core &seed,int idx){
        if(seed.count<4||seed.count>12||!(seed.x[2]>1./350.&&seed.x[2]<1./16.))return;
        ExactState o{seed.x[0],seed.x[1],seed.x[2],seed.x[3],seed.x[4],seed.x[5],seed.ref};
        std::vector<Sample> obs;std::vector<int> fixed;
        for(int j=0;j<seed.count;j++){
            int r=seed.rows[j];
            if(r<0||r>=sky.size||!std::isfinite(sky.mag[r]))return;
            obs.push_back(sky.sample(r));fixed.push_back(sky.images[sky.ex[r]].exposure.night);
        }
        std::sort(fixed.begin(),fixed.end());fixed.erase(std::unique(fixed.begin(),fixed.end()),fixed.end());
        if(fixed.size()!=4)return;
        std::vector<double> residual;
        if(!measured_fit(obs,o,residual,7)||*std::max_element(residual.begin(),residual.end())>.57)return;
        auto phot=confirmation_trials::photometry(sky,obs);
        Kepler orbit(o);double N[6][6]{},V[6][6];
        N[0][0]=N[1][1]=1.e-12;N[2][2]=3.;N[3][3]=N[4][4]=4.;N[5][5]=8.;
        for(auto &p:obs){
            double x,y,f,s,J[2][6]{};orbit.apparent(p,x,y,f,s);original::design(o,p,x,y,f,s,J);
            double w=1./(.004+.04/p.weight);
            for(int j=0;j<6;j++)for(int k=0;k<6;k++)N[j][k]+=w*(J[0][j]*J[0][k]+J[1][j]*J[1][k]);
        }
        if(!original::invert(N,V))return;
        struct Hit {Component c; double merit;};
        std::vector<Hit> hits;
        for(int e=0;e<int(sky.images.size());++e){
            auto &im=sky.images[e];
            if(std::binary_search(fixed.begin(),fixed.end(),im.exposure.night))continue;
            confirmation_trials::View v;
            if(!confirmation_trials::make_view(sky,e,depth,o,orbit,V,phot,v)||v.detect<.13||
               v.xx+v.yy>sq(23.)||std::sqrt(v.det)>47.)continue;
            double mx=2.85*std::sqrt(v.xx)+1.4*sky.seeing[e],my=2.85*std::sqrt(v.yy)+1.4*sky.seeing[e];
            sky.nearby(im,v.x,v.y,mx*as,my*as,[&](int row){
                double dm=sky.mag[row]-v.mag;
                if(!(dm<-.20&&dm> -2.85)||sky.sn[row]<6.0||(sky.flag[row]&1))return;
                ++stats[1];
                auto c=component(row);
                if(c.row<0)return;
                double variance=sq(1./c.sn);
                double xx=v.xx+variance,yy=v.yy+variance,det=xx*yy-v.xy*v.xy;
                double x=(c.x-v.x)/as,y=(c.y-v.y)/as;
                double chi=(yy*x*x-2.*v.xy*x*y+xx*y*y)/det;
                double sm=std::hypot(v.sm,1.09/c.sn),mag=c.mag-v.mag;
                if(chi>10.8||std::abs(mag)>2.65*sm)return;
                double rho=im.rho[std::clamp(int((sky.mag[row]-10.)*4.),1,78)];
                double q=std::min(4.,.35*(c.logbf-6.))-.5*(chi+sq(mag/sm))+
                         std::log(v.detect/std::max(1.e-8,15.75*std::sqrt(det)*sm*rho));
                if(q>3.)hits.push_back({c,q});
            });
        }
        std::sort(hits.begin(),hits.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        for(size_t i=0;i<std::min<size_t>(4,hits.size());i++){
            auto &hit=hits[i];auto c=hit.c;auto p=sky.sample(c.row);
            p.x=c.x;p.y=c.y;p.weight=sq(.20*c.sn);
            auto points=obs;points.push_back(p);ExactState state=o;
            if(!measured_fit(points,state,residual,7))continue;
            double chi=0.,rss=0.;bool pass=true;
            for(size_t j=0;j<points.size();j++){
                double var=.04/points[j].weight+.004;
                chi+=sq(residual[j])/var;rss+=sq(residual[j]);
                if(residual[j]>std::max(.56,3.3*std::sqrt(var)))pass=false;
            }
            if(!pass||chi>3.1*(2.*points.size()-6.))continue;
            Result r{};r.component=c;r.core=idx;r.merit=hit.merit-.28*chi;
            auto &co=r.orbit;co.count=points.size();co.nights=5;co.ref=state.ref;
            co.merit=r.merit+.35*seed.merit;
            double xs[]={state.x,state.y,state.g,state.vx,state.vy,state.h};std::copy_n(xs,6,co.x);
            for(int j=0;j<co.count;j++){co.row[j]=points[j].row;co.err[j]=residual[j];}
            found.push_back(r);++stats[4];
            break;
        }
        ++stats[0];
    }
};
}

extern "C" int blended_orbits(const original::Sky *sky,const original::Core *cores,int n,const double *depth,
        int threads,double seconds,int cap,blended_sources::Result *out,long long *counts){
    try{
        using namespace blended_sources;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::vector<Worker> jobs;std::vector<std::thread> workers;std::atomic<int> next{0};
        for(int j=0;j<threads;j++)jobs.push_back({*sky,depth});
        for(auto &j:jobs)workers.emplace_back([&,job=&j]{
            while(!expired(end)){
                int k=next.fetch_add(1);if(k>=n)break;job->core(cores[k],k);
            }
        });
        for(auto &t:workers)t.join();
        std::vector<blended_sources::Result> result;
        for(auto &j:jobs){
            result.insert(result.end(),j.found.begin(),j.found.end());
            for(int k=0;k<5;k++)counts[k]+=j.stats[k];
        }
        int done=std::min<int>(cap,result.size());
        std::partial_sort(result.begin(),result.begin()+done,result.end(),[](auto &a,auto &b){return a.orbit.merit>b.orbit.merit;});
        std::copy_n(result.begin(),done,out);return done;
    }catch(...){return -1;}
}
