// Search for remote observations of a two-night arc. A month of directions
// shares one index after removing a reference parallax and angular velocity.
namespace {
struct RemotePoint {float x,y;int id;};
struct RemoteBin {
    Endpoint ref{};
    double xmin=10.,xmax=-10.,ymin=10.,ymax=-10.;
    double tmin=1.e10,tmax=-1.e10,spread=0.;
    int count=0;
    std::vector<std::pair<uint64_t,RemotePoint>> ordered;
    int x0=0,y0=0,nx=0,ny=0;
    std::vector<int> start;
    static constexpr double width=118.*as;
    void finish() {
        if(!count)return;
        x0=int(std::floor(xmin/width));y0=int(std::floor(ymin/width));
        nx=int(std::floor(xmax/width))-x0+1;ny=int(std::floor(ymax/width))-y0+1;
        if(size_t(nx)*ny>900000){
            std::sort(ordered.begin(),ordered.end(),[](auto &a,auto &b){return a.first<b.first;});
            return;
        }
        start.assign(size_t(nx)*ny+1,0);
        for(auto &v:ordered){
            int x=int32_t(v.first>>32)-x0,y=int32_t(v.first)-y0;
            start[size_t(x)*ny+y+1]++;
        }
        for(size_t i=1;i<start.size();i++)start[i]+=start[i-1];
        // Direct offsets index cells in signed tangent-plane order.
        std::sort(ordered.begin(),ordered.end(),[&](auto &a,auto &b){
            int ax=int32_t(a.first>>32),bx=int32_t(b.first>>32);
            return ax!=bx?ax<bx:int32_t(a.first)<int32_t(b.first);
        });
    }
};
struct RemoteCatalogue {
    const Source *s;int n;
    Vec2 east;const Witness &w;
    double first,g=1./43.,vc=std::sqrt(mu/(43.*43.*43.));
    std::vector<RemoteBin> bins;
    RemoteCatalogue(const Source *s,int n,Vec2 east,const Witness &w):s(s),n(n),east(east),w(w) {
        double last=s[0].t;first=last;
        for(int i=0;i<n;i++){first=std::min(first,s[i].t);last=std::max(last,s[i].t);}
        bins.resize(int((last-first)/37.)+1);
        for(int i=0;i<n;i++){
            auto &a=s[i];auto &b=bins[int((a.t-first)/37.)];b.count++;
            auto &r=b.ref;r.t+=a.t;r.ox+=a.ox;r.oy+=a.oy;r.oz+=a.oz;
            b.tmin=std::min(b.tmin,a.t);b.tmax=std::max(b.tmax,a.t);
        }
        for(auto &b:bins){
            if(!b.count)continue;
            auto &r=b.ref;double f=1./b.count;r.t*=f;r.ox*=f;r.oy*=f;r.oz*=f;
            b.spread=std::max(r.t-b.tmin,b.tmax-r.t);
            b.ordered.reserve(b.count);
        }
        for(int i=0;i<n;i++){
            const auto &a=s[i];auto &b=bins[int((a.t-first)/37.)];const auto &r=b.ref;
            double x=a.x+g*(a.ox-r.ox-a.x*(a.oz-r.oz))-(a.t-r.t)*east.x*vc;
            double y=a.y+g*(a.oy-r.oy-a.y*(a.oz-r.oz))-(a.t-r.t)*east.y*vc;
            RemotePoint p{float(x),float(y),i};
            b.xmin=std::min(b.xmin,double(p.x));b.xmax=std::max(b.xmax,double(p.x));
            b.ymin=std::min(b.ymin,double(p.y));b.ymax=std::max(b.ymax,double(p.y));
            b.ordered.push_back({key(std::floor(p.x/b.width),std::floor(p.y/b.width)),p});
        }
        for(auto &b:bins)b.finish();
    }
};
struct RangeCut {int id;double g,h,sg,sh,slope,merit;};
struct RemoteGuess {int p,q;double dg,h,chi,merit;};
struct RemoteSearch {
    const RemoteCatalogue &cat;const int *owner;int cap;
    std::vector<Certified> five,four;
    std::array<long long,7> count{};
    std::vector<RangeCut> cuts;
    std::unordered_set<uint64_t> seen;
    Clock::time_point end;
    bool screen=false;
    std::vector<RemoteGuess> guesses;

    void check(const Unit &pair,const RemoteGuess &z,double inverse){
        const auto &p=cuts[z.p],&q=cuts[z.q];
        Unit second{p.id,q.id,0.,float(inverse)};Orbit sol;
        if(!quartet(cat.s,pair,second,inverse,sol,false,{16.,135.,6000.,false}))return;
        uint64_t key=EpochSweep::signature(pair.a,pair.b,p.id,q.id);
        if(!seen.insert(key).second)return;
        if(seen.size()>1500000)seen.clear();
        count[3]++;
        int rows[]={owner[pair.a],owner[pair.b],owner[p.id],owner[q.id]};
        Certified orbit{};
        bool save_four=sol.chi<4.&&z.chi<5.&&p.merit+q.merit>-9.;
        if(!certify_one(cat.w,rows,4,sol.g,orbit,false,save_four))return;
        orbit.merit+=.15*(pair.score+p.merit+q.merit);
        if(orbit.nights>=5){
            five.push_back(orbit);count[4]++;
            if(five.size()>size_t(2*cap))compact_certified(five,cap);
        }else{
            four.push_back(orbit);count[5]++;
            if(four.size()>size_t(cap))compact_certified(four,cap/2);
        }
    }

    void search(const Unit &pair,double fraction,double hrange) {
        const auto *s=cat.s;
        const auto &a=s[pair.a],&b=s[pair.b];double dt=b.t-a.t;
        if(dt<.65||dt>125.)return;
        double g=pair.gamma,vc=std::sqrt(mu*g*g*g);
        Vec2 d={(b.ox-b.x*b.oz-a.ox+a.x*a.oz)/dt,(b.oy-b.y*b.oz-a.oy+a.y*a.oz)/dt};
        Vec2 der=d-cat.east*(1.5*vc/g);
        double dg=std::min(.004,fraction*vc/(std::sqrt(dot(der,der))+1.e-5));
        double hscale=hrange*vc;
        Propagator base(a,b,g,0.),range(a,b,g+dg,0.),radial(a,b,g,hscale);
        double mag=.5*(a.mag+b.mag),lever=.5*(a.color+b.color);
        cuts.clear();
        // Bound the residual displacement in the month after removing
        // the reference motion. The individual measurement is tested at
        // its own exposure time below.
        double drift=std::hypot(base.rate.x-cat.east.x*cat.vc,base.rate.y-cat.east.y*cat.vc)
                     +.020*(std::abs(g-cat.g)+dg)+.000018;
        for(auto &bin:cat.bins){
            if(expired(end))return;
            if(!bin.count)continue;
            const auto &e=bin.ref;double t=e.t-a.t;
            if(std::abs(t) > 4900. || (std::abs(t-.5*dt)+bin.spread < .5*dt+15.))continue;
            Vec2 mid=base(e),gx=range(e)-mid,hr=radial(e)-mid;
            double magg=std::sqrt(dot(gx,gx))+1.e-12,magh=std::sqrt(dot(hr,hr));
            Vec2 axis=gx*(1./magg);
            double phase=vc*t;
            double ast=(.18+.28*std::abs(t/dt))*as;
            double model=std::abs(phase*phase*phase*phase)*.11+as*.30+.045*magh;
            double pad=drift*bin.spread+3.*ast+model;
            double perp=std::abs(axis.x*hr.y-axis.y*hr.x)+pad+.040*magg;
            double wx=std::abs(gx.x)+std::abs(hr.x)+pad+.04*magg;
            double wy=std::abs(gx.y)+std::abs(hr.y)+pad+.04*magg;
            if(mid.x+wx<bin.xmin||mid.x-wx>bin.xmax||mid.y+wy<bin.ymin||mid.y-wy>bin.ymax)continue;
            double w=bin.width;
            int i0=std::floor(std::max(mid.x-wx,bin.xmin)/w),i1=std::floor(std::min(mid.x+wx,bin.xmax)/w);
            int j0=std::floor(std::max(mid.y-wy,bin.ymin)/w),j1=std::floor(std::min(mid.y+wy,bin.ymax)/w);
            for(int i=i0;i<=i1;i++){
                int lo=j0,hi=j1;
                if(std::abs(axis.x)>.12){
                    double cy=mid.y+((i+.5)*w-mid.x)*axis.y/axis.x;
                    double dy=(perp+.5*w*std::abs(axis.y))/std::abs(axis.x);
                    lo=std::max(lo,int(std::floor((cy-dy)/w)));hi=std::min(hi,int(std::floor((cy+dy)/w)));
                }
                if(lo>hi)continue;
                auto inspect=[&](const RemotePoint &p){
                    double x=p.x-mid.x,y=p.y-mid.y;
                    if(std::abs(axis.x*y-axis.y*x)>perp)return;
                    const auto &c=s[p.id];
                    if(c.night==a.night||c.night==b.night)return;
                    double dm=std::max(0.,std::abs(c.mag-mag)-.23*std::abs(c.color-lever));
                    if(dm>.68)return;
                    Endpoint epoch{c.t,c.ox,c.oy,c.oz};
                    Vec2 centre=base(epoch),ray=range(epoch)-centre,h=radial(epoch)-centre;
                    Vec2 err=Vec2{c.x,c.y}-centre;
                    double gg=dot(ray,ray)+1.e-23,gn=std::sqrt(gg);
                    double hh=(ray.x*h.y-ray.y*h.x)/gn,rr=(ray.x*err.y-ray.y*err.x)/gn;
                    double tol=ast+model+.015*magh;
                    if(std::abs(rr)>1.10*std::abs(hh)+2.8*tol)return;
                    double variance=hh*hh+sq(tol/.64),hz=hh*rr/variance,sh=tol/std::sqrt(variance);
                    double peak=dot(ray,err)/gg,slope=-dot(ray,h)/gg;
                    double sg=tol/gn+.012;
                    if(std::abs(peak+slope*hz)>1.06+2.7*(sg+std::abs(slope)*sh))return;
                    double merit=c.quality-.55*sq(dm/.27)-.5*sq(rr)/variance-.45*sq(peak+slope*hz);
                    cuts.push_back({p.id,peak,hz,sg,sh,slope,merit});
                };
                if(!bin.start.empty()){
                    int basecell=(i-bin.x0)*bin.ny-bin.y0;
                    for(int k=bin.start[basecell+lo];k<bin.start[basecell+hi+1];k++)inspect(bin.ordered[k].second);
                }else{
                    for(int j=lo;j<=hi;j++){
                        uint64_t cell=key(i,j);
                        auto it=std::lower_bound(bin.ordered.begin(),bin.ordered.end(),cell,
                                                 [](auto &e,uint64_t z){return e.first<z;});
                        for(;it!=bin.ordered.end()&&it->first==cell;it++)inspect(it->second);
                    }
                }
            }
        }
        count[0]++;count[1]+=cuts.size();
        if(cuts.size()>160){
            auto weight=[](const auto &c){return c.merit-.52*std::log1p(18.*(c.sg+std::abs(c.slope)*c.sh));};
            std::nth_element(cuts.begin(),cuts.begin()+160,cuts.end(),[&](auto &p,auto &q){return weight(p)>weight(q);});
            cuts.resize(160);
        }
        auto centre=[](auto &c){return c.g+c.h*c.slope;};
        auto extent=[](auto &c){return 3.*(c.sg+std::abs(c.slope)*c.sh);};
        std::sort(cuts.begin(),cuts.end(),[&](auto &a,auto &b){return centre(a)-extent(a)<centre(b)-extent(b);});
        guesses.clear();
        for(size_t i=0;i<cuts.size();i++){
            const auto &p=cuts[i];double upper=centre(p)+extent(p);
            for(size_t k=i+1;k<cuts.size();k++){
                auto &q=cuts[k];
                if(centre(q)-extent(q)>upper)break;
                if(s[p.id].night==s[q.id].night||std::abs(p.h-q.h)>3.0*(p.sh+q.sh))continue;
                double v1=1./sq(p.sh),v2=1./sq(q.sh),v3=1./(sq(p.sg)+sq(q.sg));
                double dx=p.g-q.g,dh=p.slope-q.slope;
                double h=(v1*p.h+v2*q.h-dx*dh*v3)/(v1+v2+v3*dh*dh);
                double chi=v1*sq(h-p.h)+v2*sq(h-q.h)+v3*sq(dx+dh*h);
                if(chi>9.5||std::abs(h)>1.1)continue;
                double delta=.5*(p.g+q.g+h*(p.slope+q.slope));
                if(std::abs(delta)>1.15)continue;
                count[2]++;
                RemoteGuess z{int(i),int(k),delta,h,chi,0.};
                if(!screen){check(pair,z,g+delta*dg);continue;}
                double uncert=p.sg*p.sh+q.sg*q.sh+std::abs(p.slope-q.slope)*p.sh*q.sh;
                z.merit=p.merit+q.merit-.70*chi-.48*sq(delta)-.5*sq(h)-.65*std::log1p(30.*uncert);
                guesses.push_back(z);
            }
        }
        if(!screen||guesses.empty())return;
        auto best=[](auto &a,auto &b){return a.merit>b.merit;};
        if(guesses.size()>52){
            std::nth_element(guesses.begin(),guesses.begin()+52,guesses.end(),best);guesses.resize(52);
        }
        // A third independent strip through the intersection lends support
        // before the expensive propagation through every original visit.
        for(auto &z:guesses){
            int n1=s[cuts[z.p].id].night,n2=s[cuts[z.q].id].night;double support=-4.;
            for(auto &c:cuts){
                if(s[c.id].night==n1||s[c.id].night==n2)continue;
                double rr=sq((z.h-c.h)/(c.sh+.07))+sq((z.dg-c.g-c.slope*z.h)/(c.sg+.035));
                support=std::max(support,std::min(3.,c.merit)-.55*rr);
            }
            z.merit+=.74*support;
        }
        size_t nbest=std::min<size_t>(16,guesses.size());
        std::partial_sort(guesses.begin(),guesses.begin()+nbest,guesses.end(),best);
        for(size_t j=0;j<nbest;j++){
            const auto &z=guesses[j];check(pair,z,g+z.dg*dg);count[6]++;
        }
    }
};
}
extern "C" int remote_epochs(const Source *s,const int *owner,int n,const Unit *units,int nu,
        const Witness *w,const double *east,int threads,double seconds,double spread,double radial,
        int cap,Certified *out,long long *stats,int screen) {
    if(n<5||!nu)return 0;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        RemoteCatalogue cat(s,n,{east[0],east[1]},*w);
        std::atomic<int> next{0};std::vector<RemoteSearch> jobs;std::vector<std::thread> pool;
        for(int j=0;j<threads;j++){
            jobs.push_back({cat,owner,cap});
            jobs.back().screen=screen!=0;
            jobs.back().end=end;
        }
        for(auto &job:jobs)pool.emplace_back([&,p=&job]{
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=nu)break;
                p->search(units[i],spread,radial);
            }
        });
        for(auto &j:pool)j.join();
        std::vector<Certified> all,four;
        for(auto &j:jobs){
            all.insert(all.end(),j.five.begin(),j.five.end());
            four.insert(four.end(),j.four.begin(),j.four.end());
            for(int k=0;k<7;k++)stats[k]+=j.count[k];
        }
        compact_certified(all,3*cap/4);compact_certified(four,cap-all.size());
        std::copy(all.begin(),all.end(),out);std::copy(four.begin(),four.end(),out+all.size());
        return int(all.size()+four.size());
    }catch(...){return -1;}
}

