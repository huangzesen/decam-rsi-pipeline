// Move individual directions to a common epoch, then solve for the range
// and radial speed of four measurements inside each coarse search cell.
namespace {
struct ReducedDirection {
    Vec2 xy, gamma, radial;
    double t,tg,th;
};

struct EpochTile {uint64_t key=0;int head=-1,count=0;};
struct MiddleDirection {
    Vec2 offset,g,h;
    int id;
    double variance;
};
struct EpochTriple {
    int a,b,c;float gamma,merit;
};
void compact_triples(std::vector<EpochTriple> &v,int limit) {
    std::sort(v.begin(),v.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    std::unordered_set<uint64_t> seen;
    int n=0;
    for(auto &t:v){
        uint64_t hash=(uint64_t(unsigned(t.a))*0x9e3779b97f4a7c15ull+
                       unsigned(t.b))*0xbf58476d1ce4e5b9ull+unsigned(t.c);
        if(!seen.insert(hash).second)continue;
        v[n++]=t;
        if(n==limit)break;
    }
    v.resize(n);
}

struct EpochSweep {
    const Source *src;
    const int *owner;
    const Witness &w;
    int n,limit;
    double epoch;
    Three pole;
    Clock::time_point until;
    std::vector<EpochTile> cells;
    std::vector<int> next_row,bucket;
    std::vector<ReducedDirection> reduced;
    std::vector<Certified> reports,fours;
    std::unordered_set<uint64_t> tried;
    std::array<long long,8> counts{};
    std::vector<EpochTriple> triples;
    bool record=false;
    float threshold=-1.e20;

    void prepare(const double *trial) {
        const double r=trial[0],vt=trial[1],tilt=trial[2],vr=trial[3];
        const double g=1./r,vc=std::sqrt(mu*g*g*g),width=trial[4]*as;
        // Time and observatory location are shared by every detection in
        // an exposure. Evaluate the gravity expansion once per visit.
        struct TimeLaw {
            double dt,f,t,fg,tg,fh,th,distance,co,si;
        };
        std::vector<TimeLaw> law(w.images.size());
        for(size_t i=0;i<law.size();i++){
            double dt=w.images[i].e.t-epoch,z=sq(vc*dt),b=vc*dt*vr;
            double f=1.-.5*z*(1.-b)+z*z/24.,sf=1.-z/6.+z*b/4.+z*z/120.;
            double df=-1.5*z*(1.-b)+z*z/4.;
            double ds=-.5*z+.75*z*b+z*z/20.;
            double fh=.5*z*vc*dt,sh=.25*z*vc*dt;
            double tau=dt*sf/f;
            double phase=vc*dt*vt*(1.-b+b*b);
            law[i]={dt,f,tau,df/f,(dt*ds-tau*df)/f,fh/f,(dt*sh-tau*fh)/f,
                    r*(1.+b+.5*z*(vt*vt-1.)),std::cos(phase),std::sin(phase)};
        }
        size_t capacity=16;while(capacity<size_t(n)+size_t(n)/3)capacity*=2;
        cells.assign(capacity,{});next_row.resize(n);reduced.resize(n);
        const double incl_x=std::cos(tilt),incl_y=std::sin(tilt);
        for(int i=0;i<n;i++){
            auto &s=src[i];
            const auto &c=law[w.objects[owner[i]].exposure];
            Three ray{s.x,s.y,1.},observer{s.ox,s.oy,s.oz};
            ray=ray*(1./ray.length());
            double depth=ray.product(observer);
            double rho=std::sqrt(c.distance*c.distance-observer.product(observer)+depth*depth)-depth;
            Three u=(observer+ray*rho)*(1./c.distance);
            auto east=pole.cross(u);east=east*(1./std::max(.001,east.length()));
            auto tangent=east*incl_x+u.cross(east)*incl_y;
            auto common=u*c.co-tangent*c.si;
            int bx=int(std::floor(common.x/(common.z*width)+trial[5]));
            int by=int(std::floor(common.y/(common.z*width)+trial[6]));
            int mb=int(std::floor(s.mag/trial[7]+trial[8]));
            uint64_t key=permute(circular_key(bx,by,mb));
            size_t slot=key&(capacity-1);
            while(cells[slot].head>=0 && cells[slot].key!=key)slot=(slot+1)&(capacity-1);
            auto &cell=cells[slot];next_row[i]=cell.head;
            cell.head=i;cell.key=key;cell.count++;
            Vec2 p{s.x,s.y},par={g*(s.ox-s.x*s.oz)/c.f,g*(s.oy-s.y*s.oz)/c.f};
            // Projected distance is along the tangent frame z-axis. The
            // endpoint interpolation below also fits its time derivative.
            reduced[i]={p+par,par*(1.-c.fg),p*(vc*c.t)-par*c.fh,c.t,c.tg,c.th};
        }
    }
    static uint64_t signature(int a,int b,int c,int d) {
        std::array<int,4> k={a,b,c,d};std::sort(k.begin(),k.end());
        uint64_t hash=1469598103934665603ull;
        for(int i:k)hash=(hash^unsigned(i))*1099511628211ull;
        return hash;
    }
    void confirm(int a,int b,int c,int d,double gamma,double chi) {
        uint64_t id=signature(a,b,c,d);
        if(!tried.insert(id).second)return;
        if(tried.size()>1400000)tried.clear();
        counts[5]++;
        Unit ends{a,b,0.,float(gamma)},mids{c,d,0.,float(gamma)};
        Orbit fit;
        const Domain domain=gamma<1./110.?Domain{80.,8600.,6100.,true}:Domain{17.,140.,6100.,false};
        if(!quartet(src,ends,mids,gamma,fit,false,domain))return;
        counts[6]++;
        int rows[]={owner[a],owner[b],owner[c],owner[d]};
        Certified out{};
        // A long-baseline quartet measures the range. Starting the exact
        // fit directly also works when a truncated ephemeris would miss a
        // source at a distant epoch by several arcseconds.
        if(!certify_one(w,rows,4,fit.g,out,false,fit.chi<6.&&chi<10.))return;
        out.merit+=.12*(src[a].quality+src[b].quality+src[c].quality+src[d].quality);
        if(out.nights>=5){
            reports.push_back(out);counts[7]++;
            if(int(reports.size())>2*limit)compact_certified(reports,limit);
        }else if(out.nights==4){
            fours.push_back(out);
            if(int(fours.size())>limit)compact_certified(fours,limit/2);
        }
    }
    void tile(const int *ids,int size,const double *trial) {
        std::array<MiddleDirection,40> mid;
        const double g=1./trial[0],vc=std::sqrt(mu*g*g*g);
        int gap=record?2:3;
        for(int i=0;i+gap<size;i++)for(int j=i+gap;j<size;j++){
            int a=ids[i],b=ids[j];
            double span=src[b].t-src[a].t;
            if(span<110. || span>5100.)continue;
            if(std::abs(src[a].mag-src[b].mag)>.70+.24*std::abs(src[a].color-src[b].color))continue;
            const auto &u=reduced[a],&v=reduced[b];
            Vec2 rate=(v.xy-u.xy)*(1./(v.t-u.t));
            Vec2 baseg=u.gamma-rate*u.tg,baseh=u.radial-rate*u.th;
            Vec2 trendg=v.gamma-rate*v.tg-baseg,trendh=v.radial-rate*v.th-baseh;
            double mag=.5*(src[a].mag+src[b].mag);
            double lever=.5*(src[a].color+src[b].color);
            int nm=0;
            for(int k=i+1;k<j;k++){
                int c=ids[k];
                if(src[c].night==src[a].night||src[c].night==src[b].night)continue;
                if(std::abs(src[c].mag-mag)>.61+.28*std::abs(src[c].color-lever))continue;
                const auto &z=reduced[c];
                double t=z.t-u.t,frac=t/(v.t-u.t);
                Vec2 offset=z.xy-u.xy-rate*t;
                Vec2 axis=z.gamma-rate*z.tg-baseg-trendg*frac;
                Vec2 radial=z.radial-rate*z.th-baseh-trendh*frac;
                double gg=dot(axis,axis)+1.e-23;
                double peak=-dot(offset,axis)/gg;
                double cross=axis.x*offset.y-axis.y*offset.x;
                double hr=axis.x*radial.y-axis.y*radial.x;
                double arc=span*vc;
                double model=.22+(.00019/as)*arc*arc+
                             (.05/as)*arc*arc*arc*std::abs((src[c].t-epoch)*vc);
                double sigma=as*(std::hypot(.095,1./src[c].sn)+.10+model);
                if(std::abs(cross)>.75*std::abs(hr)+3.*sigma*std::sqrt(gg))continue;
                if(std::abs(peak)>.22+(.75*std::abs(dot(axis,radial))+3.*sigma*std::sqrt(gg))/gg)continue;
                mid[nm++]={offset,axis,radial,c,sq(sigma)};
                if(record){
                    // Integrate the uncertainty of a third position along
                    // the two directions swept out by range and range rate.
                    double cross2=sq(hr)/gg,perp=cross/std::sqrt(gg);
                    double dh=-perp*(hr/std::sqrt(gg))/(cross2+sq(sigma/.32));
                    double dg=peak-dh*dot(axis,radial)/gg;
                    double chi=sq(perp)/(sq(sigma)+.32*.32*cross2)+sq(dg/.14);
                    double dm=std::max(0.,std::abs(src[c].mag-mag)-.19*std::abs(src[c].color-lever));
                    double photo=sq(dm/.27)+.35*sq((src[a].mag-src[b].mag)/.36);
                    double area=std::hypot(sigma,.13*std::sqrt(gg))*std::hypot(sigma,.32*std::sqrt(cross2))/sq(as);
                    double merit=.80*(src[a].quality+src[b].quality+src[c].quality)-
                                 .5*(chi+photo)-.50*std::log1p(area);
                    if(chi<14. && merit>threshold){
                        triples.push_back({a,c,b,float(g*(1.+dg)),float(merit)});
                        if(triples.size()>=150000){
                            compact_triples(triples,65000);
                            threshold=triples.empty()?-1.e20:triples.back().merit;
                        }
                    }
                }
            }
            counts[2]++;counts[3]+=nm;
            for(int h=0;h<nm;h++)for(int k=h+1;k<nm;k++){
                const auto &p=mid[h],&q=mid[k];int c=p.id,d=q.id;
                if(src[c].night==src[d].night)continue;
                if(std::abs(src[c].mag-src[d].mag)>.72+.24*std::abs(src[c].color-src[d].color))continue;
                // Two transverse residuals per intermediate position, two
                // unknowns shared by all positions. Keep finite widths on
                // both unmeasured components in degenerate visit geometries.
                double aa=1./sq(.11),bb=1./sq(.32),ab=0.,ay=0.,by=0.,ss=0.;
                for(auto v:{&p,&q}){
                    double wt=1./v->variance;
                    aa+=wt*dot(v->g,v->g);bb+=wt*dot(v->h,v->h);ab+=wt*dot(v->g,v->h);
                    ay+=wt*dot(v->g,v->offset);by+=wt*dot(v->h,v->offset);ss+=wt*dot(v->offset,v->offset);
                }
                double det=aa*bb-ab*ab;
                double dg=(ab*by-ay*bb)/det,dh=(ab*ay-by*aa)/det;
                double chi=ss+ay*dg+by*dh;
                if(chi>18. || std::abs(dg)>.24 || std::abs(dh)>.85)continue;
                counts[4]++;
                confirm(a,b,c,d,g*(1.+dg),chi);
            }
        }
    }
    void sweep(const double *trial) {
        prepare(trial);
        int groups=0;
        for(const auto &cell:cells){
            int count=cell.count;
            if(count>=(record?3:4)){
                if((groups++&63)==0 && expired(until))return;
                bucket.clear();
                for(int i=cell.head;i>=0;i=next_row[i])bucket.push_back(i);
                constexpr int maxn=30;
                if(count>maxn){
                    std::nth_element(bucket.begin(),bucket.begin()+maxn,bucket.end(),[&](int a,int b){
                        return src[a].quality>src[b].quality;
                    });
                    count=maxn;
                }
                std::array<int,maxn> ids;
                std::copy_n(bucket.begin(),count,ids.begin());
                std::sort(ids.begin(),ids.begin()+count,[&](int a,int b){return src[a].t<src[b].t;});
                if(src[ids[count-1]].t-src[ids[0]].t>=110.){
                    counts[1]++;
                    tile(ids.data(),count,trial);
                }
            }
        }
        counts[0]++;
    }
};
}

extern "C" int epoch_sweeps(const Source *sources,const int *owner,int n,const Witness *w,
        const double *pole,double epoch,const double *trials,int ntrial,int threads,
        double seconds,int limit,Certified *out,long long *counts) {
    if(n<4||ntrial==0)return 0;
    try{
        auto until=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};
        const bool extend=seconds>=10.;
        auto scan_end=extend?until-std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(.30*seconds)):until;
        std::vector<EpochSweep> jobs;
        for(int i=0;i<threads;i++)
            jobs.push_back({sources,owner,*w,n,limit,epoch,{pole[0],pole[1],pole[2]},scan_end});
        std::vector<std::thread> workers;
        for(auto &job:jobs)workers.emplace_back([&,p=&job]{
            p->record=extend;
            while(!expired(scan_end)){
                int i=next.fetch_add(1);
                if(i>=ntrial)break;
                p->sweep(trials+9*i);
            }
        });
        for(auto &worker:workers)worker.join();
        if(extend&&!expired(until)){
            std::vector<EpochTriple> queue;
            for(auto &j:jobs)queue.insert(queue.end(),j.triples.begin(),j.triples.end());
            compact_triples(queue,90000);
            next=0;workers.clear();
            for(auto &job:jobs)workers.emplace_back([&,p=&job]{
                while(!expired(until)){
                    int i=next.fetch_add(1);
                    if(i>=int(queue.size()))break;
                    auto &t=queue[i];int core[]={owner[t.a],owner[t.b],owner[t.c]};
                    follow_uncertain(*w,core,3,t.gamma,p->reports,true);
                    if(p->reports.size()>size_t(2*limit))compact_certified(p->reports,limit);
                }
            });
            for(auto &worker:workers)worker.join();
        }
        std::vector<Certified> all,four;
        for(auto &job:jobs){
            all.insert(all.end(),job.reports.begin(),job.reports.end());
            four.insert(four.end(),job.fours.begin(),job.fours.end());
            for(int i=0;i<8;i++)counts[i]+=job.counts[i];
        }
        compact_certified(all,3*limit/4);compact_certified(four,limit-all.size());
        std::copy(all.begin(),all.end(),out);std::copy(four.begin(),four.end(),out+all.size());
        return int(all.size()+four.size());
    }catch(...){return -1;}
}
