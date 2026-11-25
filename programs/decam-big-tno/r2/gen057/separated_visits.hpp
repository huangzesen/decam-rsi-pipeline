// Long-baseline statistical ranging. Work in rotated tangent coordinates
// and share the pair-search cost estimate among sources on one sky tile.
namespace separated_visits {
using namespace encounter;
using Proposal = Unit;
struct Worker {
    const Bank &data;
    std::vector<Proposal> result,local;
    size_t cap;
    long long anchors=0,combinations=0;
    void add(const Tile &tile,const Point &a){
        const auto &first=data.fields[tile.exposure];
        const V pos{a.x,a.y};
        struct Partner {int e;double value;};
        std::vector<Partner> options;
        for(auto &v:tile.views){
            const auto &im=data.fields[v.e];
            double gap=std::abs(im.e.t-first.e.t);
            if(gap<110.||gap>data.family.span)continue;
            V center=v.mean+pos-tile.pos;
            const double det=Bank::detect(im.depth,a.mag);
            if(det<.09)continue;
            double cover=.28*v.prob+.72*im.coverage(center);
            if(cover<.03)continue;
            // A narrow across-track range or a sparse remote exposure
            // tests many more endpoint associations at fixed cost.
            double quality=std::log(det*cover)-.52*std::log1p(v.cost)
                          -.21*std::abs(std::log(gap/550.));
            options.push_back({v.e,quality});
        }
        if(options.empty())return;
        size_t take=std::min<size_t>(options.size(),22);
        std::partial_sort(options.begin(),options.begin()+take,options.end(),
                          [](const auto &a,const auto &b){return a.value>b.value;});
        int nights[7],nv=0;
        for(size_t j=0;j<take&&nv<7;j++){
            auto &opt=options[j];const auto &second=data.fields[opt.e];
            if(std::find(nights,nights+nv,second.e.night)!=nights+nv)continue;
            auto prior=relative(data.family,first.e,second.e,pos);
            local.clear();
            double pivot=1./data.family.radius;
            double scatter=sq(.27*(first.color-second.color))+a.variance;
            second.ellipse(prior.mean,prior.xx,prior.xy,prior.yy,prior.det,2.65,
                [&](const Point &b,double dx,double dy,double chi){
                    double phot=sq(a.mag-b.mag)/(scatter+b.variance);
                    if(phot>8.5)return;
                    double g=pivot+prior.range_axis.x*dx+prior.range_axis.y*dy;
                    if(g<=1./9000.||g>=1./17.)return;
                    combinations++;
                    int bin=std::clamp(int((b.mag-10.)*4.),0,79);
                    double expected=data.w.images[opt.e].density[bin]*std::sqrt(prior.det)/sq(as);
                    double quality=.65*(a.q+b.q)-.48*(phot+chi)-.34*std::log(.01+expected)
                                  -.20*std::log((scatter+b.variance)/.1);
                    local.push_back({a.row,b.row,float(quality),float(g)});
                });
            if(local.empty())continue;
            nights[nv++]=second.e.night;
            size_t n=std::min<size_t>(local.size(),5);
            std::partial_sort(local.begin(),local.begin()+n,local.end(),
                              [](const auto &a,const auto &b){return a.score>b.score;});
            result.insert(result.end(),local.begin(),local.begin()+n);
        }
        anchors++;
        if(result.size()>2*cap)trim(result,cap);
    }
};
}

extern "C" int separated_pairs(const Witness *w,const float *quality,const double *depth,const double *lever,
        const double *east,const encounter::Family *family,int limit,int threads,double seconds,int cap,
        separated_visits::Proposal *out,long long *stats){
    try{
        using namespace separated_visits;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Bank bank(*w,quality,depth,lever,east,*family,limit);
        bank.schedule();
        std::vector<Worker> workers;
        for(int i=0;i<threads;i++)workers.push_back({bank,{}, {},std::max<size_t>(1,cap/threads)});
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(auto &worker:workers)pool.emplace_back([&,job=&worker]{
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=int(bank.anchors.size()))break;
                auto a=bank.anchors[i];const auto &tile=bank.tiles[a.tile];
                job->add(tile,bank.fields[tile.exposure].pts[a.index]);
            }
        });
        for(auto &t:pool)t.join();
        std::vector<Proposal> all;
        for(auto &s:workers){
            stats[0]+=s.anchors;stats[1]+=s.combinations;
            all.insert(all.end(),s.result.begin(),s.result.end());
        }
        // Time ordering makes the two discovery directions one pair. Keep
        // its strongest prediction and test the actual detector coordinates.
        for(auto &p:all)if(w->images[w->objects[p.a].exposure].e.t>w->images[w->objects[p.b].exposure].e.t)
            std::swap(p.a,p.b);
        std::sort(all.begin(),all.end(),[](const auto &a,const auto &b){
            if(a.a!=b.a)return a.a<b.a;
            return a.b!=b.b?a.b<b.b:a.score>b.score;
        });
        all.erase(std::unique(all.begin(),all.end(),[](const auto &a,const auto &b){
            return a.a==b.a&&a.b==b.b;
        }),all.end());
        trim(all,size_t(cap));
        std::sort(all.begin(),all.end(),[](const auto &a,const auto &b){return a.score>b.score;});
        std::copy(all.begin(),all.end(),out);
        stats[2]=bank.anchors.size();stats[3]=all.size();
        return int(all.size());
    }catch(...){return -1;}
}
