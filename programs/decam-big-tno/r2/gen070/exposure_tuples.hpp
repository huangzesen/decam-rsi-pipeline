// Draw a path through the measured sky coverage. Use its actual encounters
// to share an exposure schedule across all sources in a camera patch.
namespace exposure_tuples {
using namespace encounter;
using cadence_ranging::Ellipse;
using tile_triplets::Pair;
using tile_triplets::Third;
using tile_triplets::Worker;
inline double pair_speed(double g){return std::sqrt(mu*g*g*g);}

struct Random {
    uint64_t state;
    double uniform(){
        state+=0x9e3779b97f4a7c15ULL;
        uint64_t z=state;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27))*0x94d049bb133111ebULL;
        return ((z^(z>>31))>>11)*1.1102230246251565e-16;
    }
    double symmetric(){return 2.*uniform()-1.;}
};

// Local distribution about one sampled orbit. Condition the angular
// position at its anchor, so two observations constrain inverse range.
struct Path {
    V pos,velocity;
    double g,h,k,radial,speed,sg,sv,sw,sh,t0;
    Path(V sky,const Exposure &e,double radius,double along,double cross,double rv,
         double range_width,double velocity_width,double radial_width){
        double norm=std::sqrt(1.+dot(sky,sky));
        g=norm/radius;sg=g*range_width;t0=e.t;
        pos=sky*(1.-g*e.oz)+V{e.ox,e.oy}*g;
        double vc=std::sqrt(mu/(radius*radius*radius));
        velocity={along*vc*norm,cross*vc*norm};
        h=(rv*norm-dot(pos,velocity)/vc)*vc;
        double ns=1.+dot(pos,pos);
        k=mu*std::pow(g/std::sqrt(ns),3);
        radial=(dot(pos,velocity)+h)/ns;speed=(dot(velocity,velocity)+h*h)/ns;
        sv=velocity_width*vc*norm;sw=sv*1.2;sh=radial_width*vc;
    }
    V at(const Exposure &e,double *scale=nullptr,double *lag=nullptr) const{
        double t=e.t-t0,z=k*t*t,u=t*radial;
        double f=1.-.5*z*(1.-u)+z*((3.*speed-2.*k)*t*t-15.*u*u)/24.;
        double s=t*(1.-z/6.+z*u/4.+z*((9.*speed-8.*k)*t*t-45.*u*u)/120.);
        double inv=1./(f+s*h-g*e.oz);
        if(scale)*scale=f*inv;
        if(lag)*lag=s*inv;
        return (pos*f+velocity*s-V{e.ox,e.oy}*g)*inv;
    }
};
struct View {
    int e; V point,dg;
    double scale,overlap,depth,jitter;
    Ellipse error;
};
struct Plan {
    const View *second;
    std::vector<Third> thirds;
};

static std::vector<View> visible(const Bank &bank,const Tile &tile,const Path &path,Random &rand){
    const auto &a=bank.fields[tile.exposure].e;
    std::vector<View> result;
    for(int j=0;j<int(bank.fields.size());j++){
        const auto &f=bank.fields[j]; const auto &e=f.e;
        if(e.night==a.night||f.pts.empty())continue;
        double scale,lag;
        V c=path.at(e,&scale,&lag);
        double pad=.0033;
        if(c.x<f.sky.l-pad||c.x>f.sky.r+pad||c.y<f.sky.b-pad||c.y>f.sky.t+pad)continue;
        double cover=.36*f.coverage(c);
        for(int x:{-1,1})for(int y:{-1,1})cover+=.16*f.coverage(c+V{.00135*x,.00135*y});
        if(cover<.045)continue;
        // The tangent to the range coordinate includes the dependence of
        // circular speed on range, plus changing topocentric perspective.
        V dg=((V{a.ox,a.oy}-tile.pos*a.oz)*scale-
              (V{e.ox,e.oy}-c*e.oz)*(scale/std::max(.65,1.-.5*path.k*sq(e.t-a.t))))*path.sg+
             (path.velocity-c*path.h)*(1.5*lag*path.sg/path.g);
        V radial=c*(lag*path.sh);
        double px=sq(lag*path.sv)+sq(.33*as),py=sq(lag*path.sw)+sq(.33*as);
        Ellipse cov{px+sq(dg.x)+sq(radial.x),dg.x*dg.y+radial.x*radial.y,py+sq(dg.y)+sq(radial.y)};
        result.push_back({j,c,dg,scale*(1.-path.g*a.oz),cover,f.depth,rand.uniform(),cov});
    }
    return result;
}

struct Rank {int i;double value;};
static bool higher(const Rank &a,const Rank &b){return a.value>b.value;}
static std::vector<View> nights(const Bank &bank,std::vector<View> &all,double &depth){
    std::sort(all.begin(),all.end(),[](auto &a,auto &b){
        return a.depth+.55*std::log(a.overlap)+.7*a.jitter>b.depth+.55*std::log(b.overlap)+.7*b.jitter;
    });
    std::vector<View> list;
    for(auto &v:all){
        auto n=bank.fields[v.e].e.night;
        if(std::none_of(list.begin(),list.end(),[&](auto &u){return bank.fields[u.e].e.night==n;}))
            list.push_back(v);
        if(list.size()>=30)break;
    }
    if(list.size()>=4){
        std::vector<double> depths;
        for(auto &v:list)depths.push_back(v.depth+.25*std::log(v.overlap));
        std::nth_element(depths.begin(),depths.begin()+3,depths.end(),std::greater<double>());
        depth=depths[3];
    }
    return list;
}

static bool plans(const Bank &bank,const Tile &tile,const std::vector<View> &seen,const Path &path,
                  double mag,double rv,Random &rand,Plan &plan){
    const auto &a=bank.fields[tile.exposure];
    bool long_pair=rand.uniform()<.54;
    std::vector<Rank> second;
    for(int i=0;i<int(seen.size());i++){
        const auto &v=seen[i];double dt=std::abs(bank.fields[v.e].e.t-a.e.t);
        if(dt<.7||dt>(path.g>.04?3100.:5300.))continue;
        double profit=v.overlap*Bank::detect(v.depth,mag);
        double cost=std::sqrt(v.error.det())/sq(as)*tile_triplets::crowd(bank,v.e,mag);
        second.push_back({i,std::log(.003+profit)-.33*std::log(.07+cost)+1.55*v.jitter
                            +(long_pair&&dt>170.?2.2:0.)});
    }
    if(second.empty())return false;
    auto &b=seen[std::max_element(second.begin(),second.end(),
                      [](auto &x,auto &y){return x.value<y.value;})->i];
    plan.second=&b;
    Point p{float(tile.pos.x),float(tile.pos.y),float(mag),0,.06f,-1};
    Point q{float(b.point.x),float(b.point.y),float(mag),0,.06f,-1};
    double remain=std::max(.005,1.-dot(b.dg,b.error.solve(b.dg)));
    Pair pair(p,q,a.e,bank.fields[b.e].e,path.g,path.sg*std::sqrt(remain),path.sh/pair_speed(path.g),rv);
    std::vector<cadence_ranging::Forecast> forecast;
    std::vector<Rank> thirds;
    for(auto &v:seen){
        if(v.e==b.e)continue;
        auto z=pair.forecast(v.e,bank.fields[v.e]);
        if(!(z.covariance.det()>0.))continue;
        double crowd=tile_triplets::crowd(bank,v.e,mag);
        double cost=crowd*std::sqrt(z.covariance.det())/sq(as);
        double profit=v.overlap*Bank::detect(v.depth,mag);
        double span=std::max(std::abs(a.e.t-bank.fields[b.e].e.t),std::abs(a.e.t-bank.fields[v.e].e.t));
        z.merit=std::log(.005+profit)-.38*std::log(.06+cost)+1.2*v.jitter;
        thirds.push_back({int(forecast.size()),z.merit+(span>110.?1.15:0.)});
        forecast.push_back(z);
    }
    std::sort(thirds.begin(),thirds.end(),higher);
    for(auto &choice:thirds){
        const auto &c=forecast[choice.i];Third plan3{c.visit,{}};
        std::vector<Rank> next;
        for(const auto &d:forecast){
            if(d.visit==c.visit)continue;
            auto cov=cadence_ranging::given(d,c,{0.,0.}).cov;
            if(!(cov.det()>0.))continue;
            double cost=tile_triplets::crowd(bank,d.visit,mag)*std::sqrt(cov.det())/sq(as);
            next.push_back({d.visit,d.merit*.36-.62*std::log(.04+cost)});
        }
        std::sort(next.begin(),next.end(),higher);
        for(auto &d:next){plan3.next.push_back(d.i);if(plan3.next.size()==7)break;}
        if(plan3.next.size()>=2)plan.thirds.push_back(std::move(plan3));
        if(plan.thirds.size()>=3)break;
    }
    return !plan.thirds.empty();
}

struct Trial {
    Worker fit;
    std::array<long long,5> count{};
    explicit Trial(const Bank &bank,Clock::time_point stop):fit{bank,stop}{}

    void search(const Tile &anchor,const Tile &corridor,const Path &path,const Plan &plan,
                double rv,double depth,Random &rand){
        auto &bank=fit.bank;
        auto &a=bank.fields[anchor.exposure];auto &vb=*plan.second;auto &b=bank.fields[vb.e];
        double det=vb.error.det();if(!(det>0.))return;
        double posterior=std::sqrt(std::max(.006,1.-dot(vb.dg,vb.error.solve(vb.dg))));
        std::vector<tile_triplets::Hit> hits;
        // Rotate the first source when an unusually dense tile uses its
        // time slice. All ordinary tiles fit inside a slice.
        size_t offset=size_t(rand.uniform()*anchor.anchors.size());
        auto local_end=std::min(fit.end,Clock::now()+std::chrono::milliseconds(38));
        for(size_t ix=0;ix<anchor.anchors.size();ix++){
            if((ix%4)==0&&expired(local_end))break;
            const auto &p=a.pts[anchor.anchors[(ix+offset)%anchor.anchors.size()]];
            if(p.mag>depth+.70||p.mag>b.depth+.72)continue;
            V centre=vb.point+(V{p.x,p.y}-anchor.pos)*vb.scale;
            double area=std::sqrt(det)/sq(as)*tile_triplets::crowd(bank,vb.e,p.mag);
            // Sample a smaller piece of a crowded velocity distribution.
            // The varying central trial path provides different pieces in
            // subsequent passes without a per-source partner ranking.
            double hook=std::clamp(std::sqrt(17./(.01+area)),.38,2.55);
            ++count[2];
            b.ellipse(centre,vb.error.xx,vb.error.xy,vb.error.yy,det,hook,
                [&](const Point &q,double dx,double dy,double chi){
                    if(expired(local_end))return;
                    double dm=sq(q.mag-p.mag)/(p.variance+q.variance+sq(.27*(a.color-b.color)));
                    if(dm>8.5)return;
                    double gamma=path.g+path.sg*dot(vb.dg,vb.error.solve({dx,dy}));
                    if(!(gamma>1./9500.&&gamma<1./15.))return;
                    ++count[3];
                    Pair pair(p,q,a.e,b.e,gamma,path.sg*posterior,path.sh/pair_speed(gamma),rv);
                    double mag=.5*(p.mag+q.mag);
                    for(auto &c:plan.thirds){
                        auto &f=bank.fields[c.visit];
                        if(f.depth<mag-.68)continue;
                        auto fc=pair.forecast(c.visit,f);auto cov=fc.covariance;double vdet=cov.det();
                        if(!(vdet>0.))continue;
                        double clutter=tile_triplets::crowd(bank,c.visit,mag)*std::sqrt(vdet)/sq(as);
                        if(clutter>410.)continue;
                        hits.clear();
                        f.ellipse(fc.center,cov.xx,cov.xy,cov.yy,vdet,3.,
                            [&](const Point &r,double,double,double residual){
                                double var=.029+r.variance+sq(.27*(f.color-.5*(a.color+b.color)));
                                double dev=sq(r.mag-mag)/var;
                                if(dev<8.)hits.push_back({&r,residual+dev-.42*r.q});
                            });
                        if(hits.size()>14){
                            std::nth_element(hits.begin(),hits.begin()+14,hits.end(),[](auto &u,auto &v){return u.chi<v.chi;});
                            hits.resize(14);
                        }
                        count[4]+=hits.size();
                        for(auto &hit:hits){
                            double merit=.42*(p.q+q.q)-.5*dm-.28*(chi+hit.chi)-
                                         .45*std::log(.03+area)-.45*std::log(.02+clutter);
                            fit.triangle(corridor,a,b,p,q,pair,c,fc,hit,merit);
                        }
                    }
                });
            if(fit.arcs.size()>5000)fit.arcs.clear();
            if(fit.confirmed.size()>20000)trim_arcs(fit.confirmed,10000);
        }
    }

    void run(const Tile &tile,Random &rand){
        ++count[0];
        if(tile.anchors.empty())return;
        auto &bank=fit.bank;auto &a=bank.fields[tile.exposure];
        double mix=rand.uniform(),radius,along,cross,rv,sg,sv,sh;
        if(mix<.51){
            radius=43.*std::exp(.15*rand.symmetric());along=1.+.09*rand.symmetric();
            cross=.22*rand.symmetric();rv=.18*rand.symmetric();sg=.072;sv=.078;sh=.20;
        }else if(mix<.90){
            radius=38.*std::exp(.43*rand.symmetric());along=.93+.25*rand.symmetric();
            cross=.57*rand.symmetric();rv=.40*rand.symmetric();sg=.11;sv=.13;sh=.32;
        }else{
            radius=std::exp(std::log(59.)+rand.uniform()*std::log(1800./59.));
            along=.77+.48*rand.symmetric();cross=.65*rand.symmetric();
            rv=.47*rand.symmetric();sg=.24;sv=.22;sh=.44;
        }
        Path path(tile.pos,a.e,radius,along,cross,rv,sg,sv,sh);
        auto all=visible(bank,tile,path,rand);
        if(all.size()<4)return;
        double depth=0.;
        auto visits=nights(bank,all,depth);
        if(visits.size()<4)return;
        double low=a.e.t,high=a.e.t;
        for(auto &v:visits){
            double t=bank.fields[v.e].e.t;low=std::min(low,t);high=std::max(high,t);
        }
        if(high-low<85.)return;
        double mag=std::min(depth,a.depth)-.2-.95*rand.uniform();
        Plan plan{};
        if(!plans(bank,tile,visits,path,mag,rv,rand,plan))return;
        ++count[1];
        Tile corridor{tile.exposure,tile.pos,{},{}};
        // Final confirmation searches the full orbit index and includes
        // overlapping, independent exposures omitted from the seed schedule.
        for(auto &v:all)corridor.views.push_back({v.e,v.overlap,0.,v.point,0.,0.,0.});
        search(tile,corridor,path,plan,rv,depth,rand);
    }
};
}

extern "C" int exposure_tuple_search(const Witness *index,const float *quality,const double *depth,
       const double *lever,const double *east,int limit,int threads,double seconds,
       int cap,Certified *out,long long *stats){
    using namespace exposure_tuples;
    try{
        auto start=Clock::now();
        auto until=[&](double f){return start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds*f));};
        const Family family{44.,.14,1.,.18,.24,.25,5200.};
        Bank bank(*index,quality,depth,lever,east,family,limit);
        if(bank.tiles.empty())return 0;
        std::vector<size_t> order(bank.tiles.size());
        Random shuffle{91638924};
        for(size_t i=0;i<order.size();i++){
            size_t other=size_t(shuffle.uniform()*(i+1));order[i]=order[other];order[other]=i;
        }
        std::vector<Trial> jobs;std::vector<std::thread> pool;std::atomic<size_t> next{0};
        for(int j=0;j<threads;j++)jobs.emplace_back(bank,until(.88));
        for(auto &job:jobs)pool.emplace_back([&,worker=&job]{
            while(!expired(worker->fit.end)){
                size_t i=next.fetch_add(1);
                if(i>=order.size()*18)break;
                Random rand{15147319ULL+uint64_t(i)*3935559000370003845ULL};
                worker->run(bank.tiles[order[i%order.size()]],rand);
            }
        });
        for(auto &t:pool)t.join();pool.clear();
        std::vector<Arc> arcs;
        for(auto &job:jobs){
            for(int i=0;i<5;i++)stats[i]+=job.count[i];
            auto &v=job.fit.confirmed;arcs.insert(arcs.end(),v.begin(),v.end());
        }
        trim_arcs(arcs,26000);stats[5]=arcs.size();next=0;
        std::vector<std::vector<Certified>> lists(threads);
        for(int j=0;j<threads;j++)pool.emplace_back([&,j]{
            while(!expired(until(1.))){
                size_t i=next.fetch_add(1);if(i>=arcs.size())break;
                const auto &arc=arcs[i];Certified c{};
                if(certify_one(*index,arc.ids,5,arc.g,c)&&c.nights>=5){
                    c.merit+=.19*arc.score;lists[j].push_back(c);
                }
            }
        });
        for(auto &t:pool)t.join();
        std::vector<Certified> joined;
        for(auto &list:lists)joined.insert(joined.end(),list.begin(),list.end());
        compact_certified(joined,cap);stats[6]=joined.size();
        std::copy(joined.begin(),joined.end(),out);return int(joined.size());
    }catch(...){return -1;}
}
