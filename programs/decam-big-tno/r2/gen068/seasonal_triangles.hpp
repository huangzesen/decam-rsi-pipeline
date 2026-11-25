// Search a short arc in one season and a measured return in another. The
// remote sky is conditioned on the observed short arc before ranking it.
namespace seasonal_triangles {
using namespace visit_triplets;
using encounter::Box;
struct Triangle {int a,b,c;float g,merit,short_merit;};
struct VisitJob {int a,b;std::vector<int> remote;double value;};
struct Measured {Point p;double g,sg;float merit;};

V displacement(const Exposure &a,const Exposure &b,V pos,V east){
    double dt=b.t-a.t,g=std::sqrt(1.+dot(pos,pos))/44.,vc=std::sqrt(mu/std::pow(44.,3));
    V par{b.ox-a.ox-pos.x*(b.oz-a.oz),b.oy-a.oy-pos.y*(b.oz-a.oz)};
    return (east*(dt*vc)-par*g)*(1./(1.-g*b.oz));
}
std::vector<VisitJob> schedule(const Catalogue &cat){
    std::vector<VisitJob> jobs;
    int n=cat.fields.size();
    for(int a=0;a<n;a++){
        const auto &f=cat.fields[a];if(f.pts.empty())continue;
        V xy{.5*(f.sky.l+f.sky.r),.5*(f.sky.b+f.sky.t)};
        struct Option{int id;double score;};
        std::vector<Option> far;
        for(int c=0;c<n;c++){
            const auto &q=cat.fields[c];
            double age=std::abs(q.e.t-f.e.t);
            if(age<120.||age>4900.||q.pts.empty())continue;
            auto off=displacement(f.e,q.e,xy,cat.east);
            auto overlap=f.sky.shift(off,.004+.000020*age).meet(q.sky);
            if(overlap.area()<=0.)continue;
            double frac=std::min(1.,overlap.area()/std::max(1.e-10,std::min(f.sky.area(),q.sky.area())));
            double depth=std::clamp(cat.depth[c]-cat.depth[a],-2.5,2.);
            far.push_back({c,std::log(.02+frac)+.48*depth-.65*std::log(.4+age/365.)});
        }
        if(far.empty())continue;
        std::sort(far.begin(),far.end(),[](const auto &l,const auto &r){return l.score>r.score;});
        std::vector<int> remote;
        for(auto q:far){
            bool same=false;int season=0;
            for(int c:remote){
                double dt=std::abs(cat.fields[c].e.t-cat.fields[q.id].e.t);
                if(dt<.6)same=true;
                season+=dt<100.;
            }
            if(same||season>=5)continue;
            remote.push_back(q.id);
            if(remote.size()==18)break;
        }
        std::vector<VisitJob> local;
        for(int b=0;b<n;b++){
            const auto &h=cat.fields[b];double dt=h.e.t-f.e.t;
            if(dt<.5||dt>78.||h.pts.empty()||h.e.night==f.e.night)continue;
            auto move=displacement(f.e,h.e,xy,cat.east);
            auto common=f.sky.shift(move,.000024*dt+.00045).meet(h.sky);
            double overlap=common.area()/std::max(1.e-10,std::min(f.sky.area(),h.sky.area()));
            if(overlap<=0.)continue;
            double value=std::log(.02+std::min(1.,overlap))-.29*std::abs(std::log(dt/5.))+
                         .27*std::clamp(std::min(cat.depth[a],cat.depth[b])-23.,-3.,3.);
            local.push_back({a,b,remote,value});
        }
        std::sort(local.begin(),local.end(),[](const auto &l,const auto &r){return l.value>r.value;});
        for(int j=0;j<int(local.size());j++){
            local[j].value-=.82*std::log(1.+j);
            jobs.push_back(std::move(local[j]));
        }
    }
    std::sort(jobs.begin(),jobs.end(),[](const auto &l,const auto &r){return l.value>r.value;});
    return jobs;
}

void measure(const Catalogue &full,const Catalogue &sparse,const VisitJob &job,
             Clock::time_point end,int allowance,std::vector<Triangle> &out,long long *stats){
    const int a=job.a,b=job.b,mbins=48;
    const auto &one=full.fields[a],&two=full.fields[b];
    std::vector<Measured> local;std::vector<Triangle> triangles;
    int processed=0;
    for(const auto &p:one.pts){
        if((++processed&63)==0&&expired(end))break;
        if(p.mag>full.depth[b]+.65)continue;
        PairGate gate(full,p.row,b,43.5,.26,.24,.36);
        double dchi=(gate.dg.x*(gate.yy*gate.dg.x-gate.xy*gate.dg.y)+gate.dg.y*(
            gate.xx*gate.dg.y-gate.xy*gate.dg.x))/gate.det;
        double sg=gate.g*gate.variance*std::sqrt(std::clamp(1.-dchi,.015,1.));
        double area=std::sqrt(gate.det)/sq(as);
        double width=std::clamp(std::sqrt(24./(.003+area*gate.rate)),.75,2.8);
        local.clear();
        two.ellipse(gate.center,gate.xx,gate.xy,gate.yy,gate.det,width,
          [&](const Point &q,double dx,double dy,double chi){
            double dm=q.mag-p.mag,v=full.varmag(p.row,q.row);
            if(dm*dm>8.5*v)return;
            double g=gate.distance({dx,dy});if(!(g>1./160.&&g<1./22.))return;
            double score=.35*std::clamp(double(q.q),-8.,12.)-.44*chi-.48*dm*dm/v;
            local.push_back({q,g,sg,float(score)});
        });
        best(local,4);stats[0]+=local.size();
        for(auto &m:local){
            double g=m.g,h=-std::sqrt(mu*g*g*g)*dot(full.east,{p.x,p.y});
            double sh=.36*std::sqrt(mu*g*g*g);
            Transfer base(p,m.p,one.e,two.e,g,h),plus(p,m.p,one.e,two.e,g+.20*m.sg,h),
                     radial(p,m.p,one.e,two.e,g,h+.20*sh);
            double mean=.5*(p.mag+m.p.mag);
            int mb=std::clamp(int((mean-11.)*2.),0,mbins-1);
            double short_merit=m.merit+.35*std::clamp(double(p.q),-8.,12.)-
                               .55*std::log(.008+area*gate.rate);
            int start=int(triangles.size());
            for(int c:job.remote){
                if(full.depth[c]+.76<mean)continue;
                const auto &e=full.fields[c].e;
                V x=base.at(e),dg=(plus.at(e)-x)*5.,dh=(radial.at(e)-x)*5.;
                double frac=(e.t-one.e.t)/(two.e.t-one.e.t);
                double phase=std::abs(e.t-one.e.t)*std::sqrt(base.k);
                double noise=sq(.27*as)+sq(frac*full.sigma(m.p.row))+sq((1.-frac)*full.sigma(p.row))+
                             sq(.006*std::pow(phase,5));
                Ellipse cov{noise+dg.x*dg.x+dh.x*dh.x,dg.x*dg.y+dh.x*dh.y,noise+dg.y*dg.y+dh.y*dh.y};
                double det=cov.det();if(!(det>0.))continue;
                double area2=std::sqrt(det)/sq(as);
                const auto &pop=area2*full.rho[c][mb]<.65?full:sparse;
                double clutter=area2*pop.rho[c][mb];
                if(clutter>45.)continue;
                double pv=sq(.24)+sq(.30*(full.raw.lever[c]-.5*(full.raw.lever[a]+full.raw.lever[b])));
                double rad=std::clamp(std::sqrt(20./(.01+clutter)),.65,2.8);
                pop.fields[c].ellipse(x,cov.xx,cov.xy,cov.yy,det,rad,
                  [&](const Point &r,double dx,double dy,double chi){
                    double dm=r.mag-mean;if(dm*dm>10.*pv)return;
                    double ug=dot(dg,cov.solve({dx,dy})),updated=g+m.sg*ug;
                    if(!(updated>1./240.&&updated<1./20.))return;
                    double score=short_merit+.39*std::clamp(double(r.q),-8.,12.)-
                                 .46*chi-.46*dm*dm/pv-.66*std::log(.008+clutter);
                    triangles.push_back({p.row,r.row,m.p.row,float(updated),float(score),float(short_merit)});
                    ++stats[1];
                });
                if(int(triangles.size())>start+24){
                    std::partial_sort(triangles.begin()+start,triangles.begin()+start+6,triangles.end(),
                        [](const auto &l,const auto &r){return l.merit>r.merit;});
                    triangles.resize(start+6);
                }
            }
            if(int(triangles.size())>start+5){
                std::partial_sort(triangles.begin()+start,triangles.begin()+start+5,triangles.end(),
                    [](const auto &l,const auto &r){return l.merit>r.merit;});
                triangles.resize(start+5);
            }
        }
        if(int(triangles.size())>3*allowance+80)best(triangles,2*allowance);
    }
    best(triangles,allowance);
    out.insert(out.end(),triangles.begin(),triangles.end());
}
}
extern "C" int measure_seasonal_triangles(const original::Sky *sky,const double *d,const double *east,
        const visit_triplets::Member *medium,int nm,const visit_triplets::Member *full,int nf,
        int threads,double seconds,int cap,seasonal_triangles::Triangle *out,long long *stats){
    using namespace seasonal_triangles;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Catalogue broad(*sky,d,east,full,nf),thin(*sky,d,east,medium,nm);
        auto jobs=schedule(broad);
        std::vector<std::vector<Triangle>> batches(threads);
        std::vector<std::array<long long,2>> counts(threads);
        std::atomic<int> next{0};std::vector<std::thread> pool;
        int per=std::clamp(cap/std::max(1,std::min(350,int(jobs.size()))),180,3200);
        for(int k=0;k<threads;k++)pool.emplace_back([&,k]{
            auto &t=batches[k];
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=int(jobs.size()))break;
                measure(broad,thin,jobs[i],end,per,t,counts[k].data());
                if(t.size()>size_t(2*cap/threads+1))best(t,cap/threads+1);
            }
        });
        for(auto &w:pool)w.join();
        std::vector<Triangle> all;for(auto &t:batches)all.insert(all.end(),t.begin(),t.end());
        for(auto &v:counts){stats[0]+=v[0];stats[1]+=v[1];}
        stats[2]=std::min(int(jobs.size()),next.load());stats[3]=jobs.size();
        best(all,cap);std::sort(all.begin(),all.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        std::copy(all.begin(),all.end(),out);return all.size();
    }catch(...){return -1;}
}
extern "C" int extend_seasonal_triangles(const original::Sky *sky,const double *d,const double *east,
        const visit_triplets::Member *medium,int nm,const visit_triplets::Member *full,int nf,
        const seasonal_triangles::Triangle *tri,int n,int threads,double seconds,int cap,
        visit_triplets::Proposal *out,long long *stats){
    using namespace seasonal_triangles;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Catalogue broad(*sky,d,east,full,nf),thin(*sky,d,east,medium,nm);
        std::vector<std::vector<Proposal>> batches(threads);
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int k=0;k<threads;k++)pool.emplace_back([&,k]{
            auto &t=batches[k];Extend w{thin,broad};
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=n)break;
                const auto &v=tri[i];w.pair({v.a,v.b,v.merit,v.g},t,v.c,v.short_merit);
                if(t.size()>size_t(2*cap/threads+1))best(t,cap/threads+1);
            }
        });
        for(auto &w:pool)w.join();
        std::vector<Proposal> all;for(auto &t:batches)all.insert(all.end(),t.begin(),t.end());
        best(all,cap);std::sort(all.begin(),all.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        stats[0]=std::min(next.load(),n);std::copy(all.begin(),all.end(),out);return all.size();
    }catch(...){return -1;}
}
