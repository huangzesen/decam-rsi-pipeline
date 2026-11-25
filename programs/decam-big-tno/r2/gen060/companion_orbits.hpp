// Link two measured two-night arcs. A common uncertainty in the first
// pair's two centroids, range and radial speed propagates into BOTH
// positions of the second pair, even for a many-year extrapolation.
namespace companion_orbits {
using namespace encounter;
using cadence_ranging::Forecast;
struct Seed {int row[3];float score,g;};
struct Images {
    const Witness &w;
    const Seed *pair;
    const double *lever;
    std::vector<Field> fields;
    std::vector<int> occupied;
    Images(const Witness &index,const Seed *pairs,int n,const double *color)
        :w(index),pair(pairs),lever(color),fields(w.images.size()){
        std::vector<std::vector<Point>> data(fields.size());
        for(int k=0;k<n;k++){
            auto &v=w.objects[pair[k].row[0]],&p=w.objects[pair[k].row[1]];
            if(!std::isfinite(v.mag+p.mag))continue;
            data[v.exposure].push_back({v.x,v.y,float(.5*(v.mag+p.mag)),pair[k].score,0.f,k});
        }
        for(size_t e=0;e<fields.size();e++){
            auto &f=fields[e];f.e=w.images[e].e;f.color=color[e];
            if(!data[e].empty())occupied.push_back(e);
            f.build(data[e],data[e].size(),false);
        }
    }
    Point at(int row)const{
        auto &p=w.objects[row];return {p.x,p.y,p.mag,0.f,0.f,row};
    }
    double error(int row)const{
        return (w.sigma?std::clamp(double(w.sigma[row]),.070,.44):.17)*as;
    }
};
struct Prediction {
    Transfer nominal,range,radial;
    double sa,sb,dt,g,h,sg,sh,vc,fb,s1,az,bz;
    Prediction(const Images &sky,const Seed &pair,const double *east):
        nominal(make(sky,pair,east,0.,0.)),range(make(sky,pair,east,.002,0.)),
        radial(make(sky,pair,east,0.,.006)),
        sa(sky.error(pair.row[0])),sb(sky.error(pair.row[1])){
        auto &a=sky.w.images[sky.w.objects[pair.row[0]].exposure].e;
        auto &b=sky.w.images[sky.w.objects[pair.row[1]].exposure].e;
        g=nominal.g;h=nominal.h;dt=b.t-a.t;vc=std::sqrt(mu*g*g*g);
        V deriv=(range.rate-nominal.rate)*(1./(.002*g));
        double along=deriv.x*east[0]+deriv.y*east[1]-1.5*vc/g;
        double cross=deriv.y*east[0]-deriv.x*east[1];
        double variance=(sa*sa+sb*sb)/(dt*dt);
        sg=1./std::sqrt(1./sq(.32*g)+sq(along)/(sq(.25*vc)+variance)+
                           sq(cross)/(sq(.36*vc)+variance));sh=.35*vc;
        nominal.coeff(dt,fb,s1);az=1.-g*a.oz;bz=fb+s1*h-g*b.oz;
    }
    static Transfer make(const Images &sky,const Seed &p,const double *east,double dg,double dh){
        auto a=sky.at(p.row[0]),b=sky.at(p.row[1]);
        auto &ta=sky.w.images[sky.w.objects[a.row].exposure].e;
        auto &tb=sky.w.images[sky.w.objects[b.row].exposure].e;
        double g=std::clamp(double(p.g),1./8000.,1./16.);
        double h=(dh-double(a.x)*east[0]-double(a.y)*east[1])*std::sqrt(mu*g*g*g);
        return Transfer(a,b,ta,tb,g*(1.+dg),h);
    }
    Forecast at(int visit,const Exposure &e)const{
        double t=e.t-nominal.t0,f,s;nominal.coeff(t,f,s);
        double den=f+s*h-g*e.oz,ratio=s/s1;
        double l=(f-ratio*fb)*az/den*sa,r=ratio*bz/den*sb;
        V center=nominal.at(e),x=(range.at(e)-center)*(sg/(.002*g)),
          y=(radial.at(e)-center)*(sh/(.006*vc));
        double noise=sq(.20*as)+sq(.003*std::pow(t*vc,4));
        double diag=l*l+r*r+noise;
        return {visit,center,x,y,l,r,noise,0.,{diag+x.x*x.x+y.x*y.x,
               x.x*x.y+y.x*y.y,diag+x.y*x.y+y.y*y.y},0.};
    }
};
struct Worker {
    const Images &sky;const double *east;Clock::time_point until;
    std::vector<Arc> arcs;std::vector<Forecast> views;std::vector<int> stamp;
    std::array<long long,5> stats{};
    void search(int k){
        const auto &w=sky.w;auto &pair=sky.pair[k];
        const auto &a=w.objects[pair.row[0]],&b=w.objects[pair.row[1]];
        const auto &ta=sky.fields[a.exposure].e,&tb=sky.fields[b.exposure].e;
        if(std::abs(ta.t-tb.t)<.45)return;
        if(views.empty()){views.resize(sky.fields.size());stamp.assign(views.size(),-1);}
        Prediction orbit(sky,pair,east);
        double mag=.5*(a.mag+b.mag),color=.5*(sky.lever[a.exposure]+sky.lever[b.exposure]);
        auto get=[&](int e)->const Forecast &{
            if(stamp[e]!=k){views[e]=orbit.at(e,sky.fields[e].e);stamp[e]=k;}return views[e];
        };
        stats[0]++;
        for(int e:sky.occupied){
            auto &field=sky.fields[e];auto &tc=field.e;
            if(tc.night==ta.night||tc.night==tb.night)continue;
            if(std::min(std::abs(tc.t-ta.t),std::abs(tc.t-tb.t))<55.)continue;
            auto &v=get(e);double det=v.covariance.det();
            if(!(det>0.))continue;
            double wx=3.1*std::sqrt(v.covariance.xx),wy=3.1*std::sqrt(v.covariance.yy);
            if(v.center.x+wx<field.sky.l||v.center.x-wx>field.sky.r||
               v.center.y+wy<field.sky.b||v.center.y-wy>field.sky.t)continue;
            stats[1]++;
            field.ellipse(v.center,v.covariance.xx,v.covariance.xy,v.covariance.yy,det,3.1,
                [&](const Point &other,double dx,double dy,double chi){
                    int j=other.row;if(j<=k)return;
                    auto &q=sky.pair[j];const auto &d=w.objects[q.row[1]];
                    const auto &td=sky.fields[d.exposure].e;
                    if(td.night==ta.night||td.night==tb.night)return;
                    if(std::min(std::abs(td.t-ta.t),std::abs(td.t-tb.t))<55.)return;
                    double mvar=.074+sq(.26*(color-.5*(field.color+sky.lever[d.exposure])));
                    double cm=sq(mag-other.mag)/mvar;if(cm>12.)return;
                    stats[2]++;
                    auto &u=get(d.exposure);
                    auto cond=cadence_ranging::given(u,v,{dx,dy});
                    // D and C share the truncation of the long extrapolation.
                    // Only the *change* in model error broadens their rate.
                    double dt=td.t-tc.t;
                    double inflate=.023*std::abs(dt)*as+
                        .022*std::abs(dt)*orbit.sh*(std::abs(dx)+std::abs(dy));
                    auto cov=cond.cov;cov.xx+=sq(inflate);cov.yy+=sq(inflate);
                    V z{d.x-cond.center.x,d.y-cond.center.y};
                    if(sq(z.x)>18.*cov.xx||sq(z.y)>18.*cov.yy||!(cov.det()>0.))return;
                    double conditional=dot(z,cov.solve(z));if(conditional>17.)return;
                    V update=v.covariance.solve({dx,dy});
                    double gamma=orbit.g+orbit.sg*dot(v.range,update);
                    if(!(gamma>1./9000.&&gamma<1./15.))return;
                    stats[3]++;
                    Arc arc{{pair.row[0],pair.row[1],q.row[0],q.row[1],-1},float(gamma),
                            float(pair.score+q.score-.5*(chi+conditional+cm))};
                    if(preliminary(w,arc)){arcs.push_back(arc);stats[4]++;}
            });
        }
        if(arcs.size()>38000)trim_arcs(arcs,20000);
    }
};
}

extern "C" int join_companions(const Witness *index,const companion_orbits::Seed *pair,int n,
    const double *lever,const double *east,int threads,double seconds,int capacity,Certified *out,long long *stats){
    using namespace companion_orbits;
    try{
        auto start=Clock::now();
        auto deadline=[&](double fraction){return start+std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(seconds*fraction));};
        Images sky(*index,pair,n,lever);
        std::vector<Worker> jobs;std::vector<std::thread> workers;std::atomic<int> next{0};
        for(int t=0;t<threads;t++)jobs.push_back({sky,east,deadline(.75)});
        for(auto &job:jobs)workers.emplace_back([&,p=&job]{
            while(!expired(p->until)){int k=next.fetch_add(1);if(k>=n)break;p->search(k);}
        });
        for(auto &t:workers)t.join();workers.clear();
        std::vector<Arc> arcs;
        for(auto &job:jobs){
            for(int j=0;j<5;j++)stats[j]+=job.stats[j];
            arcs.insert(arcs.end(),job.arcs.begin(),job.arcs.end());
        }
        trim_arcs(arcs,42000);next.store(0);
        std::vector<std::vector<Certified>> parts(threads);
        for(int j=0;j<threads;j++)workers.emplace_back([&,j]{
            std::vector<int> retry;
            while(!expired(deadline(.94))){
                int i=next.fetch_add(1);if(i>=int(arcs.size()))break;
                auto &a=arcs[i];Certified c{};
                if(certify_one(*index,a.ids,4,a.g,c,false,true)){
                    c.merit+=.32*a.score;parts[j].push_back(c);
                    if(c.nights==4)retry.push_back(i);
                }
            }
            for(int i:retry){
                if(expired(deadline(1.)))break;
                follow_uncertain(*index,arcs[i].ids,4,arcs[i].g,parts[j],true);
            }
        });
        for(auto &t:workers)t.join();
        std::vector<Certified> five,four;
        for(auto &p:parts)for(auto &c:p)(c.nights>=5?five:four).push_back(c);
        compact_certified(five,capacity);stats[5]=five.size();
        compact_certified(four,std::min<size_t>(capacity-five.size(),capacity/10));
        five.insert(five.end(),four.begin(),four.end());
        std::copy(five.begin(),five.end(),out);return int(five.size());
    }catch(...){return -1;}
}

struct SweptQuartet {int rows[4];float g,score;};
extern "C" int verify_companion_quads(const Witness *index,const SweptQuartet *seeds,int n,
    int threads,double seconds,int cap,Certified *out,long long *stats){
    using namespace encounter;
    try{
        auto start=Clock::now();
        auto until=[&](double t){return start+std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(t*seconds));};
        std::atomic<int> next{0};std::vector<std::thread> team;
        std::vector<std::vector<Arc>> chunks(threads);
        for(int k=0;k<threads;k++)team.emplace_back([&,k]{
            while(!expired(until(.20))){
                int lo=next.fetch_add(32);if(lo>=n)break;
                for(int i=lo;i<std::min(n,lo+32);i++){
                    const auto &s=seeds[i];Arc a{{s.rows[0],s.rows[1],s.rows[2],s.rows[3],-1},s.g,s.score};
                    if(preliminary(*index,a))chunks[k].push_back(a);
                }
            }
        });
        for(auto &th:team)th.join();team.clear();
        stats[0]=std::min(next.load(),n);next.store(0);
        std::vector<Arc> all;
        for(auto &v:chunks)all.insert(all.end(),v.begin(),v.end());
        stats[1]=all.size();trim_arcs(all,24000);
        std::vector<std::vector<Certified>> found(threads);
        for(int k=0;k<threads;k++)team.emplace_back([&,k]{
            while(!expired(until(1.))){
                int i=next.fetch_add(1);if(i>=int(all.size()))break;
                auto &a=all[i];Certified c{};
                if(certify_one(*index,a.ids,4,a.g,c)){
                    c.merit+=.32*a.score;found[k].push_back(c);
                }
            }
        });
        for(auto &th:team)th.join();
        std::vector<Certified> result;for(auto &v:found)result.insert(result.end(),v.begin(),v.end());
        compact_certified(result,cap);stats[2]=std::min(int(all.size()),next.load());
        stats[3]=result.size();std::copy(result.begin(),result.end(),out);return int(result.size());
    }catch(...){return -1;}
}
