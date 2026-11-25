// Complete a two-night arc in its own apparition, including measurements
// that have no independently high-scoring partner of their own.
namespace local_companions {
using namespace cadence_ranging;
struct Hit {Forecast view;V delta;int row;double score,g;};
struct Seed {int rows[4];float g,score;};
struct Worker {
    const Sky &sky;const std::vector<std::vector<int>> &schedule;
    std::vector<Arc> triples,quads;
    long long examined=0,hits=0;
    void extend(const companion_orbits::Seed &pair){
        int i=pair.row[0],j=pair.row[1],a=sky.w.objects[i].exposure,b=sky.w.objects[j].exposure;
        if(!(pair.g>1./9000.&&pair.g<1./16.))return;
        PairOrbit p(sky,i,j,pair.g);
        if(std::abs(p.dt)<.45)return;
        double mag=.5*(p.a.mag+p.b.mag),lever=.5*(sky.color[a]+sky.color[b]);
        int mb=std::clamp(int((mag-10.)*4.),1,78),opportunities=0;
        std::vector<Hit> candidates;
        examined++;
        for(int e:schedule[a]){
            auto &field=sky.fields[e];auto &t=field.e;
            if(t.night==p.left.night||t.night==p.right.night||sky.depth[e]+.8<mag)continue;
            auto v=p.at(e,t,sky.w.images[e].density[mb]);
            const auto &cov=v.covariance;double det=cov.det();
            if(!(det>0.))continue;
            double wx=3.1*std::sqrt(cov.xx),wy=3.1*std::sqrt(cov.yy);
            if(v.center.x+wx<field.sky.l||v.center.x-wx>field.sky.r||
               v.center.y+wy<field.sky.b||v.center.y-wy>field.sky.t)continue;
            double area=std::sqrt(det)/sq(as),clutter=std::max(.00005,area*v.background);
            if(clutter>2.4||area>15000.)continue;
            opportunities++;
            double ms=std::hypot(.30,.27*(sky.color[e]-lever));
            size_t first=candidates.size();
            field.ellipse(v.center,cov.xx,cov.xy,cov.yy,det,3.1,
                [&](const Point &z,double dx,double dy,double chi){
                    double dm=(z.mag-mag)/ms;if(std::abs(dm)>3.)return;
                    V delta{dx,dy};V wt=cov.solve(delta);
                    double g=p.g+p.sg*dot(v.range,wt);if(!(g>1./9500.&&g<1./15.))return;
                    double gain=1./std::max(.004,1.-dot(v.range,cov.solve(v.range)));
                    double score=-.5*(chi+sq(dm))+.30*std::clamp(double(z.q),-7.,7.)-
                         std::log(6.283185*clutter*ms)+.35*std::log(gain);
                    candidates.push_back({v,delta,z.row,score,g});
                });
            if(candidates.size()>first+3){
                auto begin=candidates.begin()+first;
                std::partial_sort(begin,begin+3,candidates.end(),[](auto &x,auto &y){return x.score>y.score;});
                candidates.resize(first+3);
            }
        }
        hits+=candidates.size();
        size_t count=std::min<size_t>(16,candidates.size());
        std::partial_sort(candidates.begin(),candidates.begin()+count,candidates.end(),
                          [](auto &x,auto &y){return x.score>y.score;});
        double initial=.71*pair.score-std::log1p(.25*opportunities);
        for(size_t k=0;k<count;k++){
            auto &t=candidates[k];
            if(k<5)triples.push_back({{i,j,t.row,-1,-1},float(t.g),float(initial+t.score)});
            for(size_t l=0;l<k;l++){
                auto &u=candidates[l];
                if(sky.fields[t.view.visit].e.night==sky.fields[u.view.visit].e.night)continue;
                auto conditional=given(t.view,u.view,u.delta);
                auto &c=conditional.cov;double det=c.det();if(!(det>0.))continue;
                V offset=t.view.center+t.delta-conditional.center;
                double chi=dot(offset,c.solve(offset));if(chi>12.)continue;
                double area=std::sqrt(det)/sq(as);
                double gain=-std::log(.00015+6.283185*t.view.background*area);
                Arc arc{{i,j,u.row,t.row,-1},float(u.g),float(initial+u.score+.34*t.score+gain-.45*chi)};
                if(encounter::preliminary(sky.w,arc))quads.push_back(arc);
            }
        }
        if(triples.size()>65000)encounter::trim_arcs(triples,33000);
        if(quads.size()>47000)encounter::trim_arcs(quads,25000);
    }
};
}
extern "C" int local_companion_arcs(const Witness *w,const companion_orbits::Seed *pairs,int n,
    const float *marks,const double *depth,const double *lever,const double *east,int threads,
    double seconds,int cap,local_companions::Seed *out,long long *stats){
    using namespace local_companions;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Sky sky(*w,lever,depth,east,marks);int ne=w->images.size();
        std::vector<std::vector<int>> schedule(ne);
        for(int a=0;a<ne;a++)for(int e=0;e<ne;e++){
            const auto &u=sky.fields[a],&v=sky.fields[e];
            if(u.pts.empty()||v.pts.empty()||u.e.night==v.e.night||std::abs(v.e.t-u.e.t)>185.)continue;
            double pad=.075+.00032*std::abs(v.e.t-u.e.t);
            if(u.sky.l>v.sky.r+pad||u.sky.r<v.sky.l-pad||u.sky.b>v.sky.t+pad||u.sky.t<v.sky.b-pad)continue;
            schedule[a].push_back(e);
        }
        std::atomic<int> next{0};std::vector<Worker> jobs;std::vector<std::thread> team;
        for(int j=0;j<threads;j++)jobs.push_back({sky,schedule});
        for(auto &job:jobs)team.emplace_back([&,s=&job]{
            while(!expired(end)){int k=next.fetch_add(1);if(k>=n)break;s->extend(pairs[k]);}
        });
        for(auto &j:team)j.join();
        std::vector<Arc> three,four;
        for(auto &s:jobs){
            stats[0]+=s.examined;stats[1]+=s.hits;
            three.insert(three.end(),s.triples.begin(),s.triples.end());
            four.insert(four.end(),s.quads.begin(),s.quads.end());
        }
        encounter::trim_arcs(four,cap/2);encounter::trim_arcs(three,cap-four.size());
        int total=0;stats[2]=three.size();stats[3]=four.size();
        for(size_t k=0;k<std::max(three.size(),four.size());k++){
            for(auto *v:{&four,&three})if(k<v->size()){
                auto &src=(*v)[k];auto &s=out[total++];std::copy_n(src.ids,4,s.rows);s.g=src.g;s.score=src.score;
            }
        }
        return total;
    }catch(...){return -1;}
}

extern "C" int companion_raw_cores(const Witness *w,const local_companions::Seed *seeds,int n,
                                   original::Core *out){
    int used=0;
    for(int i=0;i<n;i++){
        auto &s=seeds[i];if(s.rows[3]<0)continue;
        encounter::Arc a{{s.rows[0],s.rows[1],s.rows[2],s.rows[3],-1},s.g,0.f};
        ExactState fit;if(!encounter::preliminary(*w,a,&fit))continue;
        auto &c=out[used++];c.count=4;c.ref=fit.ref;
        double state[]={fit.x,fit.y,fit.g,fit.vx,fit.vy,fit.h};std::copy_n(state,6,c.x);
        c.merit=std::clamp(.16*(s.score-9.),-2.,5.);std::copy_n(s.rows,4,c.rows);
    }
    return used;
}
