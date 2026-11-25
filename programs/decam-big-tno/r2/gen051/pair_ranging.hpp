// A measured angular displacement constrains the range on a bound orbit.
// Search a different visit along this pair's uncertainty ellipse, then
// condition a fourth-night prediction on all three positions.
namespace pair_ranging {
using namespace encounter;
using velocity_votes::Triangle;

struct Window {
    int e;
    V pos, dg, dh;
    double noise, xx, xy, yy, det, merit;
};

struct Observations {
    const Witness &sky;
    const double *depth, *color;
    std::vector<Field> images;
    Observations(const Witness &w,const double *d,const double *c,const float *quality):
        sky(w),depth(d),color(c),images(w.images.size()){
        std::vector<std::vector<Point>> points(images.size());
        for(int i=0;i<w.nt;i++){
            auto &p=w.objects[i];
            if(!std::isfinite(p.mag)||quality[i]<-30.)continue;
            points[p.exposure].push_back({p.x,p.y,p.mag,quality[i],0.f,i});
        }
        for(size_t e=0;e<images.size();e++){
            images[e].e=w.images[e].e;
            images[e].build(points[e],int(points[e].size()),false);
        }
    }
};

struct Sampler {
    const Observations &obs;
    const double *east;
    std::vector<Window> windows;
    std::vector<Triangle> triangles;
    struct Estimate {Triangle t;double g,h,gg,gh,hh;int night;};
    std::vector<Estimate> matches;
    std::vector<Arc> closed;
    long long tested=0,visits=0,examined=0;
    void close_pairs(double g,double sg){
        if(matches.empty())return;
        std::vector<double> bonus(matches.size());
        struct Joint {size_t i,j;double gamma,score;};
        std::vector<Joint> joints;
        for(size_t i=1;i<matches.size();i++)for(size_t j=0;j<i;j++){
            auto &a=matches[i],&b=matches[j];if(a.night==b.night)continue;
            // Overlap of two local range/range-rate posteriors. Both used
            // the same endpoints, so allow a floor for their correlated
            // angular-rate errors and the finite difference of the map.
            double xx=a.gg+b.gg+.00055,xy=a.gh+b.gh,yy=a.hh+b.hh+.0035;
            double dx=a.g-b.g,dy=a.h-b.h;
            if(dx*dx>9.*xx||dy*dy>9.*yy)continue;
            double det=xx*yy-xy*xy;if(!(det>1.e-12))continue;
            double ux=(yy*dx-xy*dy)/det,uy=(xx*dy-xy*dx)/det,chi=ux*dx+uy*dy;
            if(chi>8.)continue;
            double overlap=-.5*std::log(det)-.5*chi+.28*(a.g*b.g+a.h*b.h);
            if(overlap<=.3)continue;
            bonus[i]=std::max(bonus[i],overlap);
            bonus[j]=std::max(bonus[j],overlap);
            double gamma=g+sg*(a.g-a.gg*ux-a.gh*uy);
            double score=.50*(a.t.merit+b.t.merit)+overlap;
            if(gamma>1./150.&&gamma<1./16.)joints.push_back({i,j,gamma,score});
        }
        for(size_t i=0;i<matches.size();i++){
            auto tr=matches[i].t;tr.merit+=.85*std::min(7.,bonus[i]);
            triangles.push_back(tr);
        }
        size_t k=std::min<size_t>(18,joints.size());
        std::partial_sort(joints.begin(),joints.begin()+k,joints.end(),
                          [](auto &a,auto &b){return a.score>b.score;});
        for(size_t i=0;i<k;i++){
            auto &j=joints[i];auto &a=matches[j.i].t;auto &b=matches[j.j].t;
            Arc arc{{a.rows[0],a.rows[1],a.rows[2],b.rows[2],-1},float(j.gamma),float(j.score)};
            if(preliminary(obs.sky,arc))closed.push_back(arc);
        }
        if(closed.size()>60000)trim_arcs(closed,26000);
    }
    void visit_pair(int a,int b,double guess,double quality){
        const auto &w=obs.sky;
        const auto &l=w.objects[a],&r=w.objects[b];
        auto &s=obs.images[l.exposure].e,&t=obs.images[r.exposure].e;
        double dt=t.t-s.t;
        if(dt<.47||dt>135.)return;
        Point p{l.x,l.y,l.mag,0.f,0.f,a},q{r.x,r.y,r.mag,0.f,0.f,b};
        double g=guess,vc=std::sqrt(mu*g*g*g);
        double dn=std::sqrt(1.+sq(p.x)+sq(p.y));
        double h0=-vc*(p.x*east[0]+p.y*east[1])/dn;
        Transfer base(p,q,s,t,g,h0);
        double dg=.0008*g,dh=.001*vc;
        Transfer dist(p,q,s,t,g+dg,h0),rad(p,q,s,t,g,h0+dh);
        V dx=(dist.rate-base.rate)*(1./dg);
        double along=dx.x*east[0]+dx.y*east[1]-1.5*vc/g;
        double across=dx.y*east[0]-dx.x*east[1];
        double sg=1./std::sqrt(1./sq(.28*g)+sq(along/(.22*vc))+sq(across/(.34*vc)));
        double sh=.34*vc;
        double m=.5*(p.mag+q.mag),lever=.5*(obs.color[l.exposure]+obs.color[r.exposure]);
        double sa=w.sigma?std::clamp(double(w.sigma[a]),.07,.35):.19;
        double sb=w.sigma?std::clamp(double(w.sigma[b]),.07,.35):.19;
        windows.clear();int mb=std::clamp(int((m-10.)*4.),0,79);
        for(size_t j=0;j<obs.images.size();j++){
            auto &field=obs.images[j];auto &e=field.e;
            if(field.pts.empty()||e.night==s.night||e.night==t.night)continue;
            double ext=e.t-s.t;
            if(std::max(dt,ext)-std::min(0.,ext)<25.)continue;
            double reach=obs.depth[j]-m;
            if(reach<-.83)continue;
            double u=ext/dt;
            V center=base.at(e),X=(dist.at(e)-center)*(sg/dg),Y=(rad.at(e)-center)*(sh/dh);
            double phase=std::abs(ext*vc);
            double ast=sq(as)*(.16*.16+sq(sa*(1.-u))+sq(sb*u))+
                       sq(.034*std::pow(phase,4)+.45*sg*std::hypot(X.x,X.y));
            double xx=ast+sq(X.x)+sq(Y.x),yy=ast+sq(X.y)+sq(Y.y),xy=X.x*X.y+Y.x*Y.y;
            double det=ast*(ast+dot(X,X)+dot(Y,Y))+sq(X.x*Y.y-X.y*Y.x);
            double wx=2.8*std::sqrt(xx),wy=2.8*std::sqrt(yy);
            if(center.x+wx<field.sky.l||center.x-wx>field.sky.r||
               center.y+wy<field.sky.b||center.y-wy>field.sky.t)continue;
            double area=std::sqrt(det)/sq(as);
            double bg=area*w.images[j].density[mb];
            if(bg>65.||std::max(wx,wy)>.15)continue;
            // Discarding geometrically uninformative epochs would favour
            // long projections with many spurious third positions.
            double leverage=dot(X,X)/ast;
            double inside=(center.x>=field.sky.l&&center.x<=field.sky.r&&
                           center.y>=field.sky.b&&center.y<=field.sky.t)?1.:.3;
            double merit=std::log(inside/(1.+std::exp(-reach/.30)))-
                         .70*std::log1p(4.*bg)+.33*std::log1p(std::min(60.,leverage));
            windows.push_back({int(j),center,X,Y,ast,xx,xy,yy,det,merit});
        }
        if(windows.empty())return;
        std::sort(windows.begin(),windows.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        int seen[22],count=0;matches.clear();
        for(auto &v:windows){
            auto &field=obs.images[v.e];int nn=field.e.night;
            if(std::find(seen,seen+count,nn)!=seen+count)continue;
            // A large third-night prediction remains worth searching if the
            // best actual matches give a precise distance for a fourth.
            size_t at=matches.size();
            double ac=(v.yy*v.dg.x-v.xy*v.dg.y)/v.det,bc=(v.xx*v.dg.y-v.xy*v.dg.x)/v.det;
            double ah=(v.yy*v.dh.x-v.xy*v.dh.y)/v.det,bh=(v.xx*v.dh.y-v.xy*v.dh.x)/v.det;
            double gg=v.noise*(v.noise+dot(v.dh,v.dh))/v.det;
            double hh=v.noise*(v.noise+dot(v.dg,v.dg))/v.det;
            double gh=-v.noise*dot(v.dg,v.dh)/v.det;
            double sm=std::hypot(.29,.26*(obs.color[v.e]-lever));
            field.ellipse(v.pos,v.xx,v.xy,v.yy,v.det,2.9,[&](const Point &z,double x,double y,double chi){
                examined++;double dm=(z.mag-m)/sm;if(std::abs(dm)>3.2)return;
                double gamma=g+sg*(ac*x+bc*y),h=h0+sh*(ah*x+bh*y);
                if(!(gamma>1./150.&&gamma<1./16.))return;
                double bg=std::sqrt(v.det)/sq(as)*w.images[v.e].density[mb];
                double score=quality+.47*z.q-.48*(chi+dm*dm)-.72*std::log(.003+bg);
                Triangle tr{{a,b,z.row},float(gamma),float(h),float(score)};
                matches.push_back({tr,ac*x+bc*y,ah*x+bh*y,gg,gh,hh,nn});
            });
            if(matches.size()==at)continue;
            seen[count++]=nn;visits++;
            size_t keep=8;
            if(matches.size()>at+keep){
                auto begin=matches.begin()+at;
                std::nth_element(begin,begin+keep,matches.end(),
                                 [](auto &a,auto &b){return a.t.merit>b.t.merit;});
                matches.resize(at+keep);
            }
            if(count==18)break;
        }
        close_pairs(g,sg);
        if(triangles.size()>125000)velocity_votes::trim_triangles(triangles,56000);
        tested++;
    }
};
}

extern "C" int pair_ranging_search(const Witness *w,const int *pairs,const float *parameters,int n,
        const float *quality,const double *depth,const double *color,const double *east,
        int threads,double seconds,int cap,Certified *out,long long *stats){
    try{
        using namespace pair_ranging;
        auto start=Clock::now();
        auto deadline=[&](double fraction){return start+std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(seconds*fraction));};
        Observations sky(*w,depth,color,quality);
        std::vector<Sampler> jobs;for(int j=0;j<threads;j++)jobs.push_back({sky,east});
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(auto &job:jobs)workers.emplace_back([&,runner=&job]{
            while(!expired(deadline(.32))){
                int i=next.fetch_add(1);if(i>=n)break;
                runner->visit_pair(pairs[2*i],pairs[2*i+1],parameters[2*i],parameters[2*i+1]);
            }
        });
        for(auto &worker:workers)worker.join();workers.clear();
        std::vector<Triangle> all;
        for(auto &job:jobs){
            stats[0]+=job.tested;stats[1]+=job.visits;stats[2]+=job.examined;
            all.insert(all.end(),job.triangles.begin(),job.triangles.end());
        }
        velocity_votes::trim_triangles(all,110000);stats[3]=all.size();
        std::vector<std::vector<Arc>> arcs(threads);
        next.store(0);
        for(int j=0;j<threads;j++)workers.emplace_back([&,j]{
            while(!expired(deadline(.79))){
                int i=next.fetch_add(1);if(i>=int(all.size()))break;
                velocity_votes::fourth_night(*w,all[i],arcs[j]);
                if(arcs[j].size()>70000)trim_arcs(arcs[j],32000);
            }
        });
        for(auto &worker:workers)worker.join();workers.clear();
        stats[4]=std::min<size_t>(next,all.size());
        std::vector<Arc> candidates;
        for(auto &a:arcs)candidates.insert(candidates.end(),a.begin(),a.end());
        trim_arcs(candidates,50000);stats[5]=candidates.size();
        std::vector<Arc> direct;
        for(auto &j:jobs)direct.insert(direct.end(),j.closed.begin(),j.closed.end());
        // Budget refinement of both proposal routes before the common
        // exact fit. Each rank is local to its own search statistic.
        trim_arcs(direct,40000);
        if(!direct.empty()){
            std::vector<Arc> merged;merged.reserve(candidates.size()+direct.size());
            for(size_t i=0,j=0;i<candidates.size()||j<direct.size();){
                if(i<candidates.size())merged.push_back(candidates[i++]);
                if(i<candidates.size())merged.push_back(candidates[i++]);
                if(j<direct.size())merged.push_back(direct[j++]);
            }
            candidates.swap(merged);
        }
        std::vector<std::vector<Certified>> confirmed(threads);
        next.store(0);
        for(int j=0;j<threads;j++)workers.emplace_back([&,j]{
            while(!expired(deadline(1.))){
                int i=next.fetch_add(1);if(i>=int(candidates.size()))break;
                auto &a=candidates[i];Certified c{};
                if((a.ids[4]>=0&&certify_one(*w,a.ids,5,a.g,c))||
                   certify_one(*w,a.ids,4,a.g,c,false,true)){
                    c.merit+=.3*a.score;confirmed[j].push_back(c);
                }
            }
        });
        for(auto &worker:workers)worker.join();
        std::vector<Certified> five,four;
        for(auto &v:confirmed)for(auto &c:v)(c.nights>=5?five:four).push_back(c);
        stats[6]=five.size();stats[7]=four.size();
        compact_certified(five,cap);compact_certified(four,std::min<size_t>(cap-five.size(),cap/4));
        five.insert(five.end(),four.begin(),four.end());std::copy(five.begin(),five.end(),out);
        return five.size();
    }catch(...){return -1;}
}
