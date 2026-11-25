// An angular rate measured over nearby nights fixes a narrow distance
// interval. Test the short arc against *all* nearby covered visits before
// allocating a several-year range search to it.
namespace local_motion {
using cadence_ranging::Sky;
using encounter::Field;
using encounter::Point;
using encounter::V;

struct Plan {
    int a,b;
    std::vector<int> nearby;
    double dt, overlap;
};

static double gap(const encounter::Box &a,const encounter::Box &b) {
    return std::hypot(std::max({0.,a.l-b.r,b.l-a.r}),std::max({0.,a.b-b.t,b.b-a.t}));
}

static std::vector<Plan> visit_pairs(const Sky &s,double maxgap) {
    std::vector<Plan> out;
    for (int a=0;a<int(s.fields.size());++a) {
        const auto &f=s.fields[a];
        if (f.pts.empty()) continue;
        for (int b=0;b<int(s.fields.size());++b) {
            const auto &v=s.fields[b];
            double dt=v.e.t-f.e.t;
            if (dt<.52 || dt>maxgap || f.e.night==v.e.night || v.pts.empty()) continue;
            if (gap(f.sky,v.sky)>.0012*dt) continue;
            Plan plan{a,b,{},dt,0.};
            for (int j=0;j<int(s.fields.size());++j) {
                if (j==a || j==b || s.fields[j].pts.empty()) continue;
                const auto &im=s.fields[j];
                double age=im.e.t-f.e.t;
                // With only two positions, centroid errors alone grow
                // linearly in the time baseline of an extrapolation.
                if (std::abs(age-.5*dt)>std::min(105.,30.+12.*dt)) continue;
                if (gap(f.sky,im.sky)>.0012*std::abs(age)+.0002) continue;
                plan.nearby.push_back(j);
            }
            if (plan.nearby.empty()) continue;
            plan.overlap=std::max(.03,f.sky.meet(v.sky).area()/std::max(1.e-8,f.sky.area()));
            out.push_back(std::move(plan));
        }
    }
    return out;
}

struct Support {int row,night; double gamma,merit;};
struct LocalFit {
    const Sky &sky;
    std::vector<ArcSeed> seeds;
    std::vector<Support> matches;
    long long pairs=0,views=0,triples=0;
    int maxout;
    Clock::time_point end;
    static bool best(const ArcSeed &a,const ArcSeed &b) {return a.score>b.score;}
    void trim(int limit) {
        if (int(seeds.size())>limit) {
            std::nth_element(seeds.begin(),seeds.begin()+limit,seeds.end(),best);
            seeds.resize(limit);
        }
    }
    void measure(const Plan &plan,const Point &p,const Point &q,double g,double sg,
                 double pairscore) {
        const auto &a=sky.fields[plan.a].e,&b=sky.fields[plan.b].e;
        const double dt=plan.dt;
        // Barycentric transverse positions for a constant line-of-sight
        // range. Over this local span the curvature terms can be expanded
        // about the first emission time. Remote recovery uses Kepler motion.
        V u{p.x,p.y},v{q.x,q.y};
        V A{a.ox-p.x*a.oz,a.oy-p.y*a.oz},B{b.ox-q.x*b.oz,b.oy-q.y*b.oz};
        double sa=sky.sigma(p.row),sb=sky.sigma(q.row);
        double vc=std::sqrt(mu*g*g*g);
        double h=-dot(u,{sky.east[0],sky.east[1]})*vc;
        double mag=.5*(p.mag+q.mag),lev=.5*(sky.color[plan.a]+sky.color[plan.b]);
        int mb=std::clamp(int((mag-10.)*4.),1,78);
        matches.clear();
        double misses=0.;
        ++pairs;
        for (int k:plan.nearby) {
            const Field &im=sky.fields[k]; const auto &c=im.e;
            double age=c.t-a.t,t=age/dt;
            V interp=u+(v-u)*t;
            V obs=A+(B-A)*t;
            V acc=obs-V{c.ox-interp.x*c.oz,c.oy-interp.y*c.oz};
            double z=1.-g*c.oz;
            V dg=acc*(1./z);
            V dh=(v-interp)*(age/z);
            V center=interp+dg*g+dh*h;
            double noise=sq(.23*as)+sq((1.-t)*sa)+sq(t*sb);
            // Solar acceleration is almost along the sight line. The
            // remaining parallax curvature is small but grows with span.
            noise+=sq(.06*as+1.1*vc*vc*std::abs(age*(age-dt))*g);
            V rg=dg*sg,rh=dh*(.45*vc);
            double xx=noise+sq(rg.x)+sq(rh.x),yy=noise+sq(rg.y)+sq(rh.y);
            double xy=rg.x*rg.y+rh.x*rh.y,det=noise*(xx+yy-noise)+sq(rg.x*rh.y-rg.y*rh.x);
            if (center.x<im.sky.l-3.6*std::sqrt(xx)||center.x>im.sky.r+3.6*std::sqrt(xx)||
                center.y<im.sky.b-3.6*std::sqrt(yy)||center.y>im.sky.t+3.6*std::sqrt(yy)) continue;
            double area=std::sqrt(det)/sq(as);
            double bg=std::max(.000001,double(sky.w.images[k].density[mb]))*area;
            if (bg>5.0) continue;
            double sm=std::hypot(.28,.27*(sky.color[k]-lev));
            double top=-1.e20,second=top,gamma=g;
            int row=-1;
            ++views;
            im.ellipse(center,xx,xy,yy,det,3.65,[&](const Point &r,double x,double y,double chi) {
                double dm=(r.mag-mag)/sm;
                if (std::abs(dm)>3.6) return;
                // A same-night companion must distinguish motion from
                // the same stationary peak recurring in a nearby exposure.
                double contrast=0.;
                if (c.night==a.night || c.night==b.night) {
                    const Point &base=c.night==a.night?p:q;
                    V motion=center-V{base.x,base.y};
                    V shift{r.x-base.x,r.y-base.y};
                    double var=sq(.19*as)+sq(sky.sigma(base.row))+sq(sky.sigma(r.row));
                    contrast=(2.*dot(motion,shift)-dot(motion,motion))/(2.*var);
                    if (contrast<.65) return;
                }
                double merit=.46*std::clamp(double(r.q),-6.,4.)-.5*(chi+dm*dm)
                             -std::log(.0001+6.28318*bg*sm)+.18*std::min(5.,contrast);
                if (merit>top) {
                    second=top;top=merit; row=r.row;
                    gamma=g+sg*((yy*x-xy*y)*rg.x+(xx*y-xy*x)*rg.y)/det;
                } else second=std::max(second,merit);
            });
            if (row>=0 && top>-.8 && gamma>1./11000. && gamma<1./12.) {
                double uniqueness=.45*std::log1p(std::exp(-std::min(30.,top-second)));
                matches.push_back({row,c.night,gamma,top-uniqueness});
            } else if (t>-.25 && t<1.25 && im.coverage(center)>.70 && bg<.12) {
                double d=sky.depth[k]-mag;
                if (d>.05) misses+=std::min(1.,d);
            }
        }
        if (matches.empty()) return;
        std::sort(matches.begin(),matches.end(),[](auto x,auto y){return x.merit>y.merit;});
        int seen[3],nseen=0;
        for (const auto &r:matches) {
            if (std::find(seen,seen+nseen,r.night)!=seen+nseen) continue;
            seen[nseen++]=r.night;
            ArcSeed seed{{p.row,q.row,r.row,-1},float(r.gamma),
                         float(pairscore+r.merit-.65*std::min(4.,misses))};
            auto companion=std::find_if(matches.begin(),matches.end(),[&](const Support &o){
                return o.night!=r.night && o.row!=r.row && o.merit>1. &&
                       std::abs(o.gamma-r.gamma)<1.5*sg;
            });
            if (companion!=matches.end()) {
                ArcSeed quad=seed;quad.rows[3]=companion->row;
                quad.score+=float(.9*std::min(11.,companion->merit));
                seeds.push_back(quad);
            }
            seeds.push_back(seed);++triples;
            if (nseen==3) break;
        }
        if (int(seeds.size())>4*maxout) trim(2*maxout);
    }

    void row(const Plan &plan,const Point &p,double distance,double relrange,double speedwidth) {
        const Field &f=sky.fields[plan.a],&v=sky.fields[plan.b];
        const double g0=1./distance,vc=std::sqrt(mu*g0*g0*g0),dt=plan.dt;
        V pos{p.x,p.y},east{sky.east[0],sky.east[1]};
        V observer{v.e.ox-f.e.ox-p.x*(v.e.oz-f.e.oz),v.e.oy-f.e.oy-p.y*(v.e.oz-f.e.oz)};
        double z=1.-g0*v.e.oz;
        V vel=east*(vc*dt),center=pos+(vel-observer*g0)*(1./z);
        V grad=(observer*(-1.)+vel*(1.5/g0))*(1./z);
        V width=grad*(g0*relrange);
        double noise=sq(speedwidth*vc*dt/z)+sq(.30*as);
        double xx=noise+sq(width.x),xy=width.x*width.y,yy=noise+sq(width.y);
        double det=noise*(noise+dot(width,width));
        int mb=std::clamp(int((p.mag-10.)*4.),1,78);
        double chance=std::sqrt(det)/sq(as)*sky.w.images[plan.b].density[mb];
        v.ellipse(center,xx,xy,yy,det,3.4,[&](const Point &q,double x,double y,double chi){
            double sm=std::hypot(.33,.29*(sky.color[plan.a]-sky.color[plan.b]));
            double dm=(p.mag-q.mag)/sm;
            if (std::abs(dm)>3.3 || chi+.6*dm*dm>12.) return;
            // Re-evaluate the conditional distance as the circular speed
            // and topocentric projection change with range.
            double g=g0,sg=g0*relrange;
            for (int it=0;it<2;++it) {
                double speed=std::sqrt(mu*g*g*g);
                V obs{v.e.ox-f.e.ox-q.x*v.e.oz+p.x*f.e.oz,
                      v.e.oy-f.e.oy-q.y*v.e.oz+p.y*f.e.oz};
                V residual=V{q.x-p.x,q.y-p.y}+obs*g-east*(speed*dt);
                V k=obs-east*(1.5*speed*dt/g);
                double var=sq(speedwidth*speed*dt)+sq(.28*as);
                double prior=1./sq(g0*relrange),precision=prior+dot(k,k)/var;
                g-=std::clamp((dot(k,residual)/var+prior*(g-g0))/precision,-.28*g,.28*g);
                sg=1./std::sqrt(precision);
            }
            if (g<1./11000. || g>1./12.) return;
            double merit=.65*(std::clamp(double(p.q),-6.,5.)+std::clamp(double(q.q),-6.,5.))
                         -.40*(chi+dm*dm)-std::log(.004+chance*sm);
            measure(plan,p,q,g,sg,merit);
        });
    }
};
}

extern "C" int local_motion_seeds(const Witness *w,const float *q,const double *depth,
        const double *lever,const double *east,double distance,double range,double velocity,
        double maxgap,int per_image,int threads,double seconds,int cap,ArcSeed *out,long long *counts) {
    using namespace local_motion;
    try {
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Sky sky(*w,lever,depth,east,q,per_image);
        auto plans=visit_pairs(sky,maxgap);
        struct Task {int plan,offset;float priority;};
        std::vector<Task> tasks;
        constexpr int chunk=96;
        for (int k=0;k<int(plans.size());++k) {
            auto &v=plans[k];const auto &pts=sky.fields[v.a].pts;
            for (int i=0;i<int(pts.size());i+=chunk) {
                // Interleave sky areas and epochs when the deadline allows
                // only a fraction of the wide or faint doublet catalogue.
                auto h=pair_closure::mix(unsigned(i/chunk)*2246822519u+unsigned(k)*3266489917u);
                double val=.38*std::log(v.overlap)+.18*std::log1p(v.nearby.size())
                          -.32*std::log(v.dt)+3.5*h/4294967296.;
                tasks.push_back({k,i,float(val)});
            }
        }
        std::sort(tasks.begin(),tasks.end(),[](auto &a,auto &b){return a.priority>b.priority;});
        std::vector<LocalFit> jobs;
        for (int j=0;j<threads;++j) jobs.push_back({sky,{},{},0,0,0,std::max(2000,cap/threads),end});
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for (int j=0;j<threads;++j) pool.emplace_back([&,j]{
            while (!expired(end)) {
                int k=next.fetch_add(1);if (k>=int(tasks.size())) break;
                const auto &t=tasks[k];const auto &v=plans[t.plan];const auto &pts=sky.fields[v.a].pts;
                for (int i=t.offset;i<std::min(t.offset+chunk,int(pts.size()));++i) {
                    if ((i%8)==0 && expired(end)) break;
                    jobs[j].row(v,pts[i],distance,range,velocity);
                }
            }
        });
        for (auto &t:pool)t.join();
        std::vector<ArcSeed> seeds;
        for (auto &j:jobs) {
            j.trim(2*j.maxout);
            seeds.insert(seeds.end(),j.seeds.begin(),j.seeds.end());
            counts[0]+=j.pairs;counts[1]+=j.views;counts[2]+=j.triples;
        }
        std::sort(seeds.begin(),seeds.end(),LocalFit::best);
        std::set<std::array<int,4>> unique;
        int n=0;
        for (auto &s:seeds) {
            std::array<int,4> key;std::copy_n(s.rows,4,key.begin());std::sort(key.begin(),key.end());
            if (!unique.insert(key).second) continue;
            out[n++]=s;if (n==cap) break;
        }
        counts[3]=tasks.size();counts[4]=std::min(size_t(next.load()),tasks.size());
        return n;
    } catch (...) {return -1;}
}
