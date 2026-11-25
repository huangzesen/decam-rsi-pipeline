#include <numeric>
namespace visit_triplets {
using encounter::Point;
using encounter::Field;
using encounter::Transfer;
using encounter::V;
using cadence_ranging::Forecast;
using cadence_ranging::Ellipse;
struct Member {int row,peer;float merit,single;};
struct Pair {int a,b;float merit,g;};
struct Proposal {int row[4];float g,h,merit;int witness=-1;};

struct Catalogue {
    const original::Sky &raw;
    const double *depth;
    V east;
    std::vector<Field> fields;
    std::vector<std::array<float,48>> rho;
    Catalogue(const original::Sky &r,const double *d,const double *direction,const Member *m,int n)
        :raw(r),depth(d),east{direction[0],direction[1]},fields(raw.images.size()),rho(raw.images.size()){
        std::vector<std::vector<Point>> lists(fields.size());
        for(int j=0;j<n;j++){
            int i=m[j].row;
            if(i<0||i>=raw.size||!std::isfinite(raw.mag[i]))continue;
            int e=raw.ex[i];
            lists[e].push_back({raw.xy[2*i],raw.xy[2*i+1],raw.mag[i],m[j].merit,m[j].single,i});
            rho[e][mb(raw.mag[i])]+=1.f;
        }
        for(int e=0;e<int(fields.size());e++){
            fields[e].e=raw.images[e].exposure;
            fields[e].build(lists[e],int(lists[e].size()),false);
            auto h=rho[e];
            for(int b=0;b<48;b++)
                rho[e][b]=(h[std::max(0,b-1)]+2.*h[b]+h[std::min(47,b+1)]+1.)/
                          (2.*raw.images[e].area);
        }
    }
    static int mb(double m){return std::clamp(int((m-11.)*2.),0,47);}
    Point point(int row)const{return {raw.xy[2*row],raw.xy[2*row+1],raw.mag[row],0.f,0.f,row};}
    double sigma(int row)const{return as*std::hypot(.096,1.10/std::max(4.5f,raw.sn[row]));}
    double varmag(int a,int b)const{
        return sq(.165)+sq(1.086/raw.sn[a])+sq(1.086/raw.sn[b])+sq(.29*(raw.lever[raw.ex[a]]-raw.lever[raw.ex[b]]));
    }
};

struct PairGate {
    V center,dg;
    double g,variance,xx,xy,yy,det,span,detect,rate;
    PairGate(const Catalogue &s,int row,int e,double radius,double range,
             double along,double cross){
        auto a=s.point(row);const auto &one=s.fields[s.raw.ex[row]].e,&two=s.fields[e].e;
        double dt=two.t-one.t,norm=std::sqrt(1.+a.x*a.x+a.y*a.y),vc=std::sqrt(mu/std::pow(radius,3))*norm;
        g=norm/radius;
        V xy0{a.x,a.y},obs{two.ox-one.ox-a.x*(two.oz-one.oz),two.oy-one.oy-a.y*(two.oz-one.oz)};
        double scale=1./(1.-g*two.oz);
        V velocity=s.east*(dt*vc);
        center=xy0+(velocity-obs*g)*scale;
        dg=(velocity*1.5-obs*g+(center-xy0)*(g*two.oz))*(scale*range);
        double along2=sq(along*vc*dt*scale),cross2=sq(cross*vc*dt*scale);
        double floor=sq(.27*as)+sq(.40*vc*std::abs(dt)*(g+std::abs(dt)*vc));
        xx=dg.x*dg.x+floor+along2*sq(s.east.x)+cross2*sq(s.east.y);
        xy=dg.x*dg.y+(along2-cross2)*s.east.x*s.east.y;
        yy=dg.y*dg.y+floor+along2*sq(s.east.y)+cross2*sq(s.east.x);
        det=xx*yy-xy*xy;span=std::abs(dt);variance=range;
        detect=1./(1.+std::exp(std::clamp((a.mag-s.depth[e]-.1)/.31,-25.,25.)));
        rate=s.rho[e][Catalogue::mb(a.mag)];
    }
    double distance(V offset)const{
        return g*(1.+variance*(dg.x*(yy*offset.x-xy*offset.y)+dg.y*(xx*offset.y-xy*offset.x))/det);
    }
};
template<class T> void best(std::vector<T> &group,size_t n){
    if(group.size()>n){
        std::nth_element(group.begin(),group.begin()+n,group.end(),
                         [](const T&a,const T&b){return a.merit>b.merit;});
        group.resize(n);
    }
}
struct TileMatch {int e;PairGate gate;double merit;};
struct PairWorker {
    const Catalogue &cat;
    std::vector<TileMatch> near,far;
    std::vector<Pair> pool;
    long long seen=0;
    void anchor(const Member &m,const double *family,std::vector<Pair> &results) {
        int a=m.row,from=cat.raw.ex[a];
        const auto &origin=cat.fields[from].e;
        double mag=cat.raw.mag[a];
        if(!std::isfinite(mag))return;
        far.clear();near.clear();
        for(int e=0;e<int(cat.fields.size());e++){
            const auto &f=cat.fields[e];double dt=std::abs(f.e.t-origin.t);
            if(f.pts.empty()||origin.night==f.e.night||dt<.44||dt>family[4]||cat.depth[e]+.83<mag)continue;
            PairGate gate(cat,a,e,family[0],family[1],family[2],family[3]);
            constexpr double width=2.8;
            double wx=width*std::sqrt(gate.xx),wy=width*std::sqrt(gate.yy);
            V c=gate.center;
            if(c.x+wx<f.sky.l||c.x-wx>f.sky.r||c.y+wy<f.sky.b||c.y-wy>f.sky.t)continue;
            double area=std::sqrt(gate.det)/sq(as);
            double prior=std::log(.013+gate.detect)-.44*std::log(.05+area*gate.rate);
            (dt>142.?far:near).push_back({e,gate,prior});
        }
        for(auto *views:{&near,&far}){
            std::sort(views->begin(),views->end(),[](auto&a,auto&b){return a.merit>b.merit;});
            int nights[20],nn=0;
            pool.clear();
            for(auto &v:*views){
                const auto &f=cat.fields[v.e];auto &g=v.gate;
                bool used=std::find(nights,nights+nn,f.e.night)!=nights+nn;
                if(used)continue;
                size_t count=pool.size();
                double radius=std::clamp(std::sqrt(170./(.01+g.rate*std::sqrt(g.det)/sq(as))),.38,2.8);
                f.ellipse(g.center,g.xx,g.xy,g.yy,g.det,radius,[&](const Point &p,double dx,double dy,double chi){
                    ++seen;
                    double dm=p.mag-mag,var=cat.varmag(a,p.row);
                    if(dm*dm>10.*var)return;
                    double gamma=g.distance({dx,dy});
                    if(!(gamma>1./6000.&&gamma<1./16.))return;
                    double q=.56*std::clamp(double(p.q),-8.,15.)-.46*chi-.48*dm*dm/var+v.merit;
                    pool.push_back({a,p.row,float(q),float(gamma)});
                });
                if(pool.size()>count){
                    nights[nn++]=f.e.night;
                    // Keep an exposure in a crowded season from filling all
                    // the partner slots of an otherwise isolated measurement.
                    if(pool.size()>count+5){
                        std::partial_sort(pool.begin()+count,pool.begin()+count+5,pool.end(),
                                          [](auto&a,auto&b){return a.merit>b.merit;});
                        pool.resize(count+5);
                    }
                }
                if(nn==12)break;
            }
            best(pool,views==&near?8:7);
            for(auto p:pool){
                p.merit+=.54f*std::clamp(m.merit,-8.f,15.f);
                if(p.a>p.b)std::swap(p.a,p.b);
                results.push_back(p);
            }
        }
    }
};

// Condition the family of orbits through two endpoints on an actual third
// position. Recompute the transfer about the posterior range at each step.
// This avoids projecting a long, curved multi-year admissible region using
// only its tangent at the population mean.
struct Model {
    const Catalogue &cat;
    Point a,b;
    const Exposure &ea,&eb;
    double g0,h0,sg,sh,g,h,sa,sb;
    Transfer mean,distance,radial;
    Model(const Catalogue &s,const Pair &p):cat(s),a(s.point(p.a)),b(s.point(p.b)),
        ea(s.fields[s.raw.ex[p.a]].e),eb(s.fields[s.raw.ex[p.b]].e),g0(p.g),
        h0(-std::sqrt(mu*g0*g0*g0)*(s.east.x*a.x+s.east.y*a.y)),
        sg(g0*.25),sh(.34*std::sqrt(mu*g0*g0*g0)),g(g0),h(h0),sa(s.sigma(p.a)),sb(s.sigma(p.b)),
        mean(a,b,ea,eb,g,h),distance(a,b,ea,eb,g+g0*.001,h),radial(a,b,ea,eb,g,h+sh*.01){
        const double initial=g0;
        double ast=(sa*sa+sb*sb)/sq(eb.t-ea.t);
        for(int k=0;k<3;k++){
            double vc=std::sqrt(mu*g*g*g)/std::sqrt(std::sqrt(1.+dot(mean.pos,mean.pos)));
            V deriv=(distance.rate-mean.rate)*(1./(.001*g0));
            V relative=mean.rate-mean.pos*h;
            double vx=dot(relative,s.east)-vc,vy=relative.y*s.east.x-relative.x*s.east.y;
            double du=dot(deriv,s.east)-1.5*vc/g,dv=deriv.y*s.east.x-deriv.x*s.east.y;
            double wu=1./(sq(.205*vc)+ast),wv=1./(sq(.33*vc)+ast),prior=1./sq(.28*initial);
            double update=-(vx*du*wu+vy*dv*wv+(g-initial)*prior)/(du*du*wu+dv*dv*wv+prior);
            g+=std::clamp(update,-.23*g,.23*g);
            g=std::clamp(g,1./7000.,1./16.);
            h=-std::sqrt(mu*g*g*g)*dot(s.east,V{a.x,a.y});
            g0=g;h0=h;sh=.34*std::sqrt(mu*g*g*g);set(0.,0.);
            if(std::abs(update)<.0001*g)break;
        }
        V deriv=(distance.rate-mean.rate)*(1./(.001*g0));
        double vc=std::sqrt(mu*g0*g0*g0);
        double along=dot(deriv,s.east)-1.5*vc/g0,cross=deriv.y*s.east.x-deriv.x*s.east.y;
        sg=1./std::sqrt(1./sq(.27*g0)+sq(along)/(sq(.25*vc)+ast)+sq(cross)/(sq(.38*vc)+ast));
    }
    void set(double u,double v){
        g=g0+u*sg;h=h0+v*sh;
        mean=Transfer(a,b,ea,eb,g,h);
        distance=Transfer(a,b,ea,eb,g+g0*.001,h);
        radial=Transfer(a,b,ea,eb,g,h+sh*.01);
    }
    Forecast predict(int j)const{
        const auto &e=cat.fields[j].e;
        V c=mean.at(e),x=(distance.at(e)-c)*(sg/(.001*g0)),y=(radial.at(e)-c)*100.;
        double dt=e.t-ea.t,df=eb.t-ea.t,phase=std::max(std::abs(dt),std::abs(dt-df))*std::sqrt(mu*g*g*g);
        double f1,s1,f2,s2;mean.coeff(dt,f1,s1);mean.coeff(df,f2,s2);
        double z=f1+s1*h-g*e.oz;
        double l=(f1-s1*f2/s2)*(1.-g*ea.oz)/z*sa, r=s1/s2*(f2+s2*h-g*eb.oz)/z*sb;
        double base=sq(.23*as)+sq(.009*std::pow(phase,5));
        double v=base+l*l+r*r;
        Ellipse cov{v+x.x*x.x+y.x*y.x,x.x*x.y+y.x*y.y,v+x.y*x.y+y.y*y.y};
        return {j,c,x,y,l,r,base,0.,cov,0.};
    }
    bool condition(int j,const Point &p,double &u,double &v,Forecast &post){
        u=v=0.;set(0.,0.);
        for(int it=0;it<3;it++){
            auto f=predict(j);V residual=V{p.x,p.y}-f.center+f.range*u+f.radial*v;
            if(!(f.covariance.det()>0.))return false;
            V weight=f.covariance.solve(residual);
            double nu=dot(f.range,weight),nv=dot(f.radial,weight);
            if(std::abs(nu)>3.8||std::abs(nv)>3.8||!(g0+sg*nu>1./8000.&&g0+sg*nu<1./15.))return false;
            double gap=std::abs(nu-u)+std::abs(nv-v);
            u=nu;v=nv;set(u,v);if(gap<.002)break;
        }
        post=predict(j);
        V residual=V{p.x,p.y}-post.center;
        return dot(residual,residual)<15.*(post.noise+sq(post.left)+sq(post.right));
    }
};
struct Third {Point p;int e;float merit;};
struct Extend {
    const Catalogue &strong,&full;
    std::vector<Forecast> views;
    std::vector<Third> thirds;
    std::vector<Proposal> here;
    long long totals[5]={};
    void pair(const Pair &pair,std::vector<Proposal> &out,int witness=-1,float witness_merit=0.){
        if(!(pair.g>1./6000.&&pair.g<1./16.)||pair.a==pair.b)return;
        Model model(strong,pair);
        if(model.ea.night==model.eb.night||std::abs(model.eb.t-model.ea.t)<.44)return;
        double mag=.5*(model.a.mag+model.b.mag);
        if(!std::isfinite(mag))return;
        int mb=Catalogue::mb(mag);
        double color=.5*(strong.raw.lever[strong.raw.ex[pair.a]]+strong.raw.lever[strong.raw.ex[pair.b]]);
        views.clear();thirds.clear();here.clear();++totals[0];
        for(int e=0;e<int(full.fields.size());e++){
            const auto &f=full.fields[e];const auto &z=f.e;
            if(f.pts.empty()||z.night==model.ea.night||z.night==model.eb.night||full.depth[e]+.84<mag)continue;
            V pos=model.mean.at(z);
            double t=z.t-model.ea.t,frac=t/(model.eb.t-model.ea.t);
            // First discard entire images. The exact visit-specific ellipse
            // follows for every image that the family can plausibly reach.
            double pad=.035+std::min(.15,std::abs(t)*.000035)+.000075*std::abs(frac);
            if(pos.x+pad<f.sky.l||pos.x-pad>f.sky.r||pos.y+pad<f.sky.b||pos.y-pad>f.sky.t)continue;
            auto v=model.predict(e);double wx=3.4*std::sqrt(v.covariance.xx),wy=3.4*std::sqrt(v.covariance.yy);
            if(pos.x+wx<f.sky.l||pos.x-wx>f.sky.r||pos.y+wy<f.sky.b||pos.y-wy>f.sky.t)continue;
            if(!(v.covariance.det()>0.))continue;
            v.background=std::max(1.e-8f,strong.rho[e][mb]);
            double clutter=std::sqrt(v.covariance.det())/sq(as)*v.background;
            v.merit=std::log(.04+encounter::Bank::detect(full.depth[e],mag))-.50*std::log(.025+clutter);
            views.push_back(v);
        }
        if(views.size()<2)return;
        std::sort(views.begin(),views.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        if(witness>=0)thirds.push_back({full.point(witness),full.raw.ex[witness],witness_merit});
        for(size_t k=0;witness<0&&k<views.size()&&k<48;k++){
            const auto &v=views[k];
            double det=v.covariance.det();
            double area=std::sqrt(det)/sq(as),clutter=area*v.background;
            double full_clutter=area*full.rho[v.visit][mb];
            const bool broad=full_clutter<.90;
            const auto &f=broad?full.fields[v.visit]:strong.fields[v.visit];
            if(broad)clutter=full_clutter;
            if(f.pts.empty()||clutter>140.)continue;
            double pv=sq(.23)+sq(.31*(strong.raw.lever[v.visit]-color));
            size_t start=thirds.size();
            f.ellipse(v.center,v.covariance.xx,v.covariance.xy,v.covariance.yy,det,3.15,
                      [&](const Point &p,double dx,double dy,double chi){
                double dm=p.mag-mag;
                if(dm*dm>10.*pv)return;
                ++totals[1];
                double q=.44*std::clamp(double(p.q),-8.,13.)-.42*chi-.5*dm*dm/pv+v.merit
                         -.40*std::log((.025+clutter)/(.025+area*v.background));
                thirds.push_back({p,v.visit,float(q)});
            });
            if(thirds.size()>start+4){
                std::partial_sort(thirds.begin()+start,thirds.begin()+start+4,thirds.end(),[](auto&a,auto&b){return a.merit>b.merit;});
                thirds.resize(start+4);
            }
        }
        best(thirds,10);
        for(auto &b:thirds){
            Forecast at;double u,v;
            if(!model.condition(b.e,b.p,u,v,at))continue;
            ++totals[2];
            V observed{b.p.x,b.p.y};V offset=observed-at.center;
            for(auto &old:views){
                int e=old.visit;const auto &f=full.fields[e];
                if(f.e.night==full.fields[b.e].e.night)continue;
                auto pred=model.predict(e);
                auto cond=cadence_ranging::given(pred,at,offset);
                // condition() has already applied the posterior mean in
                // range and radial speed. Only transfer the remaining
                // *endpoint-centroid* correlation to the conditional mean.
                double shared=pred.left*at.left+pred.right*at.right;
                // Marginalizing the orbital prior jointly with centroid
                // errors leads to this common residual correction.
                V reg=at.covariance.solve(offset+at.range*u+at.radial*v);
                cond.center=pred.center+reg*shared;
                double det=cond.cov.det();if(!(det>0.))continue;
                double clutter=std::sqrt(det)/sq(as)*full.rho[e][mb];
                if(clutter>1.9)continue;
                double pv=sq(.225)+sq(.30*(full.raw.lever[e]-color));
                f.ellipse(cond.center,cond.cov.xx,cond.cov.xy,cond.cov.yy,det,3.0,
                          [&](const Point &p,double dx,double dy,double chi){
                    double dm=p.mag-(model.a.mag+model.b.mag+b.p.mag)/3.;
                    if(dm*dm>11.*pv)return;
                    ++totals[3];
                    double merit=.37*pair.merit+.64*b.merit+.28*std::clamp(double(p.q),-8.,12.)
                                 -.48*chi-.46*dm*dm/pv-.5*(u*u+v*v)-std::log(.0015+clutter);
                    here.push_back({{pair.a,pair.b,b.p.row,p.row},float(model.g),float(model.h),float(merit)});
                });
                if(here.size()>120)best(here,16);
            }
        }
        best(here,8);totals[4]+=here.size();out.insert(out.end(),here.begin(),here.end());
    }
};
}
extern "C" int transient_pairs(const original::Sky *sky,const double *depth,const double *east,
        const visit_triplets::Member *targets,int nt,const visit_triplets::Member *anchors,int na,
        const double *family,int threads,double seconds,int cap,visit_triplets::Pair *out,long long *counts){
    using namespace visit_triplets;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Catalogue cat(*sky,depth,east,targets,nt);
        std::atomic<int> next{0};std::vector<std::thread> workers;
        std::vector<std::vector<Pair>> batches(threads);std::vector<long long> seen(threads);
        for(int th=0;th<threads;th++)workers.emplace_back([&,th]{
            PairWorker w{cat};auto &save=batches[th];size_t limit=cap/threads+1;
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=na)break;
                w.anchor(anchors[i],family,save);
                if(save.size()>2*limit)best(save,limit);
            }
            best(save,limit);seen[th]=w.seen;
        });
        for(auto &w:workers)w.join();
        std::vector<Pair> all;for(auto &z:batches)all.insert(all.end(),z.begin(),z.end());
        std::sort(all.begin(),all.end(),[](auto&a,auto&b){
            if(a.a!=b.a)return a.a<b.a;
            return a.b!=b.b?a.b<b.b:a.merit>b.merit;
        });
        all.erase(std::unique(all.begin(),all.end(),[](auto&a,auto&b){return a.a==b.a&&a.b==b.b;}),all.end());
        best(all,cap);std::sort(all.begin(),all.end(),[](auto&a,auto&b){return a.merit>b.merit;});
        std::copy(all.begin(),all.end(),out);
        counts[0]=std::min(next.load(),na);counts[1]=std::accumulate(seen.begin(),seen.end(),0LL);
        return int(all.size());
    }catch(...){return -1;}
}
extern "C" int transient_quads(const original::Sky *sky,const double *depth,const double *east,
        const visit_triplets::Member *small,int ns,const visit_triplets::Member *large,int nl,
        const visit_triplets::Pair *pairs,int np,int threads,double seconds,int cap,
        visit_triplets::Proposal *out,long long *stats){
    using namespace visit_triplets;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Catalogue strong(*sky,depth,east,small,ns),full(*sky,depth,east,large,nl);
        std::vector<std::vector<Proposal>> batches(threads);std::atomic<int> next{0};
        std::vector<std::thread> jobs;std::vector<std::array<long long,5>> counts(threads);
        for(int th=0;th<threads;th++)jobs.emplace_back([&,th]{
            Extend w{strong,full};auto &result=batches[th];size_t limit=cap/threads+1;
            while(!expired(end)){
                int j=next.fetch_add(1);if(j>=np)break;
                w.pair(pairs[j],result);
                if(result.size()>2*limit)best(result,limit);
            }
            best(result,limit);std::copy(w.totals,w.totals+5,counts[th].begin());
        });
        for(auto &j:jobs)j.join();
        for(auto &c:counts)for(int k=0;k<5;k++)stats[k]+=c[k];
        std::vector<Proposal> all;for(auto &p:batches)all.insert(all.end(),p.begin(),p.end());
        best(all,cap);std::sort(all.begin(),all.end(),[](auto&a,auto&b){return a.merit>b.merit;});
        std::copy(all.begin(),all.end(),out);return int(all.size());
    }catch(...){return -1;}
}
namespace visit_triplets {
// Track several alternatives for ambiguous fifth nights, but only spend one
// output slot on identical recovered memberships.
void distinct_recoveries(std::vector<original::Recovery> &all,int cap){
    std::vector<int> order(all.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return all[a].orbit.merit>all[b].orbit.merit;});
    std::unordered_map<int,std::vector<int>> owners;
    std::vector<original::Recovery> kept;kept.reserve(std::min(int(all.size()),cap));
    std::vector<int> votes(cap,0),touched;
    for(int i:order){
        const auto &o=all[i].orbit;
        touched.clear();
        for(int j=0;j<o.count;j++){
            auto it=owners.find(o.row[j]);
            if(it!=owners.end())for(int k:it->second)if(++votes[k]==1)touched.push_back(k);
        }
        bool covered=false;int similar=0;
        for(int k:touched){
            int shared=votes[k],previous=kept[k].orbit.count;
            covered|=shared==o.count && (previous==o.count||previous>=o.count+2);
            similar+=shared>=4&&shared*4>=3*o.count&&shared*4>=3*previous;
            votes[k]=0;
        }
        if(covered||similar>=2)continue;
        int k=int(kept.size());
        for(int j=0;j<o.count;j++)owners[o.row[j]].push_back(k);
        kept.push_back(all[i]);
        if(int(kept.size())>=cap)break;
    }
    all.swap(kept);
}
}
extern "C" int transient_confirm(const original::Sky *sky,const visit_triplets::Proposal *p,int count,
        int threads,double seconds,int cap,original::Recovery *out,int *tested,
        const original::PairedReturns *returns){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> jobs;
        std::vector<std::vector<original::Recovery>> batches(threads);
        for(int th=0;th<threads;th++)jobs.emplace_back([&,th]{
            std::vector<Sample> points(4);std::vector<double> err;
            auto &batch=batches[th];
            while(!expired(end)){
                int j=next.fetch_add(1);if(j>=count)break;
                const auto &prop=p[j];
                ExactState orbit;orbit.g=prop.g;orbit.h=prop.h;orbit.ref=0;
                for(int k=0;k<4;k++){points[k]=sky->sample(prop.row[k]);orbit.ref+=.25*points[k].t;}
                orbit.x=.5*(points[0].x+points[1].x);orbit.y=.5*(points[0].y+points[1].y);
                if(!measured_fit(points,orbit,err,8)||*std::max_element(err.begin(),err.end())>.69)continue;
                double rss=0.;for(double r:err)rss+=r*r;
                if(rss>4.*sq(.33))continue;
                original::Core core{};
                double par[]={orbit.x,orbit.y,orbit.g,orbit.vx,orbit.vy,orbit.h};
                std::copy(par,par+6,core.x);core.ref=orbit.ref;core.count=4;
                std::copy(prop.row,prop.row+4,core.rows);
                core.merit=std::clamp(.25*(prop.merit-8.)-.95*rss/sq(.24),-4.,5.);
                original::Recovery hits[2];
                int n=original::recover(*sky,core,hits,true,prop.witness,returns);
                if(returns)for(int k=0;k<n;k++)hits[k].orbit.merit+=returns->priority(hits[k].orbit);
                batch.insert(batch.end(),hits,hits+n);
                if(int(batch.size())>2*cap/threads+32)visit_triplets::distinct_recoveries(batch,cap/threads+1);
            }
        });
        for(auto &j:jobs)j.join();
        std::vector<original::Recovery> all;for(auto &p:batches)all.insert(all.end(),p.begin(),p.end());
        visit_triplets::distinct_recoveries(all,cap);
        int n=int(all.size());std::copy(all.begin(),all.end(),out);
        *tested=std::min(count,next.load());return n;
    }catch(...){return -1;}
}
extern "C" void *make_paired_returns(const original::Sky *sky,const int *rows,const int *peers,const float *prior,int count){
    try{return new original::PairedReturns(*sky,rows,peers,prior,count);}catch(...){return nullptr;}
}
extern "C" void free_paired_returns(const original::PairedReturns *p){delete p;}
extern "C" int visit_transfer_table(const original::Sky *sky,const double *depth,const double *east,
        const visit_triplets::Pair *pairs,int n,double *out){
    using namespace visit_triplets;
    try{
        Catalogue cat(*sky,depth,east,nullptr,0);
        const double u[]={0.,1.,-1.,0.,0.,1.},v[]={0.,0.,0.,1.,-1.,1.};
        for(int j=0;j<n;j++){
            Model m(cat,pairs[j]);double *a=out+68*j;
            for(int i=0;i<6;i++){
                Transfer tr(m.a,m.b,m.ea,m.eb,m.g0+u[i]*m.sg,m.h0+v[i]*m.sh);
                double data[]={tr.pos.x,tr.pos.y,tr.rate.x,tr.rate.y,tr.g,tr.h,tr.k,tr.rad,tr.vel};
                std::copy(data,data+9,a+9*i);
            }
            double info[]={m.ea.t,m.eb.t-m.ea.t,m.sg,m.sh,m.sa/as,m.sb/as,m.g0,
                .5*(m.a.mag+m.b.mag),.5*(sky->lever[sky->ex[m.a.row]]+sky->lever[sky->ex[m.b.row]]),
                double(m.ea.night),double(m.eb.night),double(pairs[j].merit),m.ea.oz,m.eb.oz};
            std::copy(info,info+14,a+54);
        }
        return n;
    }catch(...){return -1;}
}
extern "C" int transient_local_pairs(const original::Sky *sky,const double *depth,const double *direction,
        const visit_triplets::Member *table,int n,int threads,double seconds,int cap,
        visit_triplets::Pair *out,long long *stats){
    using namespace visit_triplets;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Catalogue cat(*sky,depth,direction,table,n);
        struct Job{int a,b;double value,weight;};
        std::vector<Job> tasks;
        for(int a=0;a<int(cat.fields.size());a++)for(int b=a+1;b<int(cat.fields.size());b++){
            const auto &f=cat.fields[a],&h=cat.fields[b];double t=h.e.t-f.e.t,days=std::abs(t);
            if(f.e.night==h.e.night||days<.44||days>100.||f.pts.empty()||h.pts.empty())continue;
            V pos{.5*(f.sky.l+f.sky.r),.5*(f.sky.b+f.sky.t)};
            V par{h.e.ox-f.e.ox-pos.x*(h.e.oz-f.e.oz),h.e.oy-f.e.oy-pos.y*(h.e.oz-f.e.oz)};
            V shift=cat.east*(t*.000061)-par*(1./43.);
            double pad=.32*std::sqrt(dot(par,par))/43.+days*.000040+.00007;
            double close=f.sky.shift(shift,pad).meet(h.sky).area();
            double distant=f.sky.shift(par*(-1./260.),days*.000006+.00007).meet(h.sky).area();
            if(std::max(close,distant)<=0.)continue;
            double overlap=std::max(close,.35*distant)/std::max(1.e-9,std::min(f.sky.area(),h.sky.area()));
            double w=std::sqrt(std::clamp(overlap,.02,1.))/(1.+days/9.);
            unsigned hash=unsigned(a+1)*2246822519U^unsigned(b+1)*2654435761U;
            double value=(.6+.4*(hash/4294967296.))*w;
            tasks.push_back({a,b,value,w});
        }
        std::sort(tasks.begin(),tasks.end(),[](auto&a,auto&b){return a.value>b.value;});
        if(tasks.empty())return 0;
        double sum=0.;for(auto &j:tasks)sum+=j.weight;
        std::atomic<int> next{0};std::vector<std::thread> jobs;
        std::vector<std::vector<Pair>> batches(threads);
        std::vector<long long> discoveries(threads);
        for(int th=0;th<threads;th++)jobs.emplace_back([&,th]{
            std::vector<Pair> local,slow;auto &save=batches[th];
            for(;;){
                int i=next.fetch_add(1);if(i>=int(tasks.size())||expired(end))break;
                auto job=tasks[i];int a=job.a,b=job.b;
                if(cat.fields[a].pts.size()>cat.fields[b].pts.size())std::swap(a,b);
                auto &one=cat.fields[a];auto &two=cat.fields[b];
                double t=two.e.t-one.e.t,days=std::abs(t);
                int budget=std::max(32,int(cap*1.48*job.weight/sum));
                local.clear();slow.clear();int loop=0;
                for(auto &v:one.pts){
                    if(((++loop)&1023)==0&&expired(end))break;
                    if(v.mag>depth[b]+.68)continue;
                    // Use continuous distance within a disk component; a
                    // broad second component admits nearer, distant and
                    // inclined motion at short cadences.
                    for(int family=0;family<3;family++){
                        if(family==1 && days>=20.)continue;
                        double r0=family==2?260.:(family?67.:42.5);
                        PairGate gate(cat,v.row,b,r0,family?.53:.21,family?.52:.23,family?.73:.34);
                        double area=std::sqrt(gate.det)/sq(as);
                        double radius=std::clamp(std::sqrt(70./(.001+gate.rate*area)),.40,2.9);
                        two.ellipse(gate.center,gate.xx,gate.xy,gate.yy,gate.det,radius,
                          [&](const Point &q,double dx,double dy,double chi){
                            double var=cat.varmag(v.row,q.row),dm=(q.mag-v.mag);
                            if(dm*dm>11.*var)return;
                            double gamma=gate.distance({dx,dy});
                            if(!(gamma>1./6000.&&gamma<1./16.))return;
                            double score=.46*(std::clamp(double(v.q),-10.,15.)+std::clamp(double(q.q),-10.,15.))
                                -.40*chi-.47*dm*dm/var-.40*std::log(var)
                                -.57*std::log(.018+area*std::sqrt(gate.rate*cat.rho[a][Catalogue::mb(q.mag)]))-.75*family;
                            if(family==2 && (gamma>1./110. || std::hypot(q.x-v.x,q.y-v.y)<.70*as))return;
                            (family==2?slow:local).push_back({std::min(v.row,q.row),std::max(v.row,q.row),float(score),float(gamma)});
                            ++discoveries[th];
                          });
                    }
                    if(int(local.size())>3*budget+256)best(local,2*budget);
                    if(int(slow.size())>budget+128)best(slow,budget/2+32);
                }
                best(slow,std::max(8,budget/5));best(local,budget-int(slow.size()));
                save.insert(save.end(),local.begin(),local.end());
                save.insert(save.end(),slow.begin(),slow.end());
            }
        });
        for(auto &w:jobs)w.join();
        std::vector<Pair> all;for(auto &b:batches)all.insert(all.end(),b.begin(),b.end());
        std::sort(all.begin(),all.end(),[](auto&a,auto&b){
            if(a.a!=b.a)return a.a<b.a;
            return a.b!=b.b?a.b<b.b:a.merit>b.merit;
        });
        all.erase(std::unique(all.begin(),all.end(),[](auto&a,auto&b){return a.a==b.a&&a.b==b.b;}),all.end());
        best(all,cap);std::sort(all.begin(),all.end(),[](auto&a,auto&b){return a.merit>b.merit;});
        std::copy(all.begin(),all.end(),out);stats[0]=std::min(int(tasks.size()),next.load());
        stats[1]=tasks.size();stats[2]=std::accumulate(discoveries.begin(),discoveries.end(),0LL);
        return int(all.size());
    }catch(...){return -1;}
}
