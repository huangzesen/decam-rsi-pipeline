// Enumerate measured directions in a co-moving cell, allowing the angular
// speed, range and range rate to move continuously away from the template.
namespace cell_orbits {
using encounter::Arc;
struct Point {
    float x,y,mag,quality;
    int exposure,row;
};
struct Map {
    double ox,oy,xx,xy,yx,yy,fg,inverse_f;
    int epoch,night;
};
struct Group {int head=-1,size=0;};
struct Query {
    double t,x,y,px,py;
    int row,night;
    float q,mag;
};
struct Result {int rows[4];float g,merit;};
static_assert(sizeof(Result)==24);

// Eliminate the intercept and sky rate with the outer two directions. Two
// inner directions give four constraints on inverse range and radial rate.
// Scale range and radial rate before solving so short time baselines remain
// well conditioned. The final fit uses the original unwarped observations.
struct Bend {
    double x,y,gx,gy,hx,hy,noise;
    Bend()=default;
    Bend(const Query &a,const Query &b,const Query &c,double g,double vc) {
        double u=(b.t-a.t)/(c.t-a.t);
        double l=1.-u;
        gx=(b.px-l*a.px-u*c.px)*g;
        gy=(b.py-l*a.py-u*c.py)*g;
        hx=(l*(b.x-a.x)*a.t+u*(b.x-c.x)*c.t)*vc;
        hy=(l*(b.y-a.y)*a.t+u*(b.y-c.y)*c.t)*vc;
        x=(b.x-a.x)-u*(c.x-a.x)+gx;
        y=(b.y-a.y)-u*(c.y-a.y)+gy;
        noise=1.+l*l+u*u;
    }
};
static bool intersect(const Bend &a,const Bend &b,double sigma2,double &g,double &h,double &chi) {
    double av=1./a.noise,bv=1./b.noise;
    auto dot4=[&](double x,double y,double z,double w,double X,double Y,double Z,double W){
        return av*(x*X+y*Y)+bv*(z*Z+w*W);
    };
    double gg=dot4(a.gx,a.gy,b.gx,b.gy,a.gx,a.gy,b.gx,b.gy)+sigma2*7.;
    double hh=dot4(a.hx,a.hy,b.hx,b.hy,a.hx,a.hy,b.hx,b.hy)+sigma2*8.;
    double gh=dot4(a.gx,a.gy,b.gx,b.gy,a.hx,a.hy,b.hx,b.hy);
    double gr=dot4(a.gx,a.gy,b.gx,b.gy,a.x,a.y,b.x,b.y);
    double hr=dot4(a.hx,a.hy,b.hx,b.hy,a.x,a.y,b.x,b.y);
    double det=gg*hh-gh*gh;
    if(!(det>0.))return false;
    g=(gh*hr-hh*gr)/det;h=(gh*gr-gg*hr)/det;
    if(std::abs(g)>.65||std::abs(h)>1.25)return false;
    chi=av*(sq(a.x+g*a.gx+h*a.hx)+sq(a.y+g*a.gy+h*a.hy))+
        bv*(sq(b.x+g*b.gx+h*b.hx)+sq(b.y+g*b.gy+h*b.hy));
    return chi<15.*sigma2;
}

struct Search {
    const Witness &w;
    const Point *points;
    const Map *maps;
    double gamma,vc;
    int cap;
    std::vector<Result> hits,triples;
    long long tested=0,near=0;

    void group(const int *ids,int n) {
        constexpr int max_points=24;
        Query q[max_points];
        int used=0;
        for(int j=0;j<n&&used<max_points;j++){
            const auto &p=points[ids[j]];
            const auto &m=maps[p.exposure]; const auto &e=w.images[p.exposure].e;
            // Close repeats cannot generate a new independent epoch. Preserve
            // distinct sky positions on the same night in a crowded cell.
            bool duplicate=false;
            for(int k=0;k<used;k++)if(q[k].night==m.night&&
                sq(q[k].x-p.x)+sq(q[k].y-p.y)<sq(1.6*as)){duplicate=true;break;}
            if(duplicate)continue;
            q[used++]={m.fg,p.x,p.y,(e.ox-p.x*e.oz)*m.inverse_f,
                    (e.oy-p.y*e.oz)*m.inverse_f,p.row,m.night,p.quality,p.mag};
        }
        if(used<3)return;
        std::sort(q,q+used,[](const auto &a,const auto &b){return a.t<b.t;});
        int nd=1;
        for(int i=1;i<used;i++)nd+=q[i].night!=q[i-1].night;
        if(nd<3)return;
        for(int i=0;i+2<used;i++)for(int j=i+2;j<used;j++){
            const auto &a=q[i],&c=q[j];
            double dt=c.t-a.t;
            if(dt<3.2||a.night==c.night)continue;
            // The f,g coefficients here are evaluated for the nominal orbit.
            // Their error grows cubically in angular phase. The subsequent
            // fit relinearizes gravity at the inferred distance.
            double phase=std::min(.8,vc*dt);
            double sigma=.29*as+.012*phase*phase*phase;
            Bend bends[max_points] = {};
            int inside[max_points],ni=0;
            for(int k=i+1;k<j;k++)if(q[k].night!=a.night&&q[k].night!=c.night){
                inside[ni]=k;
                Bend v(a,q[k],c,gamma,vc);
                bends[ni++]=v;
                // Integrated likelihood of an independent direction given
                // both endpoints. The rank-one range uncertainty is normally
                // much larger than the cross-track error near opposition.
                double s2=sigma*sigma*v.noise;
                double gg=.20*.20,hh=.28*.28;
                double xx=s2+gg*sq(v.gx)+hh*sq(v.hx);
                double yy=s2+gg*sq(v.gy)+hh*sq(v.hy);
                double xy=gg*v.gx*v.gy+hh*v.hx*v.hy;
                double det=xx*yy-xy*xy;
                if(!(det>0.))continue;
                double wx=(yy*v.x-xy*v.y)/det,wy=(xx*v.y-xy*v.x)/det;
                double chi=v.x*wx+v.y*wy;
                if(chi>19.)continue;
                double dg=-gg*(v.gx*wx+v.gy*wy),h=-hh*(v.hx*wx+v.hy*wy);
                double g=gamma*(1.+dg);
                if(!(g>1./9000.&&g<1./15.))continue;
                double rx=(c.x-a.x+g*(c.px-a.px)+h*vc*(c.x*c.t-a.x*a.t))/dt;
                double ry=(c.y-a.y+g*(c.py-a.py)+h*vc*(c.y*c.t-a.y*a.t))/dt;
                double speed2=(sq(rx)+sq(ry)+sq(h*vc))/(mu*g*g*g);
                if(speed2>2.08)continue;
                double m=(a.mag+q[k].mag+c.mag)/3.;
                double phot=sq(a.mag-m)+sq(q[k].mag-m)+sq(c.mag-m);
                double gap=std::min(q[k].t-a.t,c.t-q[k].t);
                // Distinct nights both tightly localize the next search
                // and test a curvature. Long gaps alone offer many trials.
                double span=std::log1p(dt/18.)+std::log1p(gap/9.);
                double merit=.64*(a.q+q[k].q+c.q)-.50*chi-3.3*phot-
                    .22*std::log(det/(sq(as)*sq(as)))-std::min(4.,3.*sq(speed2-1.))+.35*span;
                triples.push_back({{a.row,q[k].row,c.row,-1},float(g),float(merit)});
            }
            for(int k=0;k<ni;k++)for(int l=k+1;l<ni;l++){
                const auto &b=q[inside[k]],&d=q[inside[l]];
                if(b.night==d.night)continue;
                tested++;
                double dg,h,chi;
                if(!intersect(bends[k],bends[l],sigma*sigma,dg,h,chi))continue;
                double g=gamma*(1.+dg);
                double rx=(c.x-a.x+g*(c.px-a.px))/dt+h*vc*.5*(c.x+a.x);
                double ry=(c.y-a.y+g*(c.py-a.py))/dt+h*vc*.5*(c.y+a.y);
                if(sq(rx)+sq(ry)+sq(h*vc)>2.5*mu*g*g*g)continue;
                near++;
                Arc arc{{a.row,b.row,d.row,c.row,-1},float(g),0.f};
                if(!encounter::preliminary(w,arc))continue;
                double mean=.25*(a.mag+b.mag+c.mag+d.mag);
                double scatter=sq(a.mag-mean)+sq(b.mag-mean)+sq(c.mag-mean)+sq(d.mag-mean);
                float merit=arc.score+.62f*(a.q+b.q+c.q+d.q)-float(2.5*scatter+2.*dg*dg+h*h);
                hits.push_back({{a.row,b.row,d.row,c.row},arc.g,merit});
            }
        }
        for(auto *set:{&hits,&triples})if(int(set->size())>3*cap){
            std::nth_element(set->begin(),set->begin()+cap,set->end(),[](auto &a,auto &b){return a.merit>b.merit;});
            set->resize(cap);
        }
    }
};
}

// rev is an optional device-sorted permutation. The host path builds the
// partitions in linear time using a hash table with a contiguous row chain.
extern "C" int comoving_quads(const Witness *w,const cell_orbits::Point *points,int n,
        const cell_orbits::Map *maps,double scale,double offset,double gamma,
        const int *rev,const int *beg,const int *end,int groups,int cap,int threads,double seconds,
        cell_orbits::Result *out,long long *counts) {
    try{
        using namespace cell_orbits;
        auto until=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::vector<int> chain,group_ids;
        std::vector<Group> buckets;
        if(!rev){
            size_t size=1;while(size<size_t(n)*1.55)size*=2;
            std::vector<uint64_t> keys(size);
            chain.resize(n);buckets.resize(size);
            for(int j=n-1;j>=0;j--){
                const Point &p=points[j];const auto &m=maps[p.exposure];
                int x=std::floor(m.ox+m.xx*p.x+m.yx*p.y),y=std::floor(m.oy+m.xy*p.x+m.yy*p.y);
                int bin=std::floor(p.mag*scale+offset);
                uint64_t key=(uint64_t(m.epoch&255)<<48)|(uint64_t(bin&255)<<40)|
                            (uint64_t(x&0xfffff)<<20)|uint64_t(y&0xfffff);
                key++;
                uint64_t hash=key;hash^=hash>>30;hash*=0xbf58476d1ce4e5b9ULL;
                hash^=hash>>27; hash*=0x94d049bb133111ebULL;hash^=hash>>31;
                size_t z=hash&(size-1);
                while(keys[z]&&keys[z]!=key)z=(z+1)&(size-1);
                keys[z]=key;auto &b=buckets[z];chain[j]=b.head;b.head=j;
                if(++b.size==3)group_ids.push_back(z);
            }
            groups=group_ids.size();
        }
        auto better=[](const cell_orbits::Result &a,const cell_orbits::Result &b){return a.merit>b.merit;};
        std::vector<Search> workers;
        std::vector<std::thread> pool;
        std::atomic<int> next{0};
        for(int i=0;i<threads;i++)workers.push_back({*w,points,maps,gamma,std::sqrt(mu*gamma*gamma*gamma),cap});
        for(auto &worker:workers)pool.emplace_back([&,s=&worker]{
            int rows[24];
            while(!expired(until)){
                int lo=next.fetch_add(48);if(lo>=groups)break;
                for(int i=lo;i<std::min(lo+48,groups);i++){
                    if(rev)s->group(rev+beg[i],std::min(24,end[i]-beg[i]));
                    else{
                        const auto &b=buckets[group_ids[i]];int k=0;
                        for(int j=b.head;j>=0&&k<24;j=chain[j])rows[k++]=j;
                        s->group(rows,k);
                    }
                }
            }
        });
        for(auto &t:pool)t.join();
        std::vector<cell_orbits::Result> hits,triples;
        counts[0]=groups;counts[1]=std::min(groups,next.load());
        for(auto &s:workers){counts[2]+=s.tested;counts[3]+=s.near;
            hits.insert(hits.end(),s.hits.begin(),s.hits.end());
            triples.insert(triples.end(),s.triples.begin(),s.triples.end());}
        counts[4]=hits.size();
        counts[5]=triples.size();
        int written=0;
        for(auto *set:{&hits,&triples}){
            int limit=set==&hits?cap/2:cap-written;
            if(int(set->size())>limit){std::nth_element(set->begin(),set->begin()+limit,set->end(),better);set->resize(limit);}
            std::sort(set->begin(),set->end(),better);std::copy(set->begin(),set->end(),out+written);
            written+=set->size();
        }
        return written;
    }catch(...){return -1;}
}
