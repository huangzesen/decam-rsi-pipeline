// Estimate corroboration for every measured pair, using another disjoint
// arc and sky-displaced comparison arcs in the same element chart.
namespace orbit_pair_support {
struct Match {float merit=-99.f,g=0.f;int partner=-1;};
struct Neighbours {Match real;float control=-99.f;};
struct Joined {
    const ShapeCatalog &cat;
    Clock::time_point end;
    std::vector<PairHashSlot> table;
    std::vector<int> chain;
    std::vector<Neighbours> scores;
    std::array<long long,4> stats{};

    void compare(int i,int j,double ec,double es,double shiftx,double shifty){
        const auto &a=cat.units[i],&b=cat.units[j];const auto *s=cat.s;
        if(s[a.a].night==s[b.a].night||s[a.a].night==s[b.b].night||
           s[a.b].night==s[b.a].night||s[a.b].night==s[b.b].night)return;
        double gap=.5*std::abs(s[a.a].t+s[a.b].t-s[b.a].t-s[b.b].t);
        if(gap<25.)return;
        double dm=.5*(s[a.a].mag+s[a.b].mag-s[b.a].mag-s[b.b].mag);
        if(std::abs(dm)>.98)return;
        bool control=shiftx||shifty;
        const auto &p=cat.model[i],&q=cat.model[j];
        double dt=std::abs(s[a.b].t-s[a.a].t),du=std::abs(s[b.b].t-s[b.a].t);
        double err=.22*std::hypot(1./dt,1./du);
        double travel=std::max(std::abs(s[a.a].t-cat.ref),std::abs(s[b.a].t-cat.ref));
        double e2=ec*ec+es*es,ast=(2.+travel*err)*as;
        double sig[]={ast+.000045+.007*e2,ast+.000035+.0035*e2,
                      .008+.14*e2+.018*err,.006+.10*e2+.16*err};
        double aa=36.,bb=22.,ab=0.,ax=0.,bx=0.,yy=0.;
        for(int k=0;k<4;k++){
            double u=(p.dx[k]-q.dx[k])/sig[k],v=(p.dy[k]-q.dy[k])/sig[k];
            double d=(p.q[k]-q.q[k]+(k==0?shiftx:k==1?shifty:0.))/sig[k]+ec*u+es*v;
            aa+=u*u;bb+=v*v;ab+=u*v;ax+=u*d;bx+=v*d;yy+=d*d;
        }
        double det=aa*bb-ab*ab,dx=(ab*bx-bb*ax)/det,dy=(ab*ax-aa*bx)/det;
        double chi=std::max(0.,yy+dx*ax+dy*bx);
        if(chi>18.||std::abs(dx)>.31||std::abs(dy)>.44)return;
        double r=.5*(p.radius+q.radius+(ec+dx)*(p.rx+q.rx)+(es+dy)*(p.ry+q.ry));
        if(!(r>18.&&r<135.))return;
        double phot=.077+sq(.15*(s[a.a].color+s[a.b].color-s[b.a].color-s[b.b].color));
        double quality=6.5-.47*chi-.45*dm*dm/phot-.20*sq((ec+dx)/.19)-.18*sq((es+dy)/.24);
        quality+=.46*std::log1p(std::min(dt,du))-.37*std::log1p(err*travel/18.);
        if(control){
            scores[i].control=std::max(scores[i].control,float(quality));
            scores[j].control=std::max(scores[j].control,float(quality));
            ++stats[0];return;
        }
        ++stats[1];
        if(quality+1.2<std::min(scores[i].real.merit,scores[j].real.merit))return;
        Orbit orbit;
        if(!quartet(s,a,b,1./r,orbit,false,{17.,145.,6000.,false}))return;
        ++stats[2];
        // The two-epoch parallax and all four measured directions must
        // support a bound trajectory before it corroborates either pair.
        double score=quality+.9-.32*std::min(25.,orbit.chi);
        if(score>scores[i].real.merit)scores[i].real={float(score),float(orbit.g),j};
        if(score>scores[j].real.merit)scores[j].real={float(score),float(orbit.g),i};
    }
    void scan(const double *trial){
        double ec=trial[0],es=trial[1],wx=trial[2]*as,wy=.64*wx;
        const double width[]={wx,wy,.072,.047};
        size_t slots=1;while(slots<size_t(cat.n)*1.55)slots*=2;
        table.assign(slots,{});chain.resize(cat.n);
        double ang=trial[3]*6.28318530718;
        double sx=wx*(3.8+4.*trial[4])*std::cos(ang),sy=wy*(3.8+4.*trial[5])*std::sin(ang);
        auto slot_for=[&](uint64_t k){
            size_t j=permute(k)&(slots-1);
            while(table[j].head>=0&&table[j].key!=k)j=(j+1)&(slots-1);
            return j;
        };
        for(int i=0;i<cat.n;++i){
            if((i&1023)==0 && expired(end))return;
            auto &m=cat.model[i];if(!(m.radius>0.))continue;
            double v[4];int bin[4];bool good=true;
            for(int k=0;k<4;k++){
                v[k]=m.q[k]+ec*m.dx[k]+es*m.dy[k];
                if(!std::isfinite(v[k])||std::abs(v[k])>4.){good=false;break;}
                bin[k]=int(std::floor(v[k]/width[k]+trial[k+3]));
            }
            if(!good)continue;
            auto key=element_key(bin);auto real_slot=slot_for(key);
            for(int kind=0;kind<2;kind++){
                int head=table[real_slot].head;
                if(kind){
                    bin[0]=int(std::floor((v[0]+sx)/wx+trial[3]));
                    bin[1]=int(std::floor((v[1]+sy)/wy+trial[4]));
                    head=table[slot_for(element_key(bin))].head;
                }
                int n=0;for(int j=head;j>=0&&n<40;j=chain[j],n++)
                    compare(i,j,ec,es,kind?sx:0.,kind?sy:0.);
            }
            chain[i]=table[real_slot].head;table[real_slot]={key,i};
        }
        ++stats[3];
    }
};
}

extern "C" int orbital_pair_support(const Witness *w,const Unit *pairs,int n,const double *color,const double *east,
        double epoch,const double *trials,int samples,int threads,double seconds,float *support,int *partners,
        float *range,long long *stats){
    try{
        using namespace orbit_pair_support;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::vector<int> compact(w->nt,-1),owner;std::vector<Source> sources;
        std::vector<Unit> units(pairs,pairs+n);
        auto add=[&](int row){
            if(compact[row]>=0)return compact[row];
            int i=sources.size();auto &p=w->objects[row];auto &e=w->images[p.exposure].e;
            sources.push_back({p.x,p.y,e.t,e.ox,e.oy,e.oz,p.mag,color[p.exposure],7.,0.,e.night,0});
            owner.push_back(row);return compact[row]=i;
        };
        for(auto &u:units){u.a=add(u.a);u.b=add(u.b);}
        ShapeCatalog cat(sources.data(),owner.data(),units.data(),*w,n,epoch,{east[0],east[1],0.});
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int t=0;t<threads;t++)pool.emplace_back([&]{
            while(!expired(end)){
                int at=next.fetch_add(256);if(at>=n)break;
                for(int k=at;k<std::min(at+256,n);++k)cat.prepare(k);
            }
        });
        for(auto &t:pool)t.join();pool.clear();next=0;
        std::vector<Joined> jobs;
        for(int t=0;t<threads;t++){jobs.push_back({cat,end});jobs.back().scores.resize(n);}
        for(auto &j:jobs)pool.emplace_back([&,job=&j]{
            while(!expired(end)){
                int k=next.fetch_add(1);if(k>=samples)break;job->scan(trials+7*k);
            }
        });
        for(auto &t:pool)t.join();
        int matched=0;
        for(int k=0;k<n;k++){
            Neighbours best;
            for(auto &j:jobs){
                auto &v=j.scores[k];
                best.control=std::max(best.control,v.control);
                if(v.real.merit>best.real.merit)best.real=v.real;
            }
            partners[k]=best.real.partner;range[k]=best.real.g;
            support[2*k]=best.real.merit;support[2*k+1]=best.control;
            matched+=best.real.partner>=0;
        }
        for(auto &j:jobs)for(int k=0;k<4;k++)stats[k]+=j.stats[k];
        return matched;
    }catch(...){return -1;}
}
