// Over a single apparition, a bound body tens of thousands of au away
// moves less than one seeing disc after correcting for annual parallax.
namespace parallax_sweep {
using annual::V;
struct Point {int row,night;double t,mag;V sky,observer;};
uint64_t cell(int x,int y){return (uint64_t(uint32_t(x))<<32)|uint32_t(y);}

void raw_third(annual::Worker &w,const Point &a,const Point &b,double amplitude){
    const auto &s=w.s;double mag=.5*(a.mag+b.mag);
    double elapsed=b.t-a.t;
    if(std::abs(elapsed)<550.||std::abs(a.mag-b.mag)>.68)return;
    std::array<int,13> tried{};int count=0,fits=0;
    for(auto view:w.plan[s.ex[a.row]]){
        const auto &im=s.images[view.e];const auto &e=im.exposure;
        double fraction=(e.t-a.t)/elapsed;
        if(fraction<-.45||fraction>1.45||e.night==a.night||e.night==b.night||w.depth[view.e]<mag-.72||
           std::find(tried.begin(),tried.begin()+count,e.night)!=tried.begin()+count)continue;
        V center=a.sky+(b.sky-a.sky)*fraction;
        V obs=annual::observer(e,center*as),lever=obs-(a.observer+(b.observer-a.observer)*fraction);
        double length=std::sqrt(dot(lever,lever));
        if(length<.24)continue;
        V predicted=(center-obs*amplitude)*as;
        if(!annual::covered(im,predicted,1.8*as))continue;
        tried[count++]=e.night;
        double cost;int row=annual::nearest(s,view.e,predicted,mag,std::min(3.1,.66+1.2*length),cost);
        if(row>=0){
            int seed[3]={a.row,b.row,row};
            std::array<int,3> sorted={a.row,b.row,row};std::sort(sorted.begin(),sorted.end());
            uint64_t key=1469598103934665603ull;
            for(int r:sorted)key=(key^uint32_t(r))*1099511628211ull;
            annual::State state;
            if(!w.visited.count(key)&&annual::fit(s,seed,3,state)&&std::abs(state.g/as-amplitude)<1.55&&state.g>5.4*state.error){
                w.visited.insert(key);++w.count[2];++fits;
                w.recover(state,seed,(2.*mag+s.mag[row])/3.);
            }
        }
        if(count==int(tried.size())||fits>=2)break;
    }
}

void tile(annual::Worker &w,const int *rows,int n,double amplitude){
    const auto &s=w.s;
    std::vector<Point> points;
    double tmin=1.e20,tmax=-1.e20;
    for(int k=0;k<n;k++){
        int r=rows[k];const auto &e=s.images[s.ex[r]].exposure;
        if(!std::isfinite(s.mag[r])||s.flag[r]&1||s.sharp[r]<.10||s.sharp[r]>2.25)continue;
        V p=annual::position(s,r);V b=annual::observer(e,p);
        points.push_back({r,e.night,e.t,s.mag[r],p*(1./as)+b*amplitude,b});
        tmin=std::min(tmin,e.t);tmax=std::max(tmax,e.t);
    }
    if(points.size()<5||tmax-tmin<460.)return;
    const double vesc=std::sqrt(2.*mu*std::pow((amplitude+1.)*as,3))/as*1.12;
    const double reach=2.4+(tmax-tmin)*vesc;
    const double pitch=std::max(2.5,reach*.48);
    std::unordered_map<uint64_t,int> heads;
    std::vector<int> next(points.size(),-1);
    for(int i=0;i<int(points.size());i++){
        V q=points[i].sky;
        auto h=cell(int(std::floor(q.x/pitch)),int(std::floor(q.y/pitch)));
        auto it=heads.find(h);
        if(it!=heads.end())next[i]=it->second;
        heads[h]=i;
    }
    std::vector<int> near,short_arc;
    for(int i=0;i<int(points.size());i++){
        if((i%128)==0&&expired(w.end))return;
        ++w.count[0];
        const auto &a=points[i];near.clear();short_arc.clear();
        int bx=int(std::floor(a.sky.x/pitch)),by=int(std::floor(a.sky.y/pitch));
        int ext=int(std::ceil(reach/pitch));
        for(int y=by-ext;y<=by+ext;y++)for(int x=bx-ext;x<=bx+ext;x++){
            auto h=heads.find(cell(x,y));if(h==heads.end())continue;
            for(int j=h->second;j>=0;j=next[j]){
                const auto &b=points[j];
                if(b.night==a.night||std::abs(b.mag-a.mag)>.95)continue;
                double dt=std::abs(b.t-a.t);
                V base=b.observer-a.observer,err=b.sky-a.sky;
                double uncertainty=.95+1.12*std::sqrt(dot(base,base))+vesc*dt;
                if(dot(err,err)>sq(uncertainty))continue;
                near.push_back(j);
                if(j>i&&dt>1.3&&dt<590.)short_arc.push_back(j);
            }
        }
        if(near.size()<=12){
            int partners=0;
            for(int k:near)if(k>i&&std::abs(points[k].t-a.t)>550.){
                raw_third(w,a,points[k],amplitude);
                if(++partners==2)break;
            }
        }
        if(near.size()<2||near.size()>64||short_arc.empty())continue;
        for(int j:short_arc)for(int k:near){
            const auto &b=points[j],&c=points[k];
            if(c.night==b.night||std::abs(c.t-a.t)<460.||
               std::abs(c.t-b.t)<320.||std::abs(c.mag-b.mag)>.91)continue;
            int ids[3]={a.row,b.row,c.row};
            // A nearby range grid node can produce the same seed.
            std::array<int,3> sorted={ids[0],ids[1],ids[2]};std::sort(sorted.begin(),sorted.end());
            uint64_t key=1469598103934665603ull;
            for(int r:sorted)key=(key^uint32_t(r))*1099511628211ull;
            if(w.visited.count(key))continue;
            ++w.count[1];annual::State state;
            if(!annual::fit(s,ids,3,state)||std::abs(state.g/as-amplitude)>1.9)continue;
            w.visited.insert(key);++w.count[2];
            bool fourth=false;
            for(int l:near){
                const auto &q=points[l];
                if(q.night==b.night||q.night==c.night)continue;
                V residual=(annual::position(s,q.row)-annual::apparent(state,s.images[s.ex[q.row]].exposure))*(1./as);
                if(dot(residual,residual)<sq(.88)){fourth=true;break;}
            }
            if(fourth||(near.size()<=6&&state.g>7.*state.error))
                w.recover(state,ids,(a.mag+b.mag+c.mag)/3.);
        }
        if(w.visited.size()>500000)w.visited.clear();
    }
}
}

extern "C" int annual_range_sweep(const original::Sky *sky,const int *row,const int *bounds,int cells,
                        const double *depth,double seconds,int cap,annual::Report *out,int max_null,
                        double *decoy,long long *stats){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        auto plan=annual::plans(*sky,depth);
        int nn=0;for(const auto &f:sky->images)nn=std::max(nn,f.exposure.night+1);
        annual::Worker pos{*sky,depth,plan,nn,end},neg{*sky,depth,plan,nn,end};
        // Interleave distances and signed control trials across observed patches.
        const double amplitudes[]={4.1,8.2,12.3,2.05,6.15,10.25,14.35};
        int done=0;
        for(int level=0;level<7&&!expired(end);level++){
            for(int j=0;j<cells&&!expired(end);j++){
                double amplitude=amplitudes[(j+level)%7];
                for(int sign=0;sign<2;sign++){
                    bool real=((done+sign)%2)==0;
                    annual::sense=real?1.:-1.;
                    parallax_sweep::tile(real?pos:neg,row+bounds[j],bounds[j+1]-bounds[j],amplitude);
                }
                ++done;
            }
        }
        int result=0;
        for(int sign=0;sign<2;sign++){
            auto &w=sign?neg:pos;
            std::sort(w.found.begin(),w.found.end(),[](const auto &a,const auto &b){return a.merit>b.merit;});
            std::unordered_set<int> used;int count=0;
            for(const auto &r:w.found){
                int shared=0;for(int j=0;j<r.count;j++)shared+=used.count(r.rows[j]);
                if(shared>=2)continue;
                for(int j=0;j<r.count;j++)used.insert(r.rows[j]);
                if(sign&&count<max_null)decoy[count]=r.merit;
                if(!sign&&count<cap)out[count]=r;
                count++;
            }
            std::copy(w.count.begin(),w.count.end(),stats+6*sign);
            stats[6*sign+4]=done;stats[6*sign+5]=count;
            if(!sign)result=std::min(cap,count);
        }
        return result;
    }catch(...){return -1;}
}
