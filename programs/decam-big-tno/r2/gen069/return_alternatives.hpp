// Compare the reported continuation with different continuations through the
// same three or four observed nights, allowing the competitors other dates.
namespace return_alternatives {
using orbit_controls::Query;
using joint_returns::Images;
struct Value {float margin,increment,independent;int measured;};

void compare(Images &field,const std::vector<Sample> &all,const ExactState &full,int one,int two,Value &out){
    field.core.clear();field.peaks.clear();field.views.clear();field.nn=0;
    for(const auto &v:all){
        int n=field.sky.images[field.sky.ex[v.row]].exposure.night;
        if(n==one||n==two)continue;
        if(field.core.size()>=12)return;
        field.core.push_back(v);field.nights[field.nn++]=n;
    }
    if(field.core.size()<3)return;
    std::sort(field.nights,field.nights+field.nn);
    field.nn=std::unique(field.nights,field.nights+field.nn)-field.nights;
    ExactState recovered=full;std::vector<double> residual;
    if(!measured_fit(field.core,recovered,residual,8)||energy(recovered)>2.04)return;
    double V[6][6]{};if(!field.prior(recovered,V))return;
    field.forecast(recovered,V);
    if(field.peaks.size()<2)return;
    std::sort(field.peaks.begin(),field.peaks.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    Kepler orbit(full);std::vector<std::array<double,2>> position(field.views.size());
    for(size_t i=0;i<field.views.size();i++){
        const auto &e=field.sky.images[field.views[i].e].exposure;
        Sample epoch{-1,0.,0.,e.t,e.ox,e.oy,e.oz};double f,s;
        orbit.apparent(epoch,position[i][0],position[i][1],f,s);
    }
    std::vector<joint_returns::Peak> peaks;std::vector<bool> member;
    std::unordered_map<int,int> other;
    bool first=false,second=false;
    for(auto p:field.peaks){
        int n=field.views[p.view].night;auto center=position[p.view];
        double dx=(field.sky.xy[p.row*2]-center[0])/as,dy=(field.sky.xy[p.row*2+1]-center[1])/as;
        bool inside=(sq(dx)+sq(dy))<std::max(sq(.55),9.*p.noise);
        first|=inside&&n==one;second|=inside&&n==two;
        if(!inside && (other[n]++>=8||peaks.size()>=110))continue;
        peaks.push_back(p);member.push_back(inside);
        if(peaks.size()>=144)break;
    }
    if(!first||!second)return;
    double compatible=-100.,competitor=-100.,single=0.,additional=-100.;
    for(int i=0;i<int(peaks.size());i++){
        const auto &p=peaks[i];int a=field.views[p.view].night;
        single=std::max(single,p.merit);
        if(member[i]&&a!=one&&a!=two)additional=std::max(additional,p.merit);
        for(int j=0;j<i;j++){
            const auto &q=peaks[j];if(a==field.views[q.view].night)continue;
            double value=field.pair_odds(q,p);
            if(member[i]&&member[j])compatible=std::max(compatible,value);
            else competitor=std::max(competitor,value);
        }
    }
    if(compatible<=-90.)return;
    out.margin=compatible-std::max(single,competitor);out.increment=compatible-single;
    out.independent=additional;out.measured=1;
}
}

extern "C" int competing_returns(const original::Sky *sky,const original::PairedReturns *rare,
        const double *depth,const orbit_controls::Query *queries,int n,int threads,double seconds,
        return_alternatives::Value *out){
    using namespace return_alternatives;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::vector<std::thread> pool;std::atomic<int> next{0};
        for(int k=0;k<threads;k++)pool.emplace_back([&]{
            Images field{*sky,rare,depth};std::vector<Sample> all;
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=n)break;
                const auto &q=queries[i];if(q.n<5||q.n>18||!(q.gamma>1./8500.&&q.gamma<1./14.))continue;
                all.clear();std::vector<int> nights;ExactState full;full.g=q.gamma;full.ref=0.;
                for(int k=0;k<q.n;k++){
                    int r=q.rows[k];if(r<0||r>=sky->size||!std::isfinite(sky->mag[r]))break;
                    all.push_back(sky->sample(r));full.ref+=all.back().t;
                    nights.push_back(sky->images[sky->ex[r]].exposure.night);
                }
                if(int(all.size())!=q.n)continue;
                std::sort(nights.begin(),nights.end());nights.erase(std::unique(nights.begin(),nights.end()),nights.end());
                int nn=nights.size();if(nn<5||nn>8)continue;
                full.ref/=q.n;full.x=all.front().x;full.y=all.front().y;std::vector<double> res;
                if(!measured_fit(all,full,res,9)||energy(full)>2.04)continue;
                int pairs[][2]={{0,nn-1},{0,1},{nn-2,nn-1},{1,nn-2},{nn/2-1,nn/2}};
                for(int j=0;j<5&&Clock::now()<end;j++)compare(field,all,full,nights[pairs[j][0]],nights[pairs[j][1]],out[i*5+j]);
            }
        });
        for(auto &w:pool)w.join();return std::min(n,next.load());
    }catch(...){return -1;}
}
