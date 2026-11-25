// At hundreds of au, position changes are measurable across neighbouring
// nights. Choosing such a pair greatly reduces the number of unrelated
// sources encountered in the parallax strip.
namespace annual {
std::vector<std::vector<Path>> short_visits(const Sky &s,const double *depth,
                                           const std::vector<std::vector<Path>> &all){
    auto out=all;
    for(size_t i=0;i<out.size();i++){
        const auto &a=s.images[i];V p{.5*(a.xmin+a.xmax),.5*(a.ymin+a.ymax)};
        for(auto &v:out[i]){
            const auto &b=s.images[v.e];double dt=std::abs(b.exposure.t-a.exposure.t);
            V par=observer(b.exposure,p)-observer(a.exposure,p);double base=std::sqrt(dot(par,par));
            double small=std::min(1.,base/.08),overlap=a.area/b.area;
            v.merit=.53*v.merit+1.5*std::log(.06+small)-1.2*std::log1p(dt/5.)+
                    .21*std::clamp(depth[v.e]-depth[i],-2.,2.)-.1*std::abs(std::log(std::max(.01,overlap)));
        }
        std::sort(out[i].begin(),out[i].end(),[](auto &a,auto &b){return a.merit>b.merit;});
    }
    return out;
}
bool populated(const original::Frame &f,V p){
    int x=int(std::floor(p.x/original::cell))-f.bx,y=int(std::floor(p.y/original::cell))-f.by;
    if(x<1||x+1>=f.nx||y<1||y+1>=f.ny)return false;
    int k=y*f.nx+x,total=0;for(int j=-1;j<=1;j++)total+=f.starts[k+j*f.nx+2]-f.starts[k+j*f.nx-1];
    return total>=3;
}
std::vector<int> rank_visits(const Sky &s,const double *depth,const std::vector<std::vector<Path>> &plan,
                             const int *rows,int nr,Clock::time_point end){
    std::vector<std::vector<int>> repeat(plan.size());
    for(size_t i=0;i<plan.size();i++){
        const auto &a=s.images[i];
        for(size_t j=0;j<plan.size();j++){
            auto &b=s.images[j];
            if(i!=j&&a.exposure.night==b.exposure.night&&std::abs(a.exposure.t-b.exposure.t)<.32)
                repeat[i].push_back(j);
        }
        std::sort(repeat[i].begin(),repeat[i].end(),[&](int l,int r){return depth[l]>depth[r];});
    }
    struct Trial{int row;double merit;};
    std::vector<Trial> trials;trials.reserve(nr);
    std::unordered_set<int> seen;
    for(int j=0;j<nr&&!expired(end);j++){
        int i=rows[j];if(!seen.insert(i).second)continue;
        int a=s.ex[i],nt=0,nights[5],hits=0;double score=0.,mag=s.mag[i];
        V p=position(s,i);
        if(!usable(s,i,mag,.1))continue;
        auto &f=s.images[a];double shape=(s.sharp[i]-1.02)/std::hypot(.18,3.3/s.sn[i]);
        score=.67*std::log(std::min(40.,double(s.sn[i]))/5.)-.29*std::min(18.,shape*shape);
        for(const auto &v:plan[a]){
            auto &b=s.images[v.e];int night=b.exposure.night;
            if(depth[v.e]<mag-.45||std::abs(b.exposure.t-f.exposure.t)<5.||
               std::find(nights,nights+nt,night)!=nights+nt||!populated(b,p))continue;
            V diff=observer(b.exposure,p)-observer(f.exposure,p);
            if(dot(diff,diff)<sq(.10))continue;
            double value;int k=nearest(s,v.e,p,mag,.72,value);
            double prob=.91/(1.+std::exp(std::clamp((mag-depth[v.e]-.05)/.29,-20.,20.)));
            nights[nt++]=night;
            if(k>=0&&value<11.){score-=2.6;hits++;}
            else score-=.59*std::log(1.-prob);
            if(nt==5||hits>=2)break;
        }
        if(hits>=2)continue;
        double best=-10.;int checked=0;
        for(int v:repeat[a]){
            auto &b=s.images[v];if(depth[v]<mag-.46||!populated(b,p))continue;
            V d=observer(b.exposure,p)-observer(f.exposure,p);double dt=b.exposure.t-f.exposure.t;
            double radius=.64+206264.8*std::sqrt(dot(d,d))/480.+.36*std::abs(dt);
            double value;int k=nearest(s,v,p-d*std::sqrt(1./(2650.*260.)),mag,radius,value);
            if(k>=0)best=std::max(best,4.8-.32*value);
            ++checked;if(checked>=4)break;
        }
        if(checked)score+=std::max(-1.6,best);
        trials.push_back({i,score});
    }
    std::stable_sort(trials.begin(),trials.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    std::vector<int> result;result.reserve(trials.size());
    for(auto &v:trials)result.push_back(v.row);
    return result;
}
}
