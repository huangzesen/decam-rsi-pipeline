// Anchor priority for very slow tracks that resemble fixed stars in a short
// observing burst. Evaluate the stationary explanation in other deep images.
extern "C" int parallax_anchor_marks(const original::Sky *sky,const int *rows,int n,
                      const double *depth,int threads,double seconds,float *out){
    try{
        using namespace annual;
        const auto &s=*sky;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        auto plans=annual::plans(s,depth);
        std::vector<std::vector<int>> repeats(s.images.size());
        for(int e=0;e<int(s.images.size());e++){
            const auto &a=s.images[e];if(!a.nx)continue;
            for(int j=0;j<int(s.images.size());j++){
                const auto &b=s.images[j];
                if(e==j||!b.nx||b.exposure.night!=a.exposure.night||
                   a.xmin>b.xmax||a.xmax<b.xmin||a.ymin>b.ymax||a.ymax<b.ymin)continue;
                repeats[e].push_back(j);
            }
            std::sort(repeats[e].begin(),repeats[e].end(),[&](int i,int j){return depth[i]>depth[j];});
        }
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int k=0;k<threads;k++)pool.emplace_back([&]{
            while(!expired(end)){
                int first=next.fetch_add(64);if(first>=n)break;
                for(int j=first;j<std::min(first+64,n);j++){
                    int row=rows[j],e=s.ex[row];const auto &a=s.images[e];
                    double mag=s.mag[row];V p=position(s,row);
                    if(!usable(s,row,mag,1.))continue;
                    int days[6],tested=0,inspected=0;double absent=0.,fixed=0.,repeat=0.;
                    for(auto v:plans[e]){
                        const auto &b=s.images[v.e];auto day=b.exposure.night;
                        if(++inspected>70||tested>=6)break;
                        if(depth[v.e]<mag+.08||!covered(b,p,.0)||
                           std::find(days,days+tested,day)!=days+tested)continue;
                        V diff=observer(b.exposure,p)-observer(a.exposure,p);
                        if(dot(diff,diff)<.11||std::abs(b.exposure.t-a.exposure.t)>1000.)continue;
                        // Estimate usable coverage from nearby measurements,
                        // independently of a detection at the target centroid.
                        bool live=false;double best=99.;
                        s.nearby(b,p.x,p.y,18.*as,18.*as,[&](int i){
                            V d=(position(s,i)-p)*(1./as);double r2=dot(d,d);
                            if(r2>324.)return;
                            if(r2>1.&&!(s.flag[i]&5)){
                                double edge=std::min({double(s.xp[i]),2048.-s.xp[i],double(s.yp[i]),4096.-s.yp[i]});
                                if(edge>12.+3.95*std::sqrt(r2))live=true;
                            }
                            if(r2<.68*.68&&usable(s,i,mag,.83))
                                best=std::min(best,r2/(variance(s,i)+variance(s,row))+.6*sq((s.mag[i]-mag)/.33));
                        });
                        if(!live)continue;
                        days[tested++]=day;
                        if(best<8.)fixed+=std::exp(-.13*best);
                        else absent+=std::clamp(.55+.7*(depth[v.e]-mag),.45,1.4);
                        if(fixed>1.7)break;
                    }
                    if(fixed<1.7&&absent>.4){
                        int checked=0;
                        for(int k:repeats[e]){
                            if(depth[k]<mag-.4||!covered(s.images[k],p,.0))continue;
                            double c;const double dt=std::abs(s.images[k].exposure.t-a.exposure.t);
                            int i=nearest(s,k,p,mag,.50+1.7*dt,c);
                            if(i>=0&&c<10.&&std::abs(s.mag[i]-mag)<.64)
                                repeat=std::max(repeat,std::exp(-.12*c));
                            if(++checked>=8||repeat>.82)break;
                        }
                    }
                    out[j]=.62*std::min(5.,absent)-2.8*fixed+3.4*repeat*std::min(1.,absent/2.);
                }
            }
        });
        for(auto &p:pool)p.join();
        return std::min(n,next.load());
    }catch(...){return -1;}
}
