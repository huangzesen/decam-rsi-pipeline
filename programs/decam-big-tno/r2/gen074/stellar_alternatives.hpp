// Fit one or two nearby stellar positions and proper motions to the claimed
// members. Unrelated combinations of two recurring stars can have a very
// significant parallax when tested only against one stationary source.
struct Moments {
    int n=0;
    double w=0.,t=0.,tt=0.,pp=0.;
    Vec p{},tp{};
    void add(double weight,double dt,Vec d){
        ++n;w+=weight;t+=weight*dt;tt+=weight*dt*dt;pp+=weight*dot(d,d);
        p=p+d*weight;tp=tp+d*(weight*dt);
    }
    double loss()const{
        if(!n)return 0.;
        Vec rate=tp-p*(t/w);
        // Ordinary high-proper-motion stars are uncommon. Marginalize
        // over a broad slope in arcseconds per year for each component.
        double lever=std::max(0.,tt-t*t/w);
        return std::max(0.,pp-dot(p,p)/w-dot(rate,rate)/(lever+6.25))
               +2.*std::log1p(lever/6.25);
    }
};
double stellar_contrast(const std::vector<Point> &p,const std::vector<int> &ids,const Orbit &s){
    int n=ids.size();
    std::vector<Vec> d(n);std::vector<double> t(n),w(n);
    std::vector<int> order(n);std::vector<Moments> suffix(n+1);
    double orbit=0.;Moments whole;
    for(int k=0;k<n;k++){
        const auto &q=p[ids[k]];
        d[k]=q.p-s.p;t[k]=q.t-s.t;w[k]=1./(q.var+sq(.095));
        whole.add(w[k],t[k],d[k]);
        Vec residual=d[k]-(s.v*t[k]-q.obs*s.g+s.offset[q.band]);
        orbit+=w[k]*dot(residual,residual);order[k]=k;
    }
    double best=whole.loss();
    for(auto axis:std::array<Vec,6>{{{1.,0.},{.866,.5},{.5,.866},{0.,1.},{-.5,.866},{-.866,.5}}}){
        std::sort(order.begin(),order.end(),[&](int a,int b){return dot(d[a],axis)<dot(d[b],axis);});
        suffix[n]={};
        for(int k=n-1;k>=0;k--){
            int i=order[k];suffix[k]=suffix[k+1];suffix[k].add(w[i],t[i],d[i]);
        }
        Moments left;
        for(int k=0;k<n-1;k++){
            int i=order[k];left.add(w[i],t[i],d[i]);
            auto &right=suffix[k+1];
            // Integrate the unknown mixing fraction for the two sources.
            double mix=2.*(left.n*std::log(double(n)/left.n)+right.n*std::log(double(n)/right.n));
            best=std::min(best,left.loss()+right.loss()+mix+2.*std::log1p(n/3.));
        }
    }
    return std::sqrt(std::max(0.,best-orbit));
}
