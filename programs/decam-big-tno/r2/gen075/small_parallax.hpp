// Two offsets relative to an anchor jointly measure sky velocity and
// parallax. Eliminate velocity between the two measurements before
// testing additional visits. The same calculations select either sign.
namespace small_parallax {
using annual::V;
struct Offset {
    int row,night;
    double dt,variance,g,sg,phot;
    V displacement,observer;
};
struct Trial {std::array<int,3> rows;double quality;};

void screen(annual::Worker &positive,annual::Worker &negative,int anchor) {
    const auto &sky=positive.s;
    const int a=sky.ex[anchor];const auto &base=sky.images[a];
    V pos=annual::position(sky,anchor);
    const double mag=sky.mag[anchor],va=annual::variance(sky,anchor);
    if(!std::isfinite(mag)||sky.flag[anchor]&1)return;
    constexpr double gmin=1./110000.,gmax=1./5300.;
    double vmax=std::sqrt(2.*mu*gmax*gmax*gmax)*(1.+dot(pos,pos));
    const auto &plan=positive.plan[a];
    std::vector<Offset> hits;std::vector<int> visits;int stationary=0;
    annual::sense=1.;
    for(auto view:plan){
        const int j=view.e;const auto &im=sky.images[j];const auto &e=im.exposure;
        double dt=e.t-base.exposure.t;
        if(std::abs(dt)>1200. || std::abs(dt)<1.2 || positive.depth[j]<mag-.63 ||
           !annual::covered(im,pos,80.*as) ||
           std::find(visits.begin(),visits.end(),e.night)!=visits.end())continue;
        V B=annual::observer(e,pos)-annual::observer(base.exposure,pos);
        double b2=dot(B,B),b=std::sqrt(b2);
        if(b<.12)continue;
        double vp=vmax*std::abs(dt),width=vp+.87*as;
        double lower=std::min(b*gmin-std::abs(dt)*std::sqrt(2.*mu*gmin*gmin*gmin)*(1.+dot(pos,pos)),
                              b*gmax-vp);
        bool fixed=false;
        visits.push_back(e.night);
        size_t start=hits.size();
        sky.nearby(im,pos.x,pos.y,std::abs(B.x)*gmax+width,std::abs(B.y)*gmax+width,[&](int k) {
            if(!annual::usable(sky,k,mag,1.05))return;
            V d=(annual::position(sky,k)-pos)*(1./as);
            double v=annual::variance(sky,k),dm=sky.mag[k]-mag;
            if(lower>.88*as && dot(d,d)<sq(.45) && std::abs(dm)<.45)fixed=true;
            double estimate=-dot(d,B)/b2,g=std::abs(estimate)*as;
            if(g<.45*gmin || g>1.3*gmax)return;
            double rate=std::sqrt(2.*mu*g*g*g)*(1.+dot(pos,pos))/as;
            double w=std::hypot(.40*rate*dt,std::sqrt(va+v+.045*.045));
            V perp=d+B*estimate;
            if(dot(perp,perp)>sq(3.*w) || dot(d,d)<4.*(va+v))return;
            int mb=std::clamp(int((mag-10.)*4.),1,78);
            double sig=std::sqrt(sq(.16)+sq(1.086/sky.sn[k])+sq(1.086/sky.sn[anchor])+
                                 sq(.22*(sky.lever[j]-sky.lever[a])));
            double phot=-.5*sq(dm/sig)-std::log(sig)-std::log(im.rho[mb]+.00001);
            hits.push_back({k,e.night,dt,v,estimate,w/b,phot,d,B});
        });
        if(fixed && ++stationary>=2)return;
        if(hits.size()>start+9){
            std::partial_sort(hits.begin()+start,hits.begin()+start+9,hits.end(),
                              [](auto &l,auto &r){return l.phot>r.phot;});
            hits.resize(start+9);
        }
        if(visits.size()==9)break;
    }
    std::vector<Trial> proposals[2];
    for(int i=0;i<int(hits.size());++i){
        const auto &u=hits[i];
        for(int j=i+1;j<int(hits.size());++j){
            const auto &v=hits[j];
            if(u.night==v.night || std::abs(v.g-u.g)>3.9*(v.sg+u.sg))continue;
            V B=v.observer*u.dt-u.observer*v.dt;
            V d=u.displacement*v.dt-v.displacement*u.dt;
            double b2=dot(B,B);if(b2<1.e-8)continue;
            double p=dot(d,B)/b2,g=std::abs(p)*as;
            if(!(g>=gmin && g<=gmax))continue;
            double variance=va*sq(v.dt-u.dt)+u.variance*sq(v.dt)+v.variance*sq(u.dt);
            if(p*p*b2<20.*variance)continue;
            V diff=d-B*p;double chi=dot(diff,diff)/variance;
            if(chi>9.)continue;
            double denom=sq(u.dt)+sq(v.dt);
            V motion=((u.displacement+u.observer*p)*u.dt+(v.displacement+v.observer*p)*v.dt)*(1./denom);
            V cov=(u.observer*u.dt+v.observer*v.dt)*(1./denom);
            double err=std::sqrt((2.*va+u.variance+v.variance)/denom+variance*dot(cov,cov)/b2);
            double vesc=std::sqrt(2.*mu*g*g*g)*(1.+dot(pos,pos))/as;
            if(dot(motion,motion)>sq(vesc+2.1*err))continue;
            double quality=u.phot+v.phot-.5*chi-.5*dot(motion,motion)/sq(.65*vesc+err)-
                           std::log(1.+std::sqrt(variance/b2));
            proposals[p<0].push_back({{anchor,u.row,v.row},quality});
        }
    }
    for(int sign=0;sign<2;++sign){
        annual::sense=sign?-1.:1.;
        auto &worker=sign?negative:positive;auto &paths=proposals[sign];
        worker.count[0]+=visits.size();worker.count[1]+=paths.size();
        int cap=std::min(5,int(paths.size()));
        std::partial_sort(paths.begin(),paths.begin()+cap,paths.end(),
                          [](auto &a,auto &b){return a.quality>b.quality;});
        for(int j=0;j<cap;++j){
            auto ids=paths[j].rows;
            // Keep the covered anchor first for the walk over future visits.
            auto keyrows=ids;std::sort(keyrows.begin(),keyrows.end());
            uint64_t key=1469598103934665603ull;
            for(int id:keyrows)key=(key^uint32_t(id))*1099511628211ull;
            if(!worker.visited.insert(key).second)continue;
            annual::State state;
            if(!annual::fit(sky,ids.data(),3,state))continue;
            ++worker.count[2];
            worker.recover(state,ids.data(),(mag+sky.mag[ids[1]]+sky.mag[ids[2]])/3.);
        }
        if(worker.visited.size()>450000)worker.visited.clear();
    }
}
}
