// Posterior predictive sky positions from three measured directions. The
// correlated endpoint astrometry enters both the third and subsequent visits.
struct ReturnHit {
    V delta,u,v;
    double fraction,variance,score;
    int row,night;
};
inline double mahalanobis(V z,double xx,double xy,double yy,double det){
    return (yy*z.x*z.x-2.*xy*z.x*z.y+xx*z.y*z.y)/det;
}
static void three_returns(const Bank &bank,const Triple &seed,std::vector<Seed> &out,RawCore*raw=nullptr){
    int a=seed.rows[0],b=seed.rows[1],c=seed.rows[2];
    const auto &pa=bank.d[a],&pb=bank.d[b],&pc=bank.d[c];
    const auto &A=bank.ex[pa.e],&B=bank.ex[pb.e],&C=bank.ex[pc.e];
    double dt=B.t-A.t,tc=C.t-A.t,fc=tc/dt;
    double sa=sq(bank.noise(a)),sb=sq(bank.noise(b)),sc=sq(bank.noise(c))+.10*.10;
    double vthird=sc+sa*sq(1.-fc)+sb*sq(fc);
    V third{pc.x/rad,pc.y/rad};
    double g=seed.g,h=seed.h,sg=seed.sg,sh=seed.sh;
    if(!(sg>0.&&sh>0.)||std::abs(dt)<.4)return;
    Normal fit;
    for(int i=0;i<3;i++){
        State orbit(pa,pb,A,B,g,h),range(pa,pb,A,B,g+.012*sg,h),radial(pa,pb,A,B,g,h+.012*sh);
        V pos=orbit.position(tc,C),z=third-pos;
        V u=(range.position(tc,C)-pos)*(1./.012),v=(radial.position(tc,C)-pos)*(1./.012);
        fit=Normal{};fit.ax=(seed.g-g)/sg;fit.bx=(seed.h-h)/sh;
        fit.rr=sq(fit.ax)+sq(fit.bx);
        fit.add(u.x,v.x,z.x,1./vthird);fit.add(u.y,v.y,z.y,1./vthird);
        double x=0.,y=0.;double chi=fit.fit(x,y);
        if(!(chi<35.))return;
        g+=sg*std::clamp(x,-2.8,2.8);h+=sh*std::clamp(y,-2.8,2.8);
        if(!(g>1./2400.&&g<1./14.))return;
        if(sq(x)+sq(y)<1.e-7)break;
    }
    State orbit(pa,pb,A,B,g,h),range(pa,pb,A,B,g+.012*sg,h),radial(pa,pb,A,B,g,h+.012*sh);
    if((orbit.v.norm2()+h*h)*std::sqrt(1.+orbit.p.norm2())>2.45*mu*g*g*g)return;
    V pos3=orbit.position(tc,C),z3=third-pos3;
    V uc=(range.position(tc,C)-pos3)*(1./.012),vc=(radial.position(tc,C)-pos3)*(1./.012);
    double invdet=1./(fit.aa*fit.bb-fit.ab*fit.ab);
    double l1=std::sqrt(fit.bb*invdet),l2=-fit.ab*invdet/l1,l3=1./std::sqrt(fit.bb);
    auto mag=bank.phot(a,b);
    if(bank.phot_chi(mag,c)>14.)return;
    std::vector<ReturnHit> hits;int compact=-1;bool two_compact=false;
    for(int e=0;e<bank.ne;e++){
        const auto &E=bank.ex[e];const auto &im=bank.fields[e];
        if(!im.nx||E.night==A.night||E.night==B.night||E.night==C.night)continue;
        double predicted=mag.mag+mag.color*bank.lever[e];
        if(bank.depth[e]+.83<predicted)continue;
        double t=E.t-A.t,f=t/dt;
        V pos=orbit.position(t,E);
        double cross=(sa*(1.-fc)*(1.-f)+sb*fc*f)/vthird;
        V u=(range.position(t,E)-pos)*(1./.012)-uc*cross;
        V v=(radial.position(t,E)-pos)*(1./.012)-vc*cross;
        pos=pos+z3*cross;
        V d1=u*l1+v*l2,d2=v*l3;
        double lin=std::hypot(d1.x,d1.y)+std::hypot(d2.x,d2.y);
        double curvature=std::min(12.,lin*.024*std::min(1.5,std::abs(t)*sh+sg/g));
        double noise=.20*.20+std::max(0.,sa*sq(1.-f)+sb*sq(f)-sq(cross)*vthird)+sq(curvature);
        double xx=d1.x*d1.x+d2.x*d2.x+noise,xy=d1.x*d1.y+d2.x*d2.y,yy=d1.y*d1.y+d2.y*d2.y+noise;
        double det=xx*yy-xy*xy;
        if(!(det>0.)||!im.touches(pos,3.3*std::sqrt(xx),3.3*std::sqrt(yy)))continue;
        double bg=im.density[std::clamp(int((predicted-15.)*4.),0,54)]*std::sqrt(det);
        if(bg>2.5)continue;
        if(raw&&det<30.&&xx+yy<100.&&predicted<bank.depth[e]+.24){
            two_compact|=compact>=0&&compact!=E.night;compact=E.night;
        }
        double depth=.01+1./(1.+std::exp(std::clamp((predicted-bank.depth[e]-.10)/.30,-20.,20.)));
        double logbf=std::log(depth/(.003+6.2831853*bg));
        im.ellipse(pos,xx,xy,yy,3.35,[&](const Point &p,double chi){
            double phot=bank.phot_chi(mag,p.id);if(phot>12.)return;
            double score=logbf-.47*chi-.45*phot+.28*std::clamp(double(p.q),-5.,7.)+.48*(bank.peer[p.id]>=0);
            if(score < -5.2)return;
            hits.push_back({V{p.x,p.y}-pos,d1,d2,f,noise,score,p.id,E.night});
        });
        if(hits.size()>160)best(hits,72);
    }
    best(hits,70);std::vector<Seed> seeds;
    for(auto &x:hits){
        double xx=x.u.x*x.u.x+x.v.x*x.v.x+x.variance,yy=x.u.y*x.u.y+x.v.y*x.v.y+x.variance;
        double xy=x.u.x*x.u.y+x.v.x*x.v.y,det=xx*yy-xy*xy;
        if(!(det>0.))continue;
        double mx=(yy*x.delta.x-xy*x.delta.y)/det,my=(xx*x.delta.y-xy*x.delta.x)/det;
        double a1=x.u.x*mx+x.u.y*my;
        double truthg=g+sg*l1*a1;
        if(!(truthg>1./2400.&&truthg<1./14.))continue;
        int witness=-1;double gain=0.;
        double kx=sa*(1.-fc)*(1.-x.fraction)+sb*fc*x.fraction;
        for(auto &y:hits){
            if(y.night==x.night)continue;
            double ky=sa*(1.-fc)*(1.-y.fraction)+sb*fc*y.fraction;
            double corr=sa*(1.-x.fraction)*(1.-y.fraction)+sb*x.fraction*y.fraction-kx*ky/vthird;
            double cxx=y.u.x*x.u.x+y.v.x*x.v.x+corr,cxy=y.u.x*x.u.y+y.v.x*x.v.y;
            double cyx=y.u.y*x.u.x+y.v.y*x.v.x,cyy=y.u.y*x.u.y+y.v.y*x.v.y+corr;
            double w1=(yy*cxx-xy*cxy)/det,w2=(xx*cxy-xy*cxx)/det;
            double w3=(yy*cyx-xy*cyy)/det,w4=(xx*cyy-xy*cyx)/det;
            double px=y.u.x*y.u.x+y.v.x*y.v.x+y.variance;
            double py=y.u.y*y.u.y+y.v.y*y.v.y+y.variance;
            double pxy=y.u.x*y.u.y+y.v.x*y.v.y,original=px*py-pxy*pxy;
            px-=w1*cxx+w2*cxy;py-=w3*cyx+w4*cyy;pxy-=w1*cyx+w2*cyy;
            double d=px*py-pxy*pxy;if(!(d>0.&&px>0.&&py>0.))continue;
            V residual=y.delta-V{cxx*mx+cxy*my,cyx*mx+cyy*my};
            double chi=mahalanobis(residual,px,pxy,py,d);
            if(chi>14.)continue;
            double prior_chi=mahalanobis(y.delta,y.u.x*y.u.x+y.v.x*y.v.x+y.variance,
                y.u.x*y.u.y+y.v.x*y.v.y,y.u.y*y.u.y+y.v.y*y.v.y+y.variance,original);
            double val=2.5+y.score+.45*(prior_chi-chi)+.46*std::log(original/d);
            if(val>gain){gain=val;witness=y.row;}
        }
        seeds.push_back({{a,b,c,x.row,witness},float(truthg),float(.65*seed.score+x.score+gain)});
    }
    best(seeds,10);out.insert(out.end(),seeds.begin(),seeds.end());
    if(raw&&two_compact){
        double state[]={orbit.p.x,orbit.p.y,g,orbit.v.x,orbit.v.y,h};
        std::copy_n(state,6,raw->x);raw->ref=A.t-orbit.home_distance/173.1446327;
        bool witnessed=std::any_of(seeds.begin(),seeds.end(),[](const Seed&s){return s.rows[4]>=0;});
        raw->merit=seed.score-(witnessed?4.:0.);raw->count=3;
        std::copy_n(seed.rows,3,raw->rows);
    }
}
