// Called on a broad sky cell in a rotating heliocentric frame.  Each other
// position gives a transverse velocity as a function of log range and radial
// speed.  Join the velocities measured on opposite sides of the central time.
namespace rotating {
struct Drift {
    int index;
    double t, x, y, gx, gy, hx, hy, sig, perpendicular, half_width;
};

void drifting_cell(const Source *src,const View *vis,const Params &p,const Cells *xy,
                   const int *order,int count,double east,double north,
                   std::vector<Seed> &tri,std::vector<Seed> &quad) {
    constexpr int max_members=120;
    std::array<Trial,max_members> tr;
    std::array<int,360> entries;
    const int ne=std::min(count,int(entries.size()));
    for(int i=0;i<ne;i++)entries[i]=order[(long long)i*count/ne];
    if(ne>max_members)std::sort(entries.begin(),entries.begin()+ne,[&](int a,int b){
        return src[a].q>src[b].q;
    });
    int n=0;std::array<int,max_members> nights{},multiplicity{};int unique=0;
    for(int j=0;j<ne&&n<max_members;j++){
        int k=entries[j];const auto &s=src[k];const auto &e=vis[s.ex];
        int slot=0;while(slot<unique && nights[slot]!=e.night)slot++;
        if(slot==unique)nights[unique++]=e.night;
        if(++multiplicity[slot]>std::max(3,24/(1+ne/32)))continue;
        tr[n++]={s.row,e.night,s.mag,s.q,e.t,0.,xy[k]};
    }
    if(n<3||unique<3)return;
    std::sort(tr.begin(),tr.begin()+n,[](const Trial &a,const Trial &b){return a.t<b.t;});
    const double circular=arc*kepler/std::pow(p.radius,1.5);
    if(tr[n-1].t-tr[0].t<80.)return;
    auto store=[](std::array<Seed,3> &keep,int &used,Seed v){
        if(used<3){keep[used++]=v;return;}
        int worst=0;for(int j=1;j<3;j++)if(keep[j].score<keep[worst].score)worst=j;
        if(v.score>keep[worst].score)keep[worst]=v;
    };
    std::array<Drift,max_members> past,future;
    for(int a=1;a<n-1;a++){
        const auto &mid=tr[a];
        std::array<Seed,3> three,four;int n3=0,n4=0,np=0,nf=0;double max_width=0.;
        for(int j=0;j<n;j++){
            const auto &s=tr[j];double dt=s.t-mid.t;
            if(std::abs(dt)<2. || std::abs(s.mag-mid.mag)>.94)continue;
            Drift v{j,dt,(s.pos.x-mid.pos.x)/dt,(s.pos.y-mid.pos.y)/dt,
                    (s.pos.gx-mid.pos.gx)/dt,(s.pos.gy-mid.pos.gy)/dt,
                    (s.pos.hx-mid.pos.hx)/dt,(s.pos.hy-mid.pos.hy)/dt};
            v.sig=std::hypot(.47,.055*std::sqrt(std::abs(dt)/10.))/std::abs(dt);
            double along=v.x*east+v.y*north;
            v.perpendicular=v.y*east-v.x*north;
            double wg=std::abs(v.gy*east-v.gx*north),wh=std::abs(v.hy*east-v.hx*north);
            v.half_width=.23*wg+.63*wh+3.6*v.sig;
            if(std::abs(v.perpendicular)>v.half_width+.80*circular||
               std::abs(along)>.24*std::abs(v.gx*east+v.gy*north)+
                               .63*std::abs(v.hx*east+v.hy*north)+.68*circular+3.6*v.sig)continue;
            if(dt<0.)past[np++]=v;
            else{future[nf++]=v;max_width=std::max(max_width,v.half_width);}
        }
        if(!np||!nf)continue;
        std::sort(future.begin(),future.begin()+nf,[](const Drift &v,const Drift &w){
            return v.perpendicular<w.perpendicular;
        });
        for(int j=0;j<np;j++){
            const auto &one=past[j];const auto &left=tr[one.index];
            double reach=one.half_width+max_width;
            auto begin=std::lower_bound(future.begin(),future.begin()+nf,one.perpendicular-reach,
                           [](const Drift &v,double value){return v.perpendicular<value;});
            for(auto it=begin;it!=future.begin()+nf&&it->perpendicular<=one.perpendicular+reach;it++){
                const auto &two=*it;const auto &right=tr[two.index];double span=two.t-one.t;
                if(span<80.||std::abs(one.perpendicular-two.perpendicular)>one.half_width+two.half_width)continue;
                double err=std::hypot(one.sig,two.sig);
                // The same central centroid occurs in both slopes.
                err=std::sqrt(err*err+2.*.20*.20/std::abs(one.t*two.t));
                Constraint c{one.x-two.x,one.y-two.y,one.gx-two.gx,one.gy-two.gy,
                             one.hx-two.hx,one.hy-two.hy,err};
                double cross=c.x*c.gy-c.y*c.gx;
                if(std::abs(cross)>3.8*err*(.02+std::hypot(c.gx,c.gy))+
                                    .72*std::abs(c.hx*c.gy-c.hy*c.gx))continue;
                Normal normal;normal.add(c);
                double u,h,chi,volume;
                if(!normal.solve(u,h,chi,volume)||chi>15.)continue;
                double vx=one.x+u*one.gx+h*one.hx,vy=one.y+u*one.gy+h*one.hy;
                double kinetic=sq((vx*east+vy*north)/(.33*circular))+
                               sq((vy*east-vx*north)/(.50*circular));
                if(kinetic>12.)continue;
                double minmag=std::min({left.mag,mid.mag,right.mag}),maxmag=std::max({left.mag,mid.mag,right.mag});
                double q=left.q+mid.q+right.q;
                double preference=.54*std::log1p(span/80.)-.45*std::log1p(n*.14);
                double merit=.54*q+preference-.44*chi-.38*volume-.27*kinetic-.50*sq((maxmag-minmag)/.34);
                store(three,n3,{{left.row,mid.row,right.row,-1},float(std::exp(-u)/p.radius),float(merit)});
                // Forecast a fourth night from this triple. Use the covariance
                // of its range and radial speed when rejecting distant peaks.
                double det=normal.aa*normal.bb-normal.ab*normal.ab;
                for(int l=0;l<np+nf;l++){
                    const auto &v=l<np?past[l]:future[l-np];const auto &s=tr[v.index];
                    if(s.night==left.night||s.night==right.night)continue;
                    const auto &ref=std::abs(one.t)>two.t?one:two;
                    double dx=v.x-ref.x,dy=v.y-ref.y,gx=v.gx-ref.gx,gy=v.gy-ref.gy;
                    double hx=v.hx-ref.hx,hy=v.hy-ref.hy,noise=v.sig*v.sig+ref.sig*ref.sig;
                    double rx=dx+gx*u+hx*h,ry=dy+gy*u+hy*h;
                    double xx=noise+(normal.bb*gx*gx-2.*normal.ab*gx*hx+normal.aa*hx*hx)/det;
                    double yy=noise+(normal.bb*gy*gy-2.*normal.ab*gy*hy+normal.aa*hy*hy)/det;
                    if(rx*rx>16.*xx||ry*ry>16.*yy)continue;
                    Normal joint=normal;joint.add({dx,dy,gx,gy,hx,hy,std::sqrt(noise)});
                    double gu,gh,cc,vv;
                    if(!joint.solve(gu,gh,cc,vv)||cc>17.)continue;
                    double dm=std::max(s.mag,maxmag)-std::min(s.mag,minmag);
                    double score=.56*(q+s.q)+preference-.42*cc-.36*vv-.27*kinetic-.5*sq(dm/.34);
                    store(four,n4,{{left.row,mid.row,right.row,s.row},float(std::exp(-gu)/p.radius),float(score)});
                }
            }
        }
        tri.insert(tri.end(),three.begin(),three.begin()+n3);quad.insert(quad.end(),four.begin(),four.begin()+n4);
    }
}
} // namespace rotating
