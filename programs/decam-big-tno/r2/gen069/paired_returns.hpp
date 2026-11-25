// Sparse measurements in independent shutters of the same return night.
// A long orbit prediction can be precise perpendicular to its error ellipse.
struct PairedReturns {
    struct Pair {int row,mate;};
    struct Visit {
        std::vector<Pair> points;
        std::vector<int> rare;
        std::array<double,80> density{},rare_density{};
    };
    const Sky &sky;
    std::vector<Visit> images;
    std::unordered_map<int,int> mates;
    std::vector<int8_t> marks;
    PairedReturns(const Sky &s,const int *rows,const int *peers,const float *prior,int n):
        sky(s),images(s.images.size()),marks(s.size,-10){
        for(int j=0;j<n;j++){
            int row=rows[j],p=peers[j];
            if(row<0||row>=s.size)continue;
            marks[row]=int8_t(std::round(8.*std::clamp(double(prior[j]),-4.5,5.5)));
            if(prior[j]>1.9 && std::isfinite(s.mag[row])){
                auto &im=images[s.ex[row]];
                im.rare.push_back(row);
                im.rare_density[std::clamp(int((s.mag[row]-10.)*4.),0,79)]++;
            }
            if(p<0||p>=s.size||s.ex[row]==s.ex[p])continue;
            int e=s.ex[row],q=s.ex[p];
            if(s.images[e].exposure.night!=s.images[q].exposure.night||
               !std::isfinite(s.mag[row]+s.mag[p]))continue;
            images[e].points.push_back({row,p});mates.emplace(row,p);
            int bin=std::clamp(int((s.mag[row]-10.)*4.),0,79);
            images[e].density[bin]++;
        }
        for(size_t i=0;i<images.size();i++){
            auto &v=images[i];auto h=v.density;
            for(int k=0;k<80;k++)v.density[k]=(2.*h[k]+h[std::max(0,k-1)]+h[std::min(79,k+1)]+1.)/s.images[i].area;
            std::sort(v.points.begin(),v.points.end(),[&](auto &a,auto &b){return s.xy[2*a.row+1]<s.xy[2*b.row+1];});
            auto singles=v.rare_density;
            for(int k=0;k<80;k++)
                v.rare_density[k]=(2.*singles[k]+singles[std::max(0,k-1)]+singles[std::min(79,k+1)]+1.)/s.images[i].area;
            std::sort(v.rare.begin(),v.rare.end(),[&](int a,int b){return s.xy[2*a+1]<s.xy[2*b+1];});
        }
    }
    double priority(const Certified &orbit)const{
        int n=orbit.count,nd=0,dates[64];double m[64],strength[64],lever=0.;
        for(int k=0;k<n;k++){
            int r=orbit.row[k],e=sky.ex[r],night=sky.images[e].exposure.night;
            lever+=sky.lever[e]/n;m[k]=sky.mag[r];double q=marks[r]/8.;
            int j=int(std::find(dates,dates+nd,night)-dates);
            if(j==nd){dates[nd]=night;strength[nd++]=q;}
            else strength[j]=std::max(strength[j],q);
        }
        if(!nd)return 0.;
        std::sort(m,m+n);double center=.5*(m[(n-1)/2]+m[n/2]);
        double norm=1./sq(.32),dot=0.;
        for(int k=0;k<n;k++){
            int r=orbit.row[k];double l=sky.lever[sky.ex[r]]-lever;
            double w=1./(sq(.19)+sq(1.086/sky.sn[r]));
            norm+=w*l*l;dot+=w*l*(sky.mag[r]-center);
        }
        double loss=0.,slope=dot/norm;
        for(int k=0;k<n;k++){
            int r=orbit.row[k];double dm=sky.mag[r]-center-slope*(sky.lever[sky.ex[r]]-lever);
            loss+=std::min(8.,sq(dm)/(sq(.19)+sq(1.086/sky.sn[r])))/n;
        }
        double score=0.;for(int j=0;j<nd;j++)score+=std::clamp(strength[j]-.5,-3.,3.);
        std::sort(strength,strength+nd);
        return std::clamp(1.10*(score/std::sqrt(nd)+.85*strength[(nd-1)/4])+1.35*(1.-loss),-6.,6.);
    }
    Association predict(int e,const Kepler &orbit,double x,double y,double xx,double xy,double yy,
                        double det,double mag,double color)const{
        const auto &v=images[e];const auto &im=sky.images[e];Association best{-100.,-1,im.exposure.night};
        if(v.points.empty()||det<=0.||xx+yy>8.e7)return best;
        double area=std::sqrt(det);int k=std::clamp(int((mag-10.)*4.),0,79);
        double clutter=area*v.density[k];
        if(area>70000.||clutter>1.25)return best;
        double wy=3.4*std::sqrt(yy)*as,wx=3.4*std::sqrt(xx)*as;
        if(x+wx<im.xmin||x-wx>im.xmax||y+wy<im.ymin||y-wy>im.ymax)return best;
        auto lo=std::lower_bound(v.points.begin(),v.points.end(),y-wy,[&](const Pair &p,double cut){
            return sky.xy[2*p.row+1]<cut;});
        for(auto it=lo;it!=v.points.end()&&sky.xy[2*it->row+1]<=y+wy;++it){
            int row=it->row,mate=it->mate;
            double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
            double chi=(yy*dx*dx-2.*xy*dx*dy+xx*dy*dy)/det;
            if(chi>11.4||sky.flag[row]&1||sky.flag[mate]&1)continue;
            double dm=sky.mag[row]-mag,vm=sq(.21)+sq(1.09/sky.sn[row]);
            int other=sky.ex[mate];
            double dm2=sky.mag[mate]-mag-color*(sky.lever[other]-sky.lever[e]);
            if(dm*dm>10.*vm||dm2*dm2>12.*(vm+.04))continue;
            Sample p=sky.sample(mate);double u,vv,f,s;orbit.apparent(p,u,vv,f,s);
            // The uncertainty of the absolute position is common to the two
            // shutters. Compare their difference at their actual mid-times.
            double mx=(sky.xy[2*mate]-sky.xy[2*row]-u+x)/as;
            double my=(sky.xy[2*mate+1]-sky.xy[2*row+1]-vv+y)/as;
            double dt=p.t-im.exposure.t,vmotion=sq(.16)+sq(1.05/sky.sn[row])+sq(1.05/sky.sn[mate])+sq(6.*dt);
            double mchi=(mx*mx+my*my)/vmotion;
            if(mchi>10.)continue;
            double score=-std::log(std::max(1.e-8,6.2831853*clutter/.19))-
                         .5*chi-.5*dm*dm/vm-.5*std::log(6.2831853*vm)-.30*mchi-.20*dm2*dm2/(vm+.04);
            if(score>best.merit)best={score,row,im.exposure.night};
        }
        return best;
    }
};
