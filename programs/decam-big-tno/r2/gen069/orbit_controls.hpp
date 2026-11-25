// Evaluate independent-night predictions against entire displaced tracks.
// Every control retains the orbit, shutter times, local fields, flux scale
// and search over possible confirmation dates of the original hypothesis.
namespace orbit_controls {
constexpr int trials=40, maxrows=18;
struct Query {int rows[maxrows],n;float gamma;};
struct Result {float score,tail,contrast,available,expected,alternatives;int checked,omitted;};
struct Peak {double value=-50.;int night=-1;};
struct BestTwo {
    Peak first,second;
    void add(double q,int night){
        if(q>first.value){
            if(first.night!=night)second=first;
            first={q,night};
        }else if(night!=first.night&&q>second.value)second={q,night};
    }
    double value(int n)const{return n==1?first.value:first.value+second.value;}
};
struct Flux {
    double mid,lever,color,xx,xy,yy;
    Flux(const original::Sky &sky,const std::vector<Sample> &core){
        double w=0;mid=lever=0.;
        for(auto &p:core){
            double wt=1./(sq(.19)+sq(1.086/sky.sn[p.row]));
            mid+=wt*sky.mag[p.row];lever+=wt*sky.lever[sky.ex[p.row]];w+=wt;
        }
        mid/=w;lever/=w; color=0.;xx=1./w;xy=0.;yy=0.;
        double prior=1./sq(.29);
        for(int it=0;it<3;it++){
            double aa=0.,ab=0.,bb=prior,am=0.,bm=0.;
            for(auto &p:core){
                double l=sky.lever[sky.ex[p.row]]-lever,m=sky.mag[p.row]-mid;
                double var=sq(.155)+sq(1.086/sky.sn[p.row]);
                double wt=1./(var+.25*sq(m-color*l));
                aa+=wt;ab+=wt*l;bb+=wt*l*l;am+=wt*m;bm+=wt*m*l;
            }
            double det=aa*bb-ab*ab;
            xx=bb/det;xy=-ab/det;yy=aa/det;
            mid+=xx*am+xy*bm;color=std::clamp(xy*am+yy*bm,-.62,.66);
        }
    }
};
struct Aperture {
    const original::Sky &sky;
    const original::Frame &im;
    double xx,xy,yy,det,logarea,mag,var,detect;
    int band;
    double score(int row,double x,double y)const{
        if(sky.flag[row]&1||!std::isfinite(sky.mag[row]))return -50.;
        double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
        double chi=(yy*dx*dx-2.*xy*dx*dy+xx*dy*dy)/det;
        if(chi>9.65)return -50.;
        double dm=sky.mag[row]-mag,variance=var+sq(1.086/std::max(4.5f,sky.sn[row]));
        if(dm*dm>36.*(variance+.07))return -50.;
        double phot=.85*std::exp(-.5*dm*dm/variance)/std::sqrt(variance)
                   +.15*std::exp(-.5*dm*dm/(variance+.11))/std::sqrt(variance+.11);
        int b=std::clamp(int((sky.mag[row]-10.)*4.),0,79);
        double density=im.rho[b];
        if(sky.population){
            // A genuine object can be lost to static subtraction. Assign
            // positive support to each level of the proposal catalogue.
            int tier=sky.population[row];
            double levelfreq=tier==2?im.transient[1][b]:
                tier==1?im.transient[0][b]-im.transient[1][b]:im.rho[b]-im.transient[0][b];
            double signal=tier==2?.60:tier==1?.18:.22;
            density=.15*density+.85*std::max(2.e-7,levelfreq)/signal;
        }
        double shape=std::min(12.,sq((sky.sharp[row]-1.05)/std::hypot(.205,3.65/sky.sn[row])));
        return std::log(std::max(1.e-15,detect*phot/(2.50663*std::max(1.e-7,density))))-
               logarea-.5*chi-.22*shape-.58*((sky.flag[row]&6)!=0);
    }
    double at(double x,double y)const{
        double v=-50.;
        // A broad wing has little power against a catalogue containing
        // many stars. Use the same 3.1-sigma search on data and controls.
        sky.ellipse(im,x,y,xx,xy,yy,[&](int row){
            double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
            if(yy*dx*dx-2.*xy*dx*dy+xx*dy*dy<=9.65*det)v=std::max(v,score(row,x,y));
        });
        return v;
    }
};
bool measure(const original::Sky &sky,const double *depth,const Query &q,const int *night,
        int nnight,unsigned mask,Result &out,const double *offsets){
    std::vector<Sample> core;std::vector<int> held;
    ExactState o;o.ref=0.;o.g=q.gamma;
    for(int i=0;i<q.n;i++){
        int k=q.rows[i],n=sky.images[sky.ex[k]].exposure.night;
        int j=int(std::lower_bound(night,night+nnight,n)-night);
        if(mask&(1u<<j))held.push_back(k);
        else{core.push_back(sky.sample(k));o.ref+=core.back().t;}
    }
    if(core.size()<3||held.empty())return false;
    o.ref/=core.size();o.x=core[0].x;o.y=core[0].y;
    std::vector<double> res;
    if(!measured_fit(core,o,res,8)||*std::max_element(res.begin(),res.end())>.95)return false;
    Kepler orbit(o);original::OrbitDerivatives der(orbit);
    double normal[6][6]{},cov[6][6],x,y,f,s,anom[3],J[2][6];
    normal[0][0]=normal[1][1]=1.e-12;
    normal[2][2]=4.;normal[3][3]=normal[4][4]=4.;normal[5][5]=9.;
    for(size_t i=0;i<core.size();i++){
        auto &p=core[i];
        orbit.apparent(p,x,y,f,s,anom);der.design(p,x,y,f,s,anom,J);
        double var=sq(.104)+sq(1.02/sky.sn[p.row])+.28*sq(res[i]);
        int others=0;
        for(auto &b:core)others+=std::abs(b.t-p.t)<.48;
        double wt=1./(var*(1.+.35*std::max(0,others-1)));
        for(int k=0;k<6;k++)for(int l=0;l<6;l++)normal[k][l]+=wt*(J[0][k]*J[0][l]+J[1][k]*J[1][l]);
    }
    if(!original::invert(normal,cov))return false;
    Flux flux(sky,core);
    BestTwo null[trials],real;
    double available=0.,chance=0.,encounters=0.;
    bool supported=false;unsigned accessible=0;
    for(int e=0;e<int(sky.images.size());e++){
        const auto &im=sky.images[e];const auto &v=im.exposure;
        if(!im.nx)continue;
        const int *j=std::lower_bound(night,night+nnight,v.night);
        if(j!=night+nnight&&*j==v.night&&!(mask&(1u<<(j-night))))continue;
        double t=v.t-o.ref,z=mu*o.g*o.g*o.g*t*t;
        f=1.-.5*z*(1.-o.h*t);s=t*(1.-z/6.);
        double inv=1./(f+s*o.h-o.g*v.oz);
        x=(f*o.x+s*o.vx-o.g*v.ox)*inv;y=(f*o.y+s*o.vy-o.g*v.oy)*inv;
        double pad=.0028+.13*z*z;
        if(x+pad<im.xmin||x-pad>im.xmax||y+pad<im.ymin||y-pad>im.ymax)continue;
        Sample p{-1,0.,0.,v.t,v.ox,v.oy,v.oz};
        orbit.apparent(p,x,y,f,s,anom);der.design(p,x,y,f,s,anom,J);
        double xx,xy,yy;original::projected(J,cov,xx,xy,yy);
        double det=xx*yy-xy*xy;
        if(!(det>0.)||xx+yy>38000.)continue;
        double area=std::sqrt(det),l=sky.lever[e]-flux.lever,mag=flux.mid+flux.color*l;
        int b=std::clamp(int((mag-10.)*4.),0,79);
        if(area*im.rho[b]>7. || area>2000.)continue;
        double detect=std::isfinite(depth[e])?.04+.91/(1.+std::exp(std::clamp((mag-depth[e]-.10)/.29,-20.,20.))):.50;
        if(detect<.065)continue;
        Aperture a{sky,im,xx,xy,yy,det,std::log(6.2831853*area),mag,
                   sq(.155)+std::max(.003,flux.xx+2.*flux.xy*l+flux.yy*l*l),detect,b};
        // All offset lookups use the same material CCD coverage as a
        // real measured fifth date. Missing pointings contribute no trials.
        auto live=[&](double px,double py){
            int ix=int(std::floor(px/original::cell))-im.bx,iy=int(std::floor(py/original::cell))-im.by;
            if(ix<1||ix+1>=im.nx||iy<1||iy+1>=im.ny)return false;
            int k=iy*im.nx+ix;
            return im.starts[k+2]-im.starts[k-1]>0&&
                   im.starts[k+im.nx+1]-im.starts[k+im.nx]+im.starts[k-im.nx+1]-im.starts[k-im.nx]>0;
        };
        if(!live(x,y))continue;
        double held_best=-50.;
        for(int r:held)if(sky.ex[r]==e){
            accessible|=1u<<(j-night);
            held_best=std::max(held_best,a.score(r,x,y));
        }
        if(held_best> -45.){real.add(held_best,v.night);supported=true;}
        int checked=0;
        for(int k=0;k<trials;k++){
            double px=x+offsets[2*k],py=y+offsets[2*k+1];
            if(!live(px,py))continue;
            null[k].add(a.at(px,py),v.night);++checked;
        }
        available+=checked/double(trials);chance+=area*im.rho[b]*detect;encounters+=detect;
    }
    if(available<.5 || (accessible&mask)!=mask)return false;
    int n=__builtin_popcount(mask);double actual=real.value(n);
    if(!supported)actual=-50.;
    std::array<double,trials> ctl{};
    int above=0;
    for(int k=0;k<trials;k++){
        ctl[k]=null[k].value(n);above+=ctl[k]>=actual;
    }
    std::sort(ctl.begin(),ctl.end());
    out.score=std::clamp(actual,-20.,40.);
    out.tail=(above+.5)/(trials+1.);
    // Compare to a real upper quantile of the best peaks on random orbits.
    // Censor empty apertures to avoid arbitrarily large evidence.
    out.contrast=std::clamp(actual-std::max(-3.,ctl[trials*4/5]),-16.,16.);
    out.available=available;out.expected=chance;out.alternatives=encounters;out.checked=1;out.omitted=n;
    return true;
}
}
extern "C" int displaced_confirmations(const original::Sky *sky,const double *depth,
        const orbit_controls::Query *queries,int count,int threads,double seconds,
        orbit_controls::Result *out){
    using namespace orbit_controls;
    try{
        // The independent controls use the remaining wall time after the
        // proposal workers have stopped, so this deadline is separate.
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int th=0;th<threads;th++)workers.emplace_back([&]{
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=count)break;
                const auto &q=queries[i];if(q.n<5||q.n>maxrows||!(q.gamma>1./8500.&&q.gamma<1./14.))continue;
                int nights[maxrows],nt=0;bool ok=true;
                for(int k=0;k<q.n;k++){
                    int row=q.rows[k];if(row<0||row>=sky->size||!std::isfinite(sky->mag[row])){ok=false;break;}
                    nights[nt++]=sky->images[sky->ex[row]].exposure.night;
                }
                if(!ok)continue;
                std::sort(nights,nights+nt);nt=std::unique(nights,nights+nt)-nights;
                if(nt<5||nt>8)continue;
                double offsets[2*trials];unsigned hash=unsigned(q.rows[0])*2654435761u;
                double phase=6.2831853*hash/4294967296.;
                for(int k=0;k<trials;k++){
                    double radius=as*std::sqrt(sq(43.)+(sq(325.)-sq(43.))*(k+.5)/trials);
                    double angle=phase+k*2.3999632297;
                    offsets[2*k]=radius*std::cos(angle);offsets[2*k+1]=radius*std::sin(angle);
                }
                for(int k=0;k<nt;k++){
                    if(Clock::now()>=end)break;
                    measure(*sky,depth,q,nights,nt,1u<<k,out[i*12+k],offsets);
                }
                // Check two remote observations jointly when either could
                // merely be the best match to a shorter measured mover.
                if(nt>=6)for(int j=0;j<3&&Clock::now()<end;j++){
                    int first=j%(nt/2),last=nt-1-(j/2);
                    measure(*sky,depth,q,nights,nt,(1u<<first)|(1u<<last),out[i*12+8+j],offsets);
                }
            }
        });
        for(auto &w:workers)w.join();
        return std::min(count,next.load());
    }catch(...){return -1;}
}
