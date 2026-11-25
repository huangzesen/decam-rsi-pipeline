// Field trials for a held-out date. Compare its measured likelihood with the
// strongest detection a blind continuation could have selected in any visit.
namespace confirmation_trials {
using original::Frame;
using original::Sky;
constexpr int max_nights=12, controls=16;
struct Report {double gamma;int n,rows[max_nights],pad;};
struct Result {float support[max_nights],false_alarm[max_nights],controls_hit[max_nights],error[max_nights];};
struct View {int e;double x,y,xx,xy,yy,det,mag,sm,peak,detect;};
struct Phot {double mean,color,lever,var_mean,var_color,scatter;};

Phot photometry(const Sky &sky,const std::vector<Sample> &obs){
    double vals[max_nights],w[max_nights],total=0.,level=0.,lever=0.;
    for(size_t i=0;i<obs.size();++i)vals[i]=sky.mag[obs[i].row];
    std::sort(vals,vals+obs.size());
    double median=.5*(vals[(obs.size()-1)/2]+vals[obs.size()/2]);
    for(size_t i=0;i<obs.size();++i){
        int r=obs[i].row;double m=sky.mag[r]+1.3/sq(sky.sn[r]);
        w[i]=1./(.027+sq(1.09/sky.sn[r])+.18*sq(m-median));
        total+=w[i];lever+=w[i]*sky.lever[sky.ex[r]];level+=w[i]*m;
    }
    level/=total;lever/=total;
    double normal=1./sq(.28),rhs=0.;
    for(size_t i=0;i<obs.size();++i){
        int r=obs[i].row;double l=sky.lever[sky.ex[r]]-lever;
        normal+=w[i]*l*l;rhs+=w[i]*l*(sky.mag[r]-level);
    }
    double color=std::clamp(rhs/normal,-.55,.6),scatter=0.;
    for(auto &v:obs){
        double err=sky.mag[v.row]-level-color*(sky.lever[sky.ex[v.row]]-lever);
        scatter+=std::min(.36,err*err)-sq(1.09/sky.sn[v.row]);
    }
    scatter=(scatter+6.*sq(.16))/(6.+obs.size()-1);
    return {level,color,lever,1./total,1./normal,std::clamp(scatter,.012,.14)};
}

bool make_view(const Sky &sky,int e,const double *depth,const ExactState &o,const Kepler &orbit,
               double V[6][6],const Phot &phot,View &v){
    const auto &im=sky.images[e];const auto &z=im.exposure;
    if(!im.nx)return false;
    Sample p{-1,0.,0.,z.t,z.ox,z.oy,z.oz};
    double x,y,f,s,J[2][6]{},xx,xy,yy;
    orbit.apparent(p,x,y,f,s);
    if(x<im.xmin-.003||x>im.xmax+.003||y<im.ymin-.003||y>im.ymax+.003)return false;
    original::design(o,p,x,y,f,s,J);original::projected(J,V,xx,xy,yy);
    double det=xx*yy-xy*xy;
    if(!(det>0.)||xx+yy>sq(1000.))return false;
    double l=sky.lever[e]-phot.lever,m=phot.mean+phot.color*l;
    double expected=5.*std::exp(std::clamp(.921034*(depth[e]-m),-8.,8.));
    double vis=.008+.966*.5*std::erfc((4.65-expected)/std::hypot(1.6,.24*expected));
    double sm=std::sqrt(phot.scatter+phot.var_mean+l*l*phot.var_color+sq(1.086/std::max(4.7,expected)));
    int bin=std::clamp(int((m-10.)*4.),1,78);
    double rho=std::max(1.e-7,im.rho[bin]);
    double peak=std::log(vis/(15.74961*std::sqrt(det)*sm*rho));
    v={e,x,y,xx,xy,yy,det,m,sm,peak,vis};
    return true;
}

double astrom(const View &v,double x,double y){
    return (v.yy*x*x-2.*v.xy*x*y+v.xx*y*y)/v.det;
}

struct Trials {double count=0.,largest=0.,power=0.;};
void omitted(const Sky &sky,const double *depth,const Report &report,int held,Result &out,int nn){
    std::vector<Sample> core;
    std::vector<bool> fixed(nn,false);
    int row=report.rows[held];
    ExactState o;o.g=report.gamma;
    for(int i=0;i<report.n;i++)if(i!=held){
        auto p=sky.sample(report.rows[i]);core.push_back(p);
        o.ref+=p.t;o.x+=p.x;o.y+=p.y;fixed[sky.images[sky.ex[p.row]].exposure.night]=true;
    }
    o.ref/=core.size();o.x/=core.size();o.y/=core.size();
    std::vector<double> res;
    if(!measured_fit(core,o,res,9))return;
    if(*std::max_element(res.begin(),res.end())>1.)return;
    Kepler orbit(o);Phot phot=photometry(sky,core);
    double N[6][6]{},V[6][6];
    N[0][0]=N[1][1]=1.e-10;N[2][2]=2.;N[3][3]=N[4][4]=4.;N[5][5]=7.;
    for(auto &p:core){
        double x,y,f,s,J[2][6]{};
        orbit.apparent(p,x,y,f,s);original::design(o,p,x,y,f,s,J);
        double variance=sq(.075)+.04/p.weight;
        for(int a=0;a<6;a++)for(int b=0;b<6;b++)
            N[a][b]+=(J[0][a]*J[0][b]+J[1][a]*J[1][b])/variance;
    }
    if(!original::invert(N,V))return;
    View test;
    if(!make_view(sky,sky.ex[row],depth,o,orbit,V,phot,test))return;
    double dx=(sky.xy[2*row]-test.x)/as,dy=(sky.xy[2*row+1]-test.y)/as;
    double chi=astrom(test,dx,dy);
    double dm=(sky.mag[row]-test.mag)/test.sm;
    // A small broad centroid component also covers stars underneath a mover.
    double z=test.peak+std::log(.94*std::exp(-.5*std::min(120.,chi))+
                   .06/6.25*std::exp(-.08*std::min(750.,chi)))-.5*std::min(64.,dm*dm);
    double delta[controls][2];
    for(int k=0;k<controls;k++){
        double angle=(k+1)*2.399963229728653+std::fmod(report.rows[0]*.61803399,1.)*6.2831853;
        double radius=std::sqrt(18.*18.+(k+.5)/controls*(69.*69.-18.*18.));
        delta[k][0]=radius*std::cos(angle);delta[k][1]=radius*std::sin(angle);
    }
    unsigned matches=0;
    std::vector<Trials> nights(nn);
    const double survey_area=3.14159265*sq(86.);
    for(int e=0;e<int(sky.images.size());e++){
        auto &im=sky.images[e];int night=im.exposure.night;
        if(fixed[night])continue;
        View v;
        if(!make_view(sky,e,depth,o,orbit,V,phot,v))continue;
        double margin=v.peak-z;
        if(margin<=0.)continue;
        double max_chi=2.*margin,matched=0.,nearest=1.e12;
        int local=0;bool interior=false;unsigned hit=0;
        // Integrate each actual peak's acceptance area over translated
        // trajectories. This estimates rare coincidences even when none
        // of the explicitly displaced control tracks happens to hit.
        sky.nearby(im,v.x,v.y,86.*as,86.*as,[&](int r){
            double x=(sky.xy[2*r]-v.x)/as,y=(sky.xy[2*r+1]-v.y)/as;
            double rr=x*x+y*y;if(rr>sq(86.))return;
            ++local;
            if(rr<nearest){
                nearest=rr;
                double edge=std::min({double(sky.xp[r]),2048.-sky.xp[r],double(sky.yp[r]),4096.-sky.yp[r]});
                interior=edge>8.+3.95*std::sqrt(rr)&&!(sky.flag[r]&4);
            }
            if(!std::isfinite(sky.mag[r])||(sky.flag[r]&1)||r==row)return;
            double psf=(sky.sharp[r]-1.)/std::hypot(.16,3.5/std::max(4.5,double(sky.sn[r])));
            if(std::abs(psf)>3.3||sky.sharp[r]==0.)return;
            double budget=max_chi-sq((sky.mag[r]-v.mag)/v.sm);
            if(budget<=0.)return;
            matched+=budget;
            double bound=budget*(v.xx+v.yy);
            for(int k=0;k<controls;k++){
                if((hit|matches)&(1u<<k))continue;
                double a=x-delta[k][0],b=y-delta[k][1];
                if(a*a+b*b<bound&&astrom(v,a,b)<budget)hit|=1u<<k;
            }
        });
        if(!local||(!interior&&nearest>sq(12.)))continue;
        matches|=hit;
        // Shrink the local count towards the smooth empirical magnitude
        // distribution of this visit when few neighbouring rows contribute.
        double expected=0.;
        int lo=std::clamp(int((v.mag-v.sm*std::sqrt(max_chi)-10.)*4.),0,79);
        int hi=std::clamp(int((v.mag+v.sm*std::sqrt(max_chi)-10.)*4.),0,79);
        for(int b=lo;b<=hi;b++){
            double m=10.+.25*(b+.5);
            expected+=.25*im.rho[b]*std::max(0.,max_chi-sq((m-v.mag)/v.sm));
        }
        double lambda=3.14159265*std::sqrt(v.det)*(.88*matched/survey_area+.12*expected);
        auto &n=nights[night];n.count+=lambda;n.largest=std::max(n.largest,lambda);
        double u=std::sqrt(margin);
        double power=v.detect*(std::erf(u)-1.12837917*u*std::exp(-margin));
        n.power=std::max(n.power,power);
    }
    double lambda=0.,log_miss=0.;
    for(auto &n:nights){
        // Repeated images of the same source in one night are correlated.
        lambda+=n.largest+.45*(n.count-n.largest);
        log_miss+=std::log1p(-std::min(.995,n.power));
    }
    int hits=0;for(int k=0;k<controls;k++)hits+=(matches>>k)&1u;
    double poisson=-std::expm1(-std::min(100.,lambda));
    double pfalse=(22.*poisson+hits)/(22.+controls);
    double ptrue=-std::expm1(log_miss);
    if(ptrue<.00001)return;
    out.support[held]=std::log((ptrue+.0001)/(pfalse+.00001));
    out.false_alarm[held]=pfalse;out.controls_hit[held]=float(hits)/controls;out.error[held]=chi;
}
}

extern "C" int confirmation_trials_run(const original::Sky *sky,const confirmation_trials::Report *reports,
    int n,const double *depth,int threads,double seconds,confirmation_trials::Result *out){
    using namespace confirmation_trials;
    try{
        int nn=0;for(auto &v:sky->images)nn=std::max(nn,v.exposure.night+1);
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int j=0;j<threads;j++)pool.emplace_back([&]{
            // Final validation follows the proposal-stop signal and has its
            // own deadline.
            while(Clock::now()<end){
                int k=next.fetch_add(1);if(k>=n)break;
                const auto &r=reports[k];
                if(r.n<5||r.n>max_nights||!(r.gamma>1./9000.&&r.gamma<1./13.))continue;
                bool valid=true;
                for(int i=0;i<r.n;i++)valid=valid&&r.rows[i]>=0&&r.rows[i]<sky->size&&std::isfinite(sky->mag[r.rows[i]]);
                if(!valid)continue;
                for(int i=0;i<r.n;i++)omitted(*sky,depth,r,i,out[k],nn);
            }
        });
        for(auto &t:pool)t.join();
        return std::min<int>(n,next);
    }catch(...){return -1;}
}
