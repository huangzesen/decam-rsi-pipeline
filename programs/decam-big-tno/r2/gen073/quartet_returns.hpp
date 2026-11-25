// Prioritize four-position fits by independent measurements in the full images.
#include "quartet_fit.hpp"
namespace quartet_returns {
using quartet_fit::Shutter;
struct Trial {int night;double ceiling;};
template<int N=4>
bool forecast(const original::Sky &sky,const ArcSeed &seed,const int *mapping,int length,original::Core &core,
              std::array<int,3> *alternates=nullptr){
    Shutter e[N];int nights[N];double xypos[2][N],mag[N],colour[N],m=0.,c=0.;
    for(int k=0;k<N;k++){
        int id=seed.rows[k];if(id<0||id>=length)return false;
        int row=mapping?mapping[id]:id;if(row<0||row>=sky.size||!std::isfinite(sky.mag[row]))return false;
        int ei=sky.ex[row];const auto &z=sky.images[ei].exposure;
        e[k]={z.t,z.ox,z.oy,z.oz};nights[k]=z.night;core.rows[k]=row;
        xypos[0][k]=sky.xy[2*row];xypos[1][k]=sky.xy[2*row+1];
        mag[k]=sky.mag[row];colour[k]=sky.lever[ei];
        m+=mag[k]/N;c+=colour[k]/N;
    }
    quartet_fit::State orbit;double rms;
    if(!quartet_fit::positions(xypos,e,seed.gamma,orbit,rms))return false;
    const auto model=quartet_fit::uncertainty<N>(orbit,e,rms);
    double cm=0.,cc=.55;
    for(int k=0;k<N;k++){cm+=(mag[k]-m)*(colour[k]-c);cc+=sq(colour[k]-c);}
    double slope=std::clamp(cm/cc,-.45,.50);
    double best=-20.,second=-20.;int best_night=-1,best_row=-1;
    std::vector<Trial> apertures;apertures.reserve(48);
    struct Return {double value;int row,night;};
    std::vector<Return> choices;
    for(int i=0;i<int(sky.images.size());i++){
        const auto &im=sky.images[i];const auto &z=im.exposure;
        if(!im.nx||std::find(nights,nights+N,z.night)!=nights+N)continue;
        Shutter s{z.t,z.ox,z.oy,z.oz};
        double px,py,pad;quartet_fit::coarse(orbit,model,s,px,py,pad);
        if(px*as<im.xmin-pad*as||px*as>im.xmax+pad*as||
           py*as<im.ymin-pad*as||py*as>im.ymax+pad*as)continue;
        double f,u,xx,xy,yy;quartet_fit::angle(orbit,s,model.r0,px,py,f,u);
        quartet_fit::ellipse(orbit,model,s,px,py,f,u,xx,xy,yy);
        double det=xx*yy-xy*xy;
        if(!(det>0.)||xx+yy>240000.)continue;
        double area=std::sqrt(det),mp=m+slope*(sky.lever[i]-c);
        int bin=std::clamp(int((mp-10.)*4.),0,79),tier=0;
        double density=im.rho[bin];
        if(area*density>.35||area>500.){
            if(!sky.population||area>4500.)continue;
            tier=1;density=im.transient[0][bin]/.68;
            if(area*density>.55){tier=2;density=im.transient[1][bin]/.42;}
            if(area*density>.7)continue;
        }
        double base=-std::log(std::max(1.e-9,6.283185307*area*density)),peak=-20.;int winner=-1;
        apertures.push_back({z.night,base-std::log(2.506628*.29)});
        sky.ellipse(im,px,py,xx,xy,yy,[&](int row){
            if((tier&&sky.population[row]<tier)||(sky.flag[row]&1))return;
            double dm=sky.mag[row]-mp;if(!(std::abs(dm)<.94))return;
            double dx=(sky.xy[2*row]-px)/as,dy=(sky.xy[2*row+1]-py)/as;
            double chi=(yy*dx*dx+xx*dy*dy-2.*xy*dx*dy)/det;
            if(!(chi<10.))return;
            double noise=std::hypot(.25,1.09/sky.sn[row]);
            double score=base-.5*(chi+sq(dm/noise))-std::log(2.5066*noise);
            score-=.15*std::min(9.,sq((sky.sharp[row]-1.)/std::hypot(.24,3.8/sky.sn[row])));
            if(score>peak){peak=score;winner=row;}
        });
        if constexpr(N==3)if(winner>=0 && peak>-.4)choices.push_back({peak,winner,z.night});
        if(z.night==best_night){if(peak>best){best=peak;best_row=winner;}continue;}
        if(peak>best){second=best;best=peak;best_night=z.night;best_row=winner;}
        else second=std::max(second,peak);
    }
    if(best_row<0||best<(N==3?-.4:.25))return false;
    if constexpr(N==3){
        std::sort(choices.begin(),choices.end(),[](auto &a,auto &b){return a.value>b.value;});
        int taken[3]={-1,-1,-1},used=0;
        for(const auto &v:choices){
            if(std::find(taken,taken+used,v.night)!=taken+used)continue;
            taken[used]=v.night;(*alternates)[used++]=v.row;
            if(used==3)break;
        }
    }
    // Integrate the unrelated-source intensity inside the 3-D likelihood
    // contour (two astrometric axes and magnitude) of the winning return.
    // Co-located repeats on a night share some of their unrelated stars.
    std::sort(apertures.begin(),apertures.end(),[](auto &a,auto &b){return a.night<b.night;});
    double expected=0.,maximum=0.,sum=0.;int prev=-1;
    for(auto &w:apertures){
        if(w.night!=prev){
            expected+=.65*maximum+.35*sum;maximum=0.;sum=0.;prev=w.night;
        }
        double delta=w.ceiling-best;if(delta<=0.)continue;
        double count=.75225278*std::exp(-w.ceiling)*delta*std::sqrt(delta);
        maximum=std::max(maximum,count);sum+=count;
    }
    expected+=.65*maximum+.35*sum;
    double correction=std::clamp(-std::log(std::max(1.e-9,-std::expm1(-expected)))-best,-5.,2.);
    double state[]={orbit.x,orbit.y,orbit.g,orbit.vx,orbit.vy,orbit.h};
    std::copy_n(state,6,core.x);core.ref=orbit.ref;core.count=N;
    core.rows[N]=best_row;core.pad=int(std::lround(1000.*best));
    core.rows[N+1]=int(std::lround(1000.*correction));
    core.merit=.13*std::clamp(double(seed.score),-18.,22.)+best+.62*correction+
               .45*std::clamp(second,-1.,8.)-sq(rms/.32);
    return std::isfinite(core.merit);
}
extern "C" int triple_predictions(const original::Sky *sky,const ArcSeed *seeds,int n,double seconds,
                                  int capacity,original::Core *out,int *tested){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        int used=0,i=0;
        for(;i<n && used<capacity && !expired(end);i++){
            original::Core first{};
            std::array<int,3> next{-1,-1,-1};
            if(!quartet_returns::forecast<3>(*sky,seeds[i],nullptr,sky->size,first,&next))continue;
            for(int fourth:next){
                if(fourth<0||used>=capacity)break;
                ArcSeed seed=seeds[i];seed.rows[3]=fourth;
                std::sort(seed.rows,seed.rows+4,[&](int a,int b){
                    return sky->images[sky->ex[a]].exposure.t<sky->images[sky->ex[b]].exposure.t;
                });
                seed.gamma=first.x[2];seed.score=first.merit;
                if(quartet_returns::forecast<4>(*sky,seed,nullptr,sky->size,out[used]))used++;
            }
        }
        *tested=i;return used;
    }catch(...){return -1;}
}
}
extern "C" int quartet_predictions(const original::Sky *sky,const ArcSeed *seeds,int n,const int *rows,int length,
                                    double seconds,original::Core *out,int *tested){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        int used=0,i=0;
        for(;i<n;i++){
            if((i&15)==0&&expired(end))break;
            if(quartet_returns::forecast(*sky,seeds[i],rows,length,out[used]))used++;
        }
        *tested=i;return used;
    }catch(...){return -1;}
}
