// Prioritize four-position fits by independent measurements in the full images.
#include "quartet_fit.hpp"
namespace quartet_returns {
using quartet_fit::Shutter;
struct Trial {int night;double ceiling;};
bool forecast(const original::Sky &sky,const ArcSeed &seed,const int *mapping,int length,original::Core &core){
    Shutter e[4];int nights[4];float x[4],y[4];double mag[4],colour[4];
    for(int k=0;k<4;k++){
        int id=seed.rows[k];if(id<0||id>=length)return false;
        int row=mapping?mapping[id]:id;if(row<0||row>=sky.size||!std::isfinite(sky.mag[row]))return false;
        int ei=sky.ex[row];const auto &z=sky.images[ei].exposure;
        e[k]={z.t,z.ox,z.oy,z.oz};nights[k]=z.night;core.rows[k]=row;
        x[k]=sky.xy[2*row]/as;y[k]=sky.xy[2*row+1]/as;
        mag[k]=sky.mag[row];colour[k]=sky.lever[ei];
    }
    float left[]={x[0],y[0],x[1]-x[0],y[1]-y[0]},right[]={x[2],y[2],x[3]-x[2],y[3]-y[2]};
    quartet_fit::State orbit;double rms;
    if(!quartet_fit::solve(left,right,e,seed.gamma,orbit,rms))return false;
    const auto model=quartet_fit::uncertainty(orbit,e,rms);
    double m=.25*(mag[0]+mag[1]+mag[2]+mag[3]),c=.25*(colour[0]+colour[1]+colour[2]+colour[3]);
    double cm=0.,cc=.55;
    for(int k=0;k<4;k++){cm+=(mag[k]-m)*(colour[k]-c);cc+=sq(colour[k]-c);}
    double slope=std::clamp(cm/cc,-.45,.50);
    double best=-20.,second=-20.;int best_night=-1,best_row=-1;
    std::vector<Trial> apertures;apertures.reserve(48);
    for(int i=0;i<int(sky.images.size());i++){
        const auto &im=sky.images[i];const auto &z=im.exposure;
        if(!im.nx||std::find(nights,nights+4,z.night)!=nights+4)continue;
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
        if(z.night==best_night){if(peak>best){best=peak;best_row=winner;}continue;}
        if(peak>best){second=best;best=peak;best_night=z.night;best_row=winner;}
        else second=std::max(second,peak);
    }
    if(best_row<0||best<.25)return false;
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
    std::copy_n(state,6,core.x);core.ref=orbit.ref;core.count=4;
    core.rows[4]=best_row;core.pad=int(std::lround(1000.*best));
    core.rows[5]=int(std::lround(1000.*correction));
    core.merit=.13*std::clamp(double(seed.score),-18.,22.)+best+.62*correction+
               .45*std::clamp(second,-1.,8.)-sq(rms/.32);
    return std::isfinite(core.merit);
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
