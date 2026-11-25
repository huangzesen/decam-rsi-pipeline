// Deep reference visits for each sky patch. The catalogue in the member
// exposure is scored against independent visits with measured local noise.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {
constexpr double arcsec=4.84813681109536e-6;
constexpr double small_cell=12.*arcsec, patch_cell=144.*arcsec;
struct Visit {
    double time,zp,colour,limit,seeing;
    int band,pad;
};
struct Image {
    int left=0,bottom=0,width=0,height=0;
    std::vector<int> offsets,rows;
    int bin(double x,double y) const {
        int a=int(std::floor(x/small_cell))-left,b=int(std::floor(y/small_cell))-bottom;
        return a+b*width;
    }
    void build(const double *x,const double *y,int lo,int hi) {
        if(hi<=lo)return;
        auto xb=std::minmax_element(x+lo,x+hi),yb=std::minmax_element(y+lo,y+hi);
        left=std::floor(*xb.first/small_cell);bottom=std::floor(*yb.first/small_cell);
        width=int(std::floor(*xb.second/small_cell))-left+1;
        height=int(std::floor(*yb.second/small_cell))-bottom+1;
        if(width<=0||height<=0||int64_t(width)*height>4000000){width=height=0;return;}
        offsets.assign(size_t(width)*height+1,0);rows.resize(hi-lo);
        for(int i=lo;i<hi;i++)offsets[bin(x[i],y[i])+1]++;
        for(size_t j=1;j<offsets.size();j++)offsets[j]+=offsets[j-1];
        auto cursor=offsets;
        for(int i=lo;i<hi;i++)rows[cursor[bin(x[i],y[i])]++]=i;
    }
};
struct PatchVisit {int e;float limit;};
struct Survey {
    const double *x,*y;
    const float *flux,*sn,*error,*sharp,*xp,*yp;
    const uint8_t *flag;
    const int64_t *edges;
    const Visit *v;
    int ne,px,py,nx,ny;
    std::vector<Image> images;
    std::vector<std::vector<PatchVisit>> patches;
    Survey(const double *x_,const double *y_,const float *f_,const float *s_,const float *err_,
           const float *sh_,const float *xp_,const float *yp_,const uint8_t *flag_,const int64_t *edges_,
           const Visit *v_,int ne_):x(x_),y(y_),flux(f_),sn(s_),error(err_),sharp(sh_),xp(xp_),yp(yp_),
                                    flag(flag_),edges(edges_),v(v_),ne(ne_),images(ne_) {
        int n=edges[ne];
        auto xb=std::minmax_element(x,x+n),yb=std::minmax_element(y,y+n);
        px=std::floor(*xb.first/patch_cell);py=std::floor(*yb.first/patch_cell);
        nx=int(std::floor(*xb.second/patch_cell))-px+1;ny=int(std::floor(*yb.second/patch_cell))-py+1;
        if(int64_t(nx)*ny>10000000)throw std::bad_alloc();
        patches.resize(nx*ny);
    }
    int patch(int i) const {
        return int(std::floor(x[i]/patch_cell))-px+(int(std::floor(y[i]/patch_cell))-py)*nx;
    }
    void build(int threads){
        std::atomic<int> next{0};std::vector<std::thread> jobs;
        for(int k=0;k<threads;k++)jobs.emplace_back([&]{
            for(int e=next++;e<ne;e=next++)images[e].build(x,y,edges[e],edges[e+1]);
        });
        for(auto &j:jobs)j.join();
        std::vector<int> count(nx*ny),touched;std::vector<float> sum(nx*ny);
        for(int e=0;e<ne;e++){
            if(!std::isfinite(v[e].zp)||!images[e].width)continue;
            // Local depths from faint-source errors. Small patches with very
            // few sampled peaks borrow their exposure's measured magnitude.
            for(int i=edges[e];i<edges[e+1];i+=3){
                if(!(sn[i]<15.&&error[i]>0.)||(flag[i]&1))continue;
                int p=patch(i);if(!count[p]++)touched.push_back(p);
                sum[p]+=std::log(error[i]);
            }
            for(int p:touched){
                double lim=v[e].zp-1.747425-1.0857362*sum[p]/count[p];
                if(std::isfinite(v[e].limit))lim=(count[p]*lim+6.*v[e].limit)/(count[p]+6.);
                patches[p].push_back({e,float(lim)});count[p]=0;sum[p]=0.f;
            }
            touched.clear();
        }
        for(auto &list:patches)std::sort(list.begin(),list.end(),[](auto a,auto b){return a.limit>b.limit;});
    }
    struct Ref {double depth,background,halo;bool covered;};
    Ref inspect(int i,int e,double mag,double guess,int member)const{
        const auto &im=images[e];const auto &visit=v[e];
        int a=int(std::floor(x[i]/small_cell))-im.left,b=int(std::floor(y[i]/small_cell))-im.bottom;
        double near=1e8,err=0.,bright=0.,halo=0.;bool inside=false;
        double psf=std::clamp(std::hypot(v[member].seeing,visit.seeing)*.7071/2.355,.3,.94);
        for(int j=std::max(0,b-1);j<=std::min(im.height-1,b+1);j++){
            int low=std::max(0,a-1),high=std::min(im.width-1,a+1);
            if(low>high)continue;
            for(int h=im.offsets[j*im.width+low];h<im.offsets[j*im.width+high+1];h++){
                int r=im.rows[h];
                double dx=(x[r]-x[i])/arcsec,dy=(y[r]-y[i])/arcsec,d2=dx*dx+dy*dy;
                if(d2<near){
                    near=d2;err=error[r];
                    double margin=std::min({double(xp[r]),2048.-xp[r],double(yp[r]),4096.-yp[r]});
                    inside=margin>8.+4.1*std::sqrt(d2)&&!(flag[r]&4);
                }
                if(d2>100.||!(flux[r]>0.))continue;
                double ratio=flux[r]*std::exp(std::clamp(.921034*(mag-visit.zp),-60.,45.));
                if(d2<12.)bright=std::max(bright,ratio*std::exp(-d2/(4.*psf*psf)));
                halo=std::max(halo,ratio/(1.+d2)*(flag[r]&1?1.0:.50));
            }
        }
        double depth=err>0.?visit.zp-2.5*std::log10(5.*err):guess;
        return {std::min(depth,guess+.4),bright,halo,inside&&near<144.};
    }
    // Same filter and other filters can both rule out a stationary source.
    // Retain distinct dates and allow wider stellar colours across filters.
    void one(int i,int e,const std::vector<PatchVisit>& list,float &odds,uint8_t &recover)const{
        double mag=v[e].zp-2.5*std::log10(std::max(1.e-8f,flux[i]));
        if(!std::isfinite(mag))return;
        int used[6],n=0;double support=0.,matches=0.,strong=0.,halo=0.,deep=0.;
        for(int same=1;same>=0&&n<6;same--){
            for(const auto &ref:list){
                int k=ref.e;
                if((v[e].band==v[k].band)!=same||std::abs(v[e].time-v[k].time)<12.)continue;
                if(ref.limit<mag-(same?.65:.30))break;
                bool repeated=false;for(int j=0;j<n;j++)repeated|=std::abs(v[k].time-v[used[j]].time)<5.;
                if(repeated)continue;
                auto r=inspect(i,k,mag,ref.limit,e);
                if(!r.covered)continue;
                used[n++]=k;
                double colour=std::abs(v[e].colour-v[k].colour);
                double threshold=mag+.12+.42*colour+1.1/std::max(4.5f,sn[i]);
                double detectable=1./(1.+std::exp(std::clamp((threshold-r.depth)/(.19+.13*colour),-18.,18.)));
                double contam=std::min(3.,r.background*std::exp(-.29*colour));
                double steady=1./(1.+std::exp(std::clamp((.64-contam)/(.10+.07*colour),-18.,18.)));
                support+=detectable*(1.-steady);matches+=steady;
                strong=std::max(strong,steady);deep=std::max(deep,detectable*(1.-steady));
                halo=std::max(halo,r.halo);
                if(n>=6 || (same&&n>=4))break;
            }
        }
        double quality=.83*std::log1p(support)+.40*deep-1.15*std::min(2.,matches)
                      -.72*strong-.35*std::log1p(std::min(130.,halo));
        odds=float(std::clamp(quality,-4.,2.05));
        // An isolated star-like excess in several deep references can seed
        // even where most unrelated single-epoch sources are stationary.
        double shape=std::abs(sharp[i]-1.02)/std::hypot(.19,3.5/std::max(4.5f,sn[i]));
        recover=(support>=1.65&&deep>.75&&matches<.48&&strong<.40&&
                 halo<12.&&shape<2.05&&!(flag[i]&5)&&sharp[i]!=0.);
    }
};
}

extern "C" int deep_visits(const double *x,const double *y,const float *flux,const float *sn,
                           const float *error,const float *sharp,const float *xp,const float *yp,
                           const uint8_t *flag,const int64_t *edges,const Visit *visits,int ne,
                           const uint8_t *eligible,int threads,float *odds,uint8_t *rescue){
    if(ne<=0||!edges[ne])return 0;
    try{
        Survey s(x,y,flux,sn,error,sharp,xp,yp,flag,edges,visits,ne);s.build(threads);
        std::vector<std::thread> jobs;std::atomic<int> next{0};
        for(int k=0;k<threads;k++)jobs.emplace_back([&]{
            for(int e=next++;e<ne;e=next++){
                for(int i=edges[e];i<edges[e+1];i++)if(eligible[i])
                    s.one(i,e,s.patches[s.patch(i)],odds[i],rescue[i]);
            }
        });
        for(auto &j:jobs)j.join();
        return 0;
    }catch(...){return -1;}
}
