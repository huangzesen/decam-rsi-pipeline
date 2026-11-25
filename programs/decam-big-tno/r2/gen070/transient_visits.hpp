// Test a catalogue peak against the same sky in deeper reference visits.
// A coincident source in a second shutter can confirm the peak even when
// the time between shutters is too short to resolve any angular motion.
namespace transient_visits {
constexpr double patch_width = 300. * as;
constexpr double two_pi = 6.283185307179586;
struct Evidence {
    int peer = -1, references = 0;
    float blank = 0, repeat = 0, motion = 0, opportunities = 0;
};
struct FrameAt {
    int id;
    double depth;
};
struct Patch {
    std::vector<FrameAt> frames;
};
struct Visits {
    const original::Sky &sky;
    const double *depth;
    std::unordered_map<uint64_t,Patch> patches;
    Visits(const original::Sky &sky_, const double *depth_):sky(sky_),depth(depth_){
        for (int e=0;e<int(sky.images.size());++e){
            const auto &im=sky.images[e];
            if (!im.nx || !std::isfinite(depth[e])) continue;
            for (int y=bin(im.ymin);y<=bin(im.ymax);++y)
                for (int x=bin(im.xmin);x<=bin(im.xmax);++x)
                    patches[key(x,y)].frames.push_back({e,depth[e]});
        }
        for (auto &entry:patches){
            auto &views=entry.second.frames;
            std::sort(views.begin(),views.end(),[](const FrameAt &a,const FrameAt &b){
                return a.depth!=b.depth?a.depth>b.depth:a.id<b.id;
            });
        }
    }
    static int bin(double x){return int(std::floor(x/patch_width));}
    // Coverage from the 24-arcsecond original catalogue grid. Require
    // light on adjacent silicon before using a catalogue nondetection.
    bool live(const original::Frame &im,double x,double y) const {
        int ix=int(std::floor(x/original::cell))-im.bx, iy=int(std::floor(y/original::cell))-im.by;
        if(ix<1||ix+1>=im.nx||iy<1||iy+1>=im.ny)return false;
        int k=iy*im.nx+ix;
        auto count=[&](int i){return im.starts[i+1]-im.starts[i];};
        if(count(k)>1)return true;
        return count(k-1)+count(k+1)>0 && count(k-im.nx)+count(k+im.nx)>0 &&
               count(k-1)+count(k+1)+count(k-im.nx)+count(k+im.nx)>=4;
    }
    Evidence measure(int row, double east, double north) const {
        Evidence out;
        int e=sky.ex[row];
        const auto &a=sky.images[e].exposure;
        double x=sky.xy[2*row], y=sky.xy[2*row+1], m=sky.mag[row], sn=sky.sn[row];
        if(!std::isfinite(m)||!std::isfinite(x+y)||sky.flag[row]&1)return out;
        auto found=patches.find(key(bin(x),bin(y)));
        if(found==patches.end())return out;
        double reference[3]={0.,0.,0.},colors[3]={-.53,0.,.55};
        int nights[5];
        for(auto &visit:found->second.frames){
            int j=visit.id;
            const auto &im=sky.images[j];
            int night=im.exposure.night;
            if(visit.depth<m-.95)break;
            if(std::abs(im.exposure.t-a.t)<3. || std::find(nights,nights+out.references,night)!=nights+out.references)continue;
            if(!live(im,x,y))continue;
            nights[out.references++]=night;
            double counterpart[3]={0.,0.,0.};
            double dl=sky.lever[j]-sky.lever[e];
            sky.nearby(im,x,y,1.45*as,1.45*as,[&](int k){
                double r2=(sq(sky.xy[2*k]-x)+sq(sky.xy[2*k+1]-y))/sq(as);
                if(r2>2.1 || !std::isfinite(sky.mag[k]))return;
                int mb=std::clamp(int((sky.mag[k]-10.)*4.),0,79);
                double err=sq(.16)+sq(1.15/sn)+sq(1.15/std::max(4.5f,sky.sn[k]));
                double psf=std::exp(-.5*r2/err)/(two_pi*err);
                double photvar=sq(.19)+sq(1.086/sn)+sq(1.086/std::max(4.5f,sky.sn[k]))+sq(.11*dl);
                double rate=std::max(2.e-5,im.rho[mb]);
                for(int c=0;c<3;c++){
                    double dm=sky.mag[k]-m-colors[c]*dl;
                    // A variable star or imperfect deblend can also repeat.
                    double phot=.90*std::exp(-.5*dm*dm/photvar)/std::sqrt(two_pi*photvar)
                               +.10*std::exp(-.5*dm*dm/1.1)/std::sqrt(two_pi*1.1);
                    counterpart[c]+=psf*phot/rate;
                }
            });
            for(int c=0;c<3;c++){
                double detect=.92/(1.+std::exp(std::clamp((m+colors[c]*dl-visit.depth-.08)/.27,-25.,25.)));
                reference[c]+=std::log(1.-detect+detect*counterpart[c]);
            }
            if(out.references==5)break;
        }
        double peak=*std::max_element(reference,reference+3);
        double steady=peak+std::log(.24*std::exp(reference[0]-peak)+.52*std::exp(reference[1]-peak)
                                     +.24*std::exp(reference[2]-peak));
        out.blank=std::clamp(-steady,-15.,9.);
        int used=0;
        for(auto &visit:found->second.frames){
            int j=visit.id; const auto &im=sky.images[j];const auto &b=im.exposure;
            double dt=b.t-a.t;
            if(e==j||a.night!=b.night||std::abs(dt)>.54||visit.depth<m-1.1)continue;
            double ax=(b.ox-a.ox-x*(b.oz-a.oz))/as,ay=(b.oy-a.oy-y*(b.oz-a.oz))/as;
            double scale=std::sqrt(1.+x*x+y*y),g=scale/43.,vc=std::sqrt(mu/std::pow(43.,3))*scale/as;
            double dx=(east*vc*dt-g*ax)/(1.-g*b.oz), dy=(north*vc*dt-g*ay)/(1.-g*b.oz);
            if(!live(im,x+dx*as,y+dy*as))continue;
            double cra=.20*(1.5*east*vc*dt-g*ax),cde=.20*(1.5*north*vc*dt-g*ay);
            double d=std::min(1.15,.32*std::abs(vc*dt));
            double detect=.91/(1.+std::exp(std::clamp((m-visit.depth-.10)/.27,-25.,25.)));
            out.opportunities+=detect;
            double best=-1.e20,star=0.,fast=0.,other=0.;int member=-1;
            double pad=2.0+std::abs(dx)+std::abs(dy)+3.4*(std::abs(cra)+std::abs(cde)+d);
            sky.nearby(im,x,y,pad*as,pad*as,[&](int k){
                if((sky.flag[k]&1)||!std::isfinite(sky.mag[k]))return;
                double u=(sky.xy[2*k]-x)/as,v=(sky.xy[2*k+1]-y)/as;
                double dm=sky.mag[k]-m;
                double pv=sq(.14)+sq(1.086/sn)+sq(1.086/std::max(4.5f,sky.sn[k]))+sq(.30*(sky.lever[e]-sky.lever[j]));
                if(dm*dm>18.*pv)return;
                int mb=std::clamp(int((sky.mag[k]-10.)*4.),0,79);
                double phot=std::exp(-.5*dm*dm/pv)/(std::sqrt(two_pi*pv)*std::max(2.e-5,im.rho[mb]));
                double sv=sq(.105)+sq(.96/sn)+sq(.96/std::max(4.5f,sky.sn[k]));
                double xx=sv+d*d+cra*cra,xy=cra*cde,yy=sv+d*d+cde*cde,det=xx*yy-xy*xy;
                double rx=u-dx,ry=v-dy;
                double chi=(yy*rx*rx-2.*xy*rx*ry+xx*ry*ry)/det;
                double orbit=phot*std::exp(-.5*chi)/(two_pi*std::sqrt(det));
                double r2=u*u+v*v;
                star+=phot*std::exp(-.5*r2/sv)/(two_pi*sv);
                double asteroid=sv+sq(240.*dt);
                fast+=phot*std::exp(-.5*r2/asteroid)/(two_pi*asteroid);
                if(orbit>best){other+=std::max(0.,best);best=orbit;member=k;}else other+=orbit;
            });
            if(member>=0){
                // Starting from a detected peak, the alternatives are a
                // moving foreground source, a steady counterpart, or noise.
                // Reference-image dropouts change the steady-source prior.
                double competitor=1.+.10*detect*fast+.13*detect*star*std::exp(std::clamp(steady,-10.,5.));
                double evidence=std::log((.002+detect*best)/(competitor+.42*detect*other));
                if(evidence>out.repeat || out.peer<0){
                    out.repeat=float(evidence);out.peer=member;
                    out.motion=float(std::log((1.+best)/(1.+star)));
                }
            }
            if(++used==5)break;
        }
        return out;
    }
};
}
extern "C" int measure_transient_visits(const original::Sky *sky,const double *depth,const double *east,
        const int *rows,int count,int threads,double seconds,transient_visits::Evidence *out) {
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        transient_visits::Visits field(*sky,depth);
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int th=0;th<threads;th++)workers.emplace_back([&]{
            while(!expired(end)){
                int at=next.fetch_add(64);if(at>=count)break;
                for(int j=at;j<std::min(at+64,count);++j)out[j]=field.measure(rows[j],east[0],east[1]);
            }
        });
        for(auto &worker:workers)worker.join();
        return std::min(count,next.load());
    }catch(...){return -1;}
}
