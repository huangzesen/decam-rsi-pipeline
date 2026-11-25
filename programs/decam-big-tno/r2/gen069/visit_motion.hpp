// Sequential marginal likelihood of a burst, conditional on its anchor.
// Each Gaussian branch is one assignment of measured peaks to the source.
namespace visit_motion {
struct Query {
    int row,pad;
    double vx,vy,g,mag,colour,magvar,colourvar;
};
struct Value { float linked,steady,fast,slow;int visits,peaks; };
struct Peak {double x,y,m,vp,vm,background;};
struct Image {
    std::vector<Peak> peaks;
    double t,l,eff,depth,dx,dy;
};
struct State {
    double x=0.,y=0.,vx=0.,vy=0.,p=0.,pv=0.,v=0.;
    double m=0.,c=0.,pm=0.,pc=0.,cc=0.,logw=0.;
};
inline double logsum(const std::vector<State> &s){
    if(s.empty())return -80.;
    double top=s.front().logw,z=0.;
    for(auto &v:s)top=std::max(top,v.logw);
    for(auto &v:s)z+=std::exp(v.logw-top);
    return top+std::log(z);
}
double filter(const Query &q,const std::vector<Image> &images,const original::Sky &sky,int mode){
    int i=q.row;
    double measured=sky.mag[i],sn=std::max(4.5f,sky.sn[i]);
    State s;
    s.p=sq(.055)+sq(.88/sn);
    double av=sq(.045)+sq(1.086/sn);
    // Local brightness is common to a burst. Other nights constrain its
    // mean and colour while rotation allows the entire night to be offset.
    if(mode==0){
        double vp=q.magvar+.12*.12;
        s.m=(measured*vp+q.mag*av)/(vp+av);s.pm=av*vp/(vp+av);
        s.c=q.colour;s.cc=q.colourvar;
    }else{s.m=measured;s.pm=av+.055*.055;s.c=0.;s.cc=mode==1?.58*.58:.36*.36;}
    s.v=mode==1?sq(.020):mode==2?sq(26.):mode==3?sq(1.6):sq(.045);
    std::vector<State> bank{s},next;next.reserve(500);
    for(const auto &e:images){
        double t=e.t,l=e.l,ox=0.,oy=0.;
        if(mode==0||mode==3){ox=e.dx;oy=e.dy;}
        next.clear();
        for(const auto &z:bank){
            double nc=z.m+z.c*l;
            double ss=5.*std::pow(10.,std::clamp(.4*(e.depth-nc),-4.,3.));
            double pm=z.pm+2*l*z.pc+l*l*z.cc;
            double sig=std::sqrt(1.+sq(ss/1.086)*(pm+.05*.05));
            double detect=.92*e.eff*.5*std::erfc((4.55-ss)/(1.41421356*sig));
            detect=std::clamp(detect,.002,.94);
            // Very fast foreground tracks may leave the queried aperture.
            if(mode==2)detect*=std::min(1.,1.-std::exp(-.5*sq(185.)/std::max(.001,t*t*z.v+z.p)));
            auto miss=z;miss.logw+=std::log1p(-detect);next.push_back(miss);
            double dx=z.x+z.vx*t+ox,dy=z.y+z.vy*t+oy;
            double p=z.p+2*t*z.pv+t*t*z.v;
            for(auto &h:e.peaks){
                double var=p+h.vp,u=h.x-dx,v=h.y-dy,chi=(u*u+v*v)/var;
                if(chi>23.)continue;
                double mv=pm+h.vm,dm=h.m-nc;if(dm*dm>18.*mv)continue;
                double score=std::log(detect)-.5*chi-std::log(2*M_PI*var)
                            -.5*dm*dm/mv-.5*std::log(2*M_PI*mv)-h.background;
                // Normalize the magnitude density for this thresholded
                // catalogue, including the draw from the flux posterior.
                double trunc=.5*std::erfc((4.5-ss)/(1.41421356*sig));
                score-=std::log(std::max(.10,trunc));
                State n=z;n.logw+=score;
                double w0=z.p+t*z.pv,w1=z.pv+t*z.v;
                n.x+=u*w0/var;n.y+=v*w0/var;n.vx+=u*w1/var;n.vy+=v*w1/var;
                n.p-=w0*w0/var;n.pv-=w0*w1/var;n.v-=w1*w1/var;
                w0=z.pm+l*z.pc;w1=z.pc+l*z.cc;
                n.m+=dm*w0/mv;n.c+=dm*w1/mv;
                n.pm-=w0*w0/mv;n.pc-=w0*w1/mv;n.cc-=w1*w1/mv;
                next.push_back(n);
            }
        }
        if(next.size()>20){
            std::nth_element(next.begin(),next.begin()+20,next.end(),[](auto &a,auto &b){return a.logw>b.logw;});
            next.resize(20);
        }
        bank.swap(next);
    }
    return logsum(bank);
}

Value assess(const original::Sky &sky,const Query &q,const double *zp,const double *depth,
             const std::vector<int> &visits){
    Value out{};int row=q.row,e0=sky.ex[row];const auto &base=sky.images[e0].exposure;
    if(!std::isfinite(q.mag+q.vx+q.vy+sky.mag[row]))return out;
    double x=sky.xy[2*row],y=sky.xy[2*row+1];
    std::vector<Image> frames;
    for(int e:visits){
        const auto &im=sky.images[e];const auto &d=im.exposure;
        if(e==e0||im.nx<=0||x<im.xmin-35*as||x>im.xmax+35*as||y<im.ymin-35*as||y>im.ymax+35*as)continue;
        double dt=d.t-base.t,lever=sky.lever[e]-sky.lever[e0];
        if(depth[e]+1.30<q.mag+q.colour*lever)continue;
        Image f;f.t=dt*24.;f.l=lever;
        f.dx=dt*q.vx-q.g*(d.ox-base.ox-x*(d.oz-base.oz))/as;
        f.dy=dt*q.vy-q.g*(d.oy-base.oy-y*(d.oz-base.oz))/as;
        // Establish local live-silicon coverage and noise around the actual
        // orbit prediction. A gap in the mosaic supplies no negative trial.
        double near=1.e9,margin=-1.,err[64];int nerr=0,areas=0;
        sky.nearby(im,x+f.dx*as,y+f.dy*as,33.*as,33.*as,[&](int j){
            double dx=(sky.xy[2*j]-x)/as-f.dx,dy=(sky.xy[2*j+1]-y)/as-f.dy,r=dx*dx+dy*dy;
            if(r>33.*33.)return;
            areas|=1<<((dx>0)+2*(dy>0));
            if(r<near){
                near=r;margin=std::min({double(sky.xp[j]),2048.-sky.xp[j],double(sky.yp[j]),4096.-sky.yp[j]});
            }
            if(nerr<64&&sky.sn[j]<25&&sky.ferr[j]>0.&&!(sky.flag[j]&1))err[nerr++]=sky.ferr[j];
        });
        if(!(near<sq(22.)) || (!(areas==15)&&margin*.263<std::sqrt(near)+3.))continue;
        f.depth=depth[e];
        if(nerr>=3&&std::isfinite(zp[e])){
            std::nth_element(err,err+nerr/2,err+nerr);
            f.depth=zp[e]-2.5*std::log10(5.*err[nerr/2]);
        }
        if(!std::isfinite(f.depth))continue;
        f.eff=near<sq(14.)||areas==15?1.:.65;
        double rmax=std::min(195.,std::max(std::hypot(f.dx,f.dy)+2.5,4.+95.*std::abs(f.t)));
        sky.nearby(im,x,y,rmax*as,rmax*as,[&](int j){
            if(!std::isfinite(sky.mag[j])||sky.flag[j]&1||std::abs(sky.mag[j]-sky.mag[row])>2.1)return;
            double u=(sky.xy[2*j]-x)/as,v=(sky.xy[2*j+1]-y)/as;
            if(u*u+v*v>rmax*rmax)return;
            double sn=std::max(4.5f,sky.sn[j]);
            int mb=std::clamp(int((sky.mag[j]-10.)*4.),0,79);
            double bg=std::log(std::max(.000002,im.rho[mb]));
            f.peaks.push_back({u,v,sky.mag[j],sq(.068)+sq(1.0/sn),
                            sq(.055)+sq(1.086/sn),bg});
        });
        frames.push_back(std::move(f));
    }
    // Keep independent filters and distinct portions of the night when many
    // near-identical shallow exposures revisit the same pointing.
    std::vector<Image> used;
    while(!frames.empty()&&used.size()<7){
        auto value=[&](const Image &a){
            double z=std::clamp(a.depth-q.mag,-1.3,1.7)+.35*std::log1p(std::abs(a.t));
            for(auto &u:used)z-=.92*std::exp(-sq((u.t-a.t)/.33)-sq((a.l-u.l)/.24));
            return z;
        };
        auto best=std::max_element(frames.begin(),frames.end(),[&](auto &a,auto &b){return value(a)<value(b);});
        used.push_back(std::move(*best));frames.erase(best);
    }
    if(used.empty())return out;
    // The best, often most separated, two visits localize foreground motion.
    out.linked=filter(q,used,sky,0);out.steady=filter(q,used,sky,1);
    out.fast=filter(q,used,sky,2);out.slow=filter(q,used,sky,3);
    out.visits=used.size();for(auto &u:used)out.peaks+=u.peaks.size();
    return out;
}
}

extern "C" int burst_alternatives(const original::Sky *sky,const visit_motion::Query *q,int n,
                    const double *zp,const double *depth,int threads,double seconds,visit_motion::Value *out){
    try{
        int ne=sky->images.size();
        std::vector<std::vector<int>> visits(ne);
        for(int i=0;i<ne;i++)for(int j=0;j<ne;j++)
            if(i!=j&&sky->images[i].exposure.night==sky->images[j].exposure.night)visits[i].push_back(j);
        std::atomic<int> next{0};std::vector<std::thread> workers;
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        for(int t=0;t<threads;t++)workers.emplace_back([&]{
            while(Clock::now()<end){
                int start=next.fetch_add(16);if(start>=n)break;
                for(int j=start;j<std::min(start+16,n);j++)out[j]=visit_motion::assess(*sky,q[j],zp,depth,visits[sky->ex[q[j].row]]);
            }
        });
        for(auto &w:workers)w.join();
        return std::min(next.load(),n);
    }catch(...){return -1;}
}
