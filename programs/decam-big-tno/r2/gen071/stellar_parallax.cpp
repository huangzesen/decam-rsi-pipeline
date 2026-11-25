// Differential annual parallax from the original repeated-source catalogue.
// Positions in each sky tile are registered on neighbouring stars before
// testing coherent parallax of either sign and a bounded proper motion.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace stellar {
constexpr double arc=206264.80624709636,year=365.25,kepler=.01720209895;
using Clock=std::chrono::steady_clock;
struct Vec {
    double x=0.,y=0.;
    Vec operator+(Vec q)const{return {x+q.x,y+q.y};}
    Vec operator-(Vec q)const{return {x-q.x,y-q.y};}
    Vec operator*(double w)const{return {x*w,y*w};}
};
double dot(Vec a,Vec b){return a.x*b.x+a.y*b.y;}
double sq(double v){return v*v;}
struct Visit {double t,ox,oy,oz,depth;int night,band;};
struct Point {Vec p,obs;double t,m,var;int e,night,row,band;};
struct Report {double g,rms,sig,score,span;int count,miss,rows[48];};
struct Orbit {
    Vec p,v;double g=0.,t=0.,err=0.,rms=0.,peak=0.,velocity_error=0.,mag=0.;
    std::array<Vec,8> offset{};
};

// Profile two independent centroids and slopes out of the normal equations.
bool fit(const std::vector<Point> &points,const std::vector<int> &ids,Orbit &s,
         bool colour=false,int skip=-1){
    struct Band {double w=0.,t=0.;Vec p{},o{};};
    std::array<Band,8> bands{};
    double wt=0.,st=0.,sm=0.;Vec mean{},ob{};
    for(int j:ids){const auto &p=points[j];if(p.night==skip)continue;
        double w=1./p.var;wt+=w;st+=w*p.t;mean=mean+p.p*w;ob=ob+p.obs*w;sm+=w*p.m;
        if(colour){auto &b=bands[p.band];b.w+=w;b.t+=w*p.t;b.p=b.p+p.p*w;b.o=b.o+p.obs*w;}}
    if(wt<=0.)return false;
    s.mag=sm/wt;double precision=1./sq(.15);
    if(colour){
        // Profile the shared intercept with the same inverse covariance
        // as the slope and parallax. Measurements in a band share the
        // uncertainty in its centroid offset.
        wt=0.;st=0.;mean={};ob={};
        for(const auto &b:bands)if(b.w){
            double factor=precision/(precision+b.w);
            wt+=factor*b.w;st+=factor*b.t;mean=mean+b.p*factor;ob=ob+b.o*factor;
        }
    }
    st/=wt;mean=mean*(1./wt);ob=ob*(1./wt);
    Vec tx{},to{};double tt=0.,oo=0.,op=0.;
    for(int j:ids){const auto &p=points[j];if(p.night==skip)continue;
        double w=1./p.var,t=p.t-st;Vec d=p.p-mean,o=p.obs-ob;
        tt+=w*t*t;oo+=w*dot(o,o);op+=w*dot(o,d);
        to=to+o*(w*t);tx=tx+d*(w*t);
    }
    if(colour)for(auto &b:bands)if(b.w){
        // Blended stars can have a centroid offset in each passband.
        // Marginalizing it prevents a filter change from measuring parallax.
        b.t-=b.w*st;b.p=b.p-mean*b.w;b.o=b.o-ob*b.w;
        double scale=1./(b.w+precision);
        tt-=sq(b.t)*scale;oo-=dot(b.o,b.o)*scale;op-=dot(b.o,b.p)*scale;
        to=to-b.o*(b.t*scale);tx=tx-b.p*(b.t*scale);
    }
    if(tt<.00001)return false;
    double info=oo-dot(to,to)/tt;if(info<.05)return false;
    s.g=(dot(to,tx)/tt-op)/info;s.err=1./std::sqrt(info);
    s.v=(tx+to*s.g)*(1./tt);s.p=mean+ob*s.g;s.t=st;
    s.velocity_error=1./std::sqrt(tt)+std::sqrt(dot(to,to))/tt*s.err;
    s.offset.fill({});
    if(colour)for(size_t k=0;k<bands.size();k++){
        const auto &b=bands[k];
        s.offset[k]=(b.p-s.v*b.t+b.o*s.g)*(1./(b.w+precision));
    }
    s.rms=0.;s.peak=0.;int n=0;
    for(int j:ids){const auto &p=points[j];if(p.night==skip)continue;
        Vec residual=p.p-(s.p+s.v*(p.t-st)-p.obs*s.g+s.offset[p.band]);
        double q=dot(residual,residual);s.rms+=q;n++;s.peak=std::max(s.peak,q/p.var);}
    s.rms=std::sqrt(s.rms/std::max(1,n));return true;
}
bool motion(const Orbit &s,double sigma,double factor){
    double p=std::abs(s.g);
    if(p<.59||p>37.||p<sigma*s.err)return false;
    double esc=std::sqrt(2.)*kepler*year*arc*std::pow(p/arc,1.5);
    return std::hypot(s.v.x,s.v.y)<1.07*esc+factor*s.velocity_error;
}
struct Catalogue {
    const float *xy,*mag,*sn,*sharp;const uint8_t *flag;const int *exp;
    const Visit *visit;int ne;
    bool usable(int i)const{
        double s=sharp[i];
        return !(flag[i]&1)&&std::isfinite(mag[i])&&std::isfinite(xy[2*i]+xy[2*i+1])&&
               std::abs(s-1.)<std::max(.65,8.7/std::max(4.5f,sn[i]))&&s!=0.;
    }
};
struct Tile {
    const Catalogue &cat;std::vector<Point> p;double width=720.,margin=65.;
    int links;std::vector<std::vector<int>> local;
    std::vector<int> all,head,next,tmp,one,active;
    std::vector<float> best;
    std::vector<float> upper;
    std::vector<Vec> correction;
    std::vector<double> weights;
    std::unordered_set<uint64_t> tried;
    std::vector<Report> positive,negative;
    long long groups=0,confirmed=0;
    Tile(const Catalogue &catalogue,double x0,double y0,const std::vector<int> &rows):cat(catalogue){
        for(int i:rows){
            double X=cat.xy[2*i]*arc,Y=cat.xy[2*i+1]*arc;
            double x=X-x0+margin,y=Y-y0+margin;
            if(std::min(x,y)<0.||std::max(x,y)>=width+2*margin)continue;
            int e=cat.exp[i];const auto &v=cat.visit[e];
            p.push_back({{x,y},{v.ox-X/arc*v.oz,v.oy-Y/arc*v.oz},
                v.t/year,cat.mag[i],sq(.083)+sq(1.05/std::max(4.5f,cat.sn[i])),e,v.night,i,v.band});
        }
        links=int((width+2*margin)/20.)+1;local.resize(links*links);
        for(int i=0;i<int(p.size());i++){
            int x=int(p[i].p.x/20.),y=int(p[i].p.y/20.);
            local[x+y*links].push_back(i);
        }
        next.resize(p.size());upper.assign(p.size(),100.f);correction.resize(cat.ne);weights.resize(cat.ne);
        best.resize(cat.ne);one.assign(cat.ne,-1);tmp.reserve(200);
    }
    void representatives(std::vector<int> &ids,double mag){
        active.clear();
        for(int j:ids){auto &q=p[j];double cost=q.var+.09*sq(q.m-mag);
            if(std::abs(q.m-mag)>.74)continue;
            if(one[q.night]<0){active.push_back(q.night);one[q.night]=j;best[q.night]=cost;}
            else if(cost<best[q.night]){best[q.night]=cost;one[q.night]=j;}}
        ids.clear();for(int day:active){ids.push_back(one[day]);one[day]=-1;}
    }
    void register_stars(const std::vector<int> &ids,const std::vector<int> &members,size_t lo,size_t hi){
        if(ids.size()<6)return;
        double weight=0.,t=0.;Vec c{};
        for(int j:ids){const auto &q=p[j];double w=1./q.var;weight+=w;t+=w*q.t;c=c+q.p*w;}
        t/=weight;c=c*(1./weight);Vec slope{};double tt=.5;
        for(int j:ids){const auto &q=p[j];double u=q.t-t;tt+=sq(u)/q.var;slope=slope+(q.p-c)*(u/q.var);}
        slope=slope*(1./tt);
        for(int j:ids){auto &q=p[j];Vec d=q.p-(c+slope*(q.t-t));
            double r=std::hypot(d.x,d.y),w=std::min(1.,.13/std::max(.001,r))*std::min(1.,.025/q.var);
            if(r>.46)continue;
            correction[q.e]=correction[q.e]+d*w;weights[q.e]+=w;}
        // A repeatedly measured source can exclude a large parallax even
        // though its centroids remain available to fit a crossing object.
        // The bound must survive fitting arbitrary proper motion and
        // passband offsets in a set of independent nights.
        Orbit state;
        if(ids.size()<7||!fit(p,ids,state,true)||state.rms>.23||state.peak>16.)return;
        double high=std::abs(state.g)+3.9*state.err;
        if(high>10.)return;
        for(size_t a=lo;a<hi;a++){
            int j=members[a];const auto &q=p[j];
            Vec d=q.p-(state.p+state.v*(q.t-state.t)-q.obs*state.g+state.offset[q.band]);
            if(dot(d,d)<6.5*q.var&&std::abs(q.m-state.mag)<.52)
                upper[j]=std::min(upper[j],float(high));
        }
    }
    void extend(const Orbit &s,std::vector<int> &ids){
        ids.clear();active.clear();
        double span=13.5;
        double rad=std::abs(s.g)*1.25+span*std::hypot(s.v.x,s.v.y)+2.1;
        int bx0=std::max(0,int((s.p.x-rad)/20.)),bx1=std::min(links-1,int((s.p.x+rad)/20.));
        int by0=std::max(0,int((s.p.y-rad)/20.)),by1=std::min(links-1,int((s.p.y+rad)/20.));
        for(int y=by0;y<=by1;y++)for(int x=bx0;x<=bx1;x++)for(int j:local[x+y*links]){
            const auto &q=p[j];Vec miss=q.p-(s.p+s.v*(q.t-s.t)-q.obs*s.g+s.offset[q.band]);
            double r2=dot(miss,miss),dm=q.m-s.mag;
            if(r2>std::min(1.9,q.var*16.+.14)||std::abs(dm)>.85)continue;
            double loss=r2/(q.var+.015)+sq(dm/.32);
            if(one[q.night]<0){active.push_back(q.night);one[q.night]=j;best[q.night]=loss;}
            else if(loss<best[q.night]){best[q.night]=loss;one[q.night]=j;}
        }
        for(int day:active){ids.push_back(one[day]);one[day]=-1;}
    }
    double exposure_check(const Orbit &s,const std::vector<int> &ids,int &miss){
        // Revisit the same places when the trial body should have left.
        // A pair of alternately selected stars produces counterparts at
        // these fixed positions even while both stars are off the orbit.
        std::array<Vec,6> anchors{};
        std::array<int,6> anchor_band{};
        int na=0;
        for(int k=0;k<6;k++){
            int bestj=-1;double far=-1.;
            for(int j:ids){
                Vec a=p[j].p;double near=1.e10;
                for(int i=0;i<na;i++)near=std::min(near,dot(a-anchors[i],a-anchors[i]));
                if(near>far){bestj=j;far=near;}
            }
            if(bestj<0||far<.46*.46)break;
            anchor_band[na]=p[bestj].band;anchors[na++]=p[bestj].p;
        }
        struct Evidence {int nearby=0,quadrants=0;bool hit=false;
                         unsigned int stationary=0;double local=0.;int noise=0;
                         std::array<uint8_t,6> sectors{};};
        std::vector<Evidence> visit(cat.ne);
        std::vector<double> detect(cat.ne);
        std::vector<char> has(cat.ne);
        double rad=std::abs(s.g)*1.25+14.*std::hypot(s.v.x,s.v.y)+53.;
        int bx0=std::max(0,int((s.p.x-rad)/20.)),bx1=std::min(links-1,int((s.p.x+rad)/20.));
        int by0=std::max(0,int((s.p.y-rad)/20.)),by1=std::min(links-1,int((s.p.y+rad)/20.));
        for(int y=by0;y<=by1;y++)for(int x=bx0;x<=bx1;x++)for(int j:local[x+y*links]){
            const auto &q=p[j];auto &e=visit[q.e];
            Vec predict=s.p+s.v*(q.t-s.t)-q.obs*s.g+s.offset[q.band],d=q.p-predict;
            double r2=dot(d,d),dm=std::abs(q.m-s.mag);
            if(r2<sq(47.)){
                e.nearby++;e.quadrants|=1<<((d.x>0.)+2*(d.y>0.));
                if(cat.sn[q.row]<16.){e.local+=q.m+2.5*std::log10(cat.sn[q.row]/5.);e.noise++;}
            }
            if(r2<.27+9.*q.var&&dm<.65)e.hit=true;
            for(int k=0;k<na;k++){
                Vec d0=q.p-anchors[k],d1=predict-anchors[k];
                double dist=dot(d0,d0);
                if(dist<sq(30.))e.sectors[k]|=1<<((d0.x>0.)+2*(d0.y>0.));
                if(dm<.62&&dist<sq(.41)&&dot(d1,d1)>sq(.88))e.stationary|=1u<<k;
            }
        }
        for(int j:ids)has[p[j].night]=1;
        std::array<double,6> rec{};
        std::vector<unsigned int> ghost(cat.ne);
        std::vector<std::array<double,6>> vacancy(cat.ne);
        for(int i=0;i<cat.ne;i++){
            const auto &e=visit[i];int n=cat.visit[i].night;
            if(e.hit)has[n]=1;
            ghost[n]|=e.stationary;
            if(e.nearby<5||__builtin_popcount((unsigned)e.quadrants)<3)continue;
            double depth=cat.visit[i].depth;
            if(e.noise>=3)depth=(e.local+3.*depth)/(e.noise+3.);
            double prob=.94/(1.+std::exp(std::clamp((s.mag-depth-.03)/.24,-25.,25.)));
            detect[n]=std::max(detect[n],prob);
            const auto &v=cat.visit[i];
            // Trials at the former source position are informative only
            // when it is on a populated part of the sensor and separated
            // from a current orbital prediction. Use the local tangent
            // direction of the measured orbit for the parallax factor.
            if(prob<.35)continue;
            const auto &near=p[ids.front()];
            const double xrad=cat.xy[2*near.row],yrad=cat.xy[2*near.row+1];
            Vec observer{v.ox-xrad*v.oz,v.oy-yrad*v.oz};
            Vec prediction=s.p+s.v*(v.t/year-s.t)-observer*s.g+s.offset[v.band];
            for(int k=0;k<na;k++){
                Vec d=prediction-anchors[k];
                if(dot(d,d)<sq(.90)||__builtin_popcount((unsigned)e.sectors[k])<3)continue;
                // Colour differences make a cross-filter non-detection less informative.
                double chance=prob*(v.band==anchor_band[k]?.83:.51);
                vacancy[n][k]=std::max(vacancy[n][k],chance);
            }
        }
        std::array<double,6> log_vacant{},opportunities{};
        double missing=0.,surprise=0.;miss=0;
        for(int n=0;n<cat.ne;n++){
            if(!has[n]){missing-=std::log1p(-detect[n]);miss+=detect[n]>.78;}
            for(int k=0;k<na;k++){
                bool found=ghost[n]&(1u<<k);
                if(found)rec[k]++;
                double chance=vacancy[n][k];
                if(!chance)continue;
                opportunities[k]+=chance;
                // Allow a star with incomplete coverage or varying brightness.
                log_vacant[k]+=found?std::log(.06/(.06+.91*chance)):
                                      std::log(.94/(.94-.91*chance));
            }
        }
        // Several anchors may share their blank shutters. Average the
        // reference tests and temper correlated depth errors across nights.
        double absence=0.;int independent=0;
        for(int k=0;k<na;k++)if(opportunities[k]>=1.7){
            absence+=std::clamp(log_vacant[k]/std::sqrt(1.+opportunities[k]/8.),-7.,10.);independent++;
        }
        absence/=std::max(2,independent);
        std::sort(rec.begin(),rec.end(),std::greater<double>());
        if(na)surprise=std::min(5.,rec[0]);
        if(na>=2)surprise+=1.3*std::min(7.,rec[1]);
        if(na>=3)surprise+=.7*std::min(5.,rec[2]);
        // Dropout and uncertain footprint edges couple visits. A few
        // missed measurements are possible even in otherwise deep imaging.
        return 1.25*absence-1.75*surprise-.81*std::max(0.,missing-1.5);
    }
    void candidate(std::vector<int> &ids,double trial){
        Orbit s;if(!fit(p,ids,s)||!motion(s,4.7,4.)||s.rms>.56)return;
        if(trial!=0.&&trial*s.g<0.)return;
        for(int loop=0;loop<2;loop++){
            extend(s,ids);
            if(ids.size()<5||!fit(p,ids,s,loop==1)||!motion(s,5.7,3.5))return;
        }
        if(s.rms>.39||s.peak>22.||std::min(s.p.x,s.p.y)<margin||std::max(s.p.x,s.p.y)>=width+margin)return;
        std::sort(ids.begin(),ids.end(),[&](int a,int b){return p[a].t<p[b].t;});
        double years=p[ids.back()].t-p[ids.front()].t;
        if(years<.48)return;
        uint64_t hash=1469598103934665603ull;
        for(int j:ids){hash^=uint64_t(p[j].row+1);hash*=1099511628211ull;}
        if(!tried.insert(hash).second)return;
        double low=std::abs(s.g)/s.err;int repeats=std::min(12,int(ids.size()));
        for(int l=0;l<repeats;l++){
            Orbit t;if(!fit(p,ids,t,true,p[ids[l*ids.size()/repeats]].night))return;
            low=std::min(low,t.g*(s.g>0.?1.:-1.)/t.err);
        }
        if(low<3.6)return;
        double cm=0.,cp=0.;
        for(int j:ids){cm+=std::min(9.,sq((p[j].m-s.mag)/.26));cp+=p[j].var;}
        double sig=std::abs(s.g)/s.err,n=ids.size();
        double score=.7*std::min(27.,sig)+.83*std::min(23.,low)+.9*std::min(14.,n-4.)
                    -2.2*s.rms/std::sqrt(cp/n)-.36*cm/n;
        Report r{};r.g=std::abs(s.g)/arc;r.rms=s.rms;r.sig=sig;r.span=years*year;
        r.score=score+exposure_check(s,ids,r.miss);
        r.count=std::min(48,int(n));
        for(int j=0;j<r.count;j++)r.rows[j]=p[ids[(long long)j*ids.size()/r.count]].row;
        (s.g>0.?positive:negative).push_back(r);confirmed++;
    }
    void pass(double par,double cell,int shift,bool registration=false){
        int w=int((width+2*margin+2.*std::abs(par)+16.)/cell)+4;
        head.assign(w*w,-1);all.clear();
        for(int j=0;j<int(p.size());j++){
            if(!registration&&upper[j]<.54*std::abs(par))continue;
            auto &q=p[j];Vec h=q.p+q.obs*par;
            int x=int(std::floor((h.x+std::abs(par)+4.)/cell+(shift?.51:0.)));
            int y=int(std::floor((h.y+std::abs(par)+4.)/cell+(shift?.43:0.)));
            if(x<0||x>=w||y<0||y>=w)continue;
            int b=y*w+x;if(head[b]<0)all.push_back(b);
            next[j]=head[b];head[b]=j;
        }
        std::vector<int> trial;
        for(int b:all){
            tmp.clear();
            for(int j=head[b];j>=0;j=next[j])tmp.push_back(j);
            if(tmp.size()<5||tmp.size()>430)continue;
            if(tmp.size()>1)std::sort(tmp.begin(),tmp.end(),[&](int a,int b){return p[a].m<p[b].m;});
            size_t start=0;
            while(start+4<tmp.size()){
                size_t end=start+1;
                while(end<tmp.size()&&p[tmp[end]].m<p[tmp[start]].m+.91)end++;
                if(end>=start+5){
                    trial.assign(tmp.begin()+start,tmp.begin()+end);
                    double m=p[tmp[(start+end-1)/2]].m;representatives(trial,m);
                    if(trial.size()>=5){groups++;
                        if(registration)register_stars(trial,tmp,start,end);
                        else candidate(trial,par);}
                }
                double previous=p[tmp[start]].m;
                do{start++;}while(start<end&&p[tmp[start]].m<previous+.48);
            }
        }
    }
    void scan(bool compact){
        if(p.size()<5)return;
        pass(0.,1.5,0,true);
        for(auto &q:p)q.p=q.p-correction[q.e]*(1./(weights[q.e]+12.));
        if(compact){
            pass(0.,2.6,0);pass(0.,3.5,1);
            return;
        }
        for(double amp:{0.,1.8,4.8,10.3,19.5}){
            double cell=amp==0.?2.6:2.8+.23*amp;
            for(int shift=0;shift<2;shift++){
                pass(amp,cell,shift);if(amp)pass(-amp,cell,shift);
            }
        }
    }
};
}
extern "C" int stellar_tracks(const float *xy,const float *mag,const float *sn,const float *sharp,
        const uint8_t *flag,const int *exp,int n,const stellar::Visit *vis,int ne,
        double seconds,stellar::Report *pos,stellar::Report *neg,int cap,int *size,long long *stat){
    using namespace stellar;
    try{
        auto end=Clock::now()+std::chrono::duration<double>(seconds);
        Catalogue c{xy,mag,sn,sharp,flag,exp,vis,ne};
        double xmin=1.e12,ymin=1.e12,xmax=-1.e12,ymax=-1.e12;
        for(int i=0;i<n;i++){
            if((i&1048575)==0&&Clock::now()>end)return 0;
            if(!c.usable(i))continue;
            double x=xy[2*i]*arc,y=xy[2*i+1]*arc;
            xmin=std::min(x,xmin);ymin=std::min(y,ymin);xmax=std::max(x,xmax);ymax=std::max(y,ymax);}
        if(xmin>xmax)return 0;
        int w=int((xmax-xmin)/720.)+1,h=int((ymax-ymin)/720.)+1;
        if(int64_t(w)*h>10000000)return -1;
        std::vector<std::vector<int>> tiles(w*h),images(w*h);
        for(int i=0;i<n;i++){
            if((i&1048575)==0&&Clock::now()>end)return 0;
            if(!c.usable(i))continue;
            int x=int((xy[2*i]*arc-xmin)/720.),y=int((xy[2*i+1]*arc-ymin)/720.);
            int b=x+w*y;tiles[b].push_back(i);
            if(images[b].empty()||images[b].back()!=exp[i])images[b].push_back(exp[i]);}
        std::vector<int> order;std::vector<double> information(w*h);
        std::vector<int> seen(ne,-1);
        for(int b=0;b<w*h;b++)if(tiles[b].size()>=12){
            int nd=0;double t0=1.e9,t1=-1.e9,st=0.,s2=0.,oo=0.;
            Vec so{},to{};
            auto observe=[&](int e){const auto &v=vis[e];if(seen[v.night]==b)return;
                seen[v.night]=b;nd++;double t=(v.t-vis[0].t)/year;Vec o{v.ox,v.oy};
                t0=std::min(t0,t);t1=std::max(t1,t);st+=t;s2+=t*t;
                oo+=dot(o,o);so=so+o;to=to+o*t;};
            for(int e:images[b])observe(e);
            // A trajectory at a tile edge can have observations in the
            // adjacent footprint. Use that union for the feasibility test.
            int core=nd;
            if(nd<5||t1-t0<.48){
                int x=b%w,y=b/w;
                for(int yy=std::max(0,y-1);yy<=std::min(h-1,y+1);yy++)
                    for(int xx=std::max(0,x-1);xx<=std::min(w-1,x+1);xx++)
                        for(int e:images[xx+w*yy])observe(e);
            }
            if(nd<5||t1-t0<.47)continue;
            to=to-so*(st/nd);
            double parallax=oo-dot(so,so)/nd-dot(to,to)/std::max(.000001,s2-st*st/nd);
            information[b]=(.16+std::sqrt(std::min(5.,std::max(0.,parallax))))*
                (.25+std::min(1.8,core/10.))/std::pow(std::max(60.,double(tiles[b].size())),.14);
            order.push_back(b);
        }
        auto priority=[&](int i){
            uint32_t v=uint32_t(i)*2654435761u+2246822519u;v^=v>>16;v*=2246822519u;v^=v>>13;
            return -std::log((double(v)+1.)/4294967298.)/information[i];};
        std::sort(order.begin(),order.end(),[&](int a,int b){return priority(a)<priority(b);});
        std::vector<Report> yes,no;std::vector<int> rows;
        auto trim=[&](std::vector<Report> &v){
            if(int(v.size())>cap){std::nth_element(v.begin(),v.begin()+cap,v.end(),
               [](auto &a,auto &b){return a.score>b.score;});v.resize(cap);}};
        auto visit_tile=[&](int b,bool compact){
            int bx=b%w,by=b/w;rows.clear();
            for(int y=std::max(0,by-1);y<=std::min(h-1,by+1);y++)
            for(int x=std::max(0,bx-1);x<=std::min(w-1,bx+1);x++){
                const auto &pool=tiles[x+w*y];rows.insert(rows.end(),pool.begin(),pool.end());}
            Tile t(c,xmin+bx*720.,ymin+by*720.,rows);t.scan(compact);
            yes.insert(yes.end(),t.positive.begin(),t.positive.end());
            no.insert(no.end(),t.negative.begin(),t.negative.end());
            if(int(yes.size())>cap*2)trim(yes);if(int(no.size())>cap*2)trim(no);
            stat[1]+=t.p.size();stat[2]+=t.groups;stat[3]+=t.confirmed;stat[5]+=!compact;
        };
        std::vector<int> quick;int i=0;
        for(int b:order){
            if(Clock::now()>=end)break;
            bool compact=(i++%5)!=0;
            visit_tile(b,compact);stat[0]++;
            if(compact)quick.push_back(b);
        }
        for(int b:quick){if(Clock::now()>=end)break;visit_tile(b,false);}
        trim(yes);trim(no);std::copy(yes.begin(),yes.end(),pos);std::copy(no.begin(),no.end(),neg);
        size[0]=yes.size();size[1]=no.size();stat[4]=order.size();return 0;
    }catch(...){return -1;}
}
