// Compare a member measurement with short, independently observed asteroid arcs.
namespace foreground {
struct Result {float odds[3],span;int other[3],images;};
struct Vector {
    double x,y,z;
    Vector operator+(Vector b)const{return {x+b.x,y+b.y,z+b.z};}
    Vector operator-(Vector b)const{return {x-b.x,y-b.y,z-b.z};}
    Vector operator*(double s)const{return {x*s,y*s,z*s};}
    double product(Vector b)const{return x*b.x+y*b.y+z*b.z;}
    Vector cross(Vector b)const{return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    Vector unit()const{return *this*(1./std::sqrt(product(*this)));}
};
inline Vector observer(const Exposure &e){return {e.ox,e.oy,e.oz};}
inline Vector heliocentric(Vector ray,Vector obs,double r){
    double p=ray.product(obs);
    return obs+ray*(std::sqrt(r*r+p*p-obs.product(obs))-p);
}
struct Choice {int row;double score;Vector ray;};
inline double mark(const original::Sky &s,const original::Frame &im,int j,double magnitude,double lev){
    if(!std::isfinite(s.mag[j]) || s.flag[j]&1)return -80.;
    double w=sq(.16)+sq(1.086/std::max(4.5f,s.sn[j]))+sq(.28*lev)+sq(.13);
    double dm=s.mag[j]-magnitude;
    if(dm*dm>10.*w)return -80.;
    int bin=std::clamp(int((s.mag[j]-10)*4.),0,79);
    double broad=std::log(.09)+.5*std::log(w/(w+.24))-.5*sq(dm)/(w+.24);
    double like=std::log(std::exp(-.5*dm*dm/w)+std::exp(broad))-.5*std::log(2*M_PI*w);
    double density=im.rho[bin],fraction=1.;
    if(s.population){
        int tier=s.population[j];
        if(tier){density=im.transient[tier-1][bin];fraction=tier==2?.72:.20;}
        else{density=std::max(.00002,density-im.transient[0][bin]-im.transient[1][bin]);fraction=.08;}
    }
    double shape=(s.sharp[j]-1.)/std::hypot(.25,3.8/std::max(4.5f,s.sn[j]));
    return like+std::log(fraction/std::max(.000003,density))-.29*std::min(12.,sq(shape))
           -.38*bool(s.flag[j]&2);
}
Result search(const original::Sky &s,int row,const std::vector<int> &visits,
              const double *pole,const double *depth,double far_speed){
    Result out{{-40,-40,-40},0,{-1,-1,-1},0};
    int home=s.ex[row];
    double x=s.xy[2*row],y=s.xy[2*row+1],mag=s.mag[row];
    if(!std::isfinite(mag))return out;
    auto &A=s.images[home].exposure;
    Vector ray=Vector{x,y,1}.unit(),O=observer(A),north{pole[0],pole[1],pole[2]};
    std::vector<std::pair<double,int>> selected;
    for(int e:visits){
        const auto &im=s.images[e];double t=std::abs(im.exposure.t-A.t);
        double w=t*.021+.002;
        if(!im.nx||x<im.xmin-w||x>im.xmax+w||y<im.ymin-w||y>im.ymax+w)continue;
        if(std::isfinite(depth[e]) && mag>depth[e]+1.3)continue;
        double value=std::clamp(depth[e]-mag,-1.4,1.6);
        selected.emplace_back(std::isfinite(value)?value:0.,e);
    }
    if(selected.size()<2)return out;
    std::sort(selected.rbegin(),selected.rend());
    std::vector<int> other;
    for(auto v:selected){
        double t=s.images[v.second].exposure.t;
        bool duplicate=false;
        for(int j:other)if(std::abs(s.images[j].exposure.t-t)<.0022)duplicate=true;
        if(!duplicate)other.push_back(v.second);
        if(other.size()==13)break;
    }
    if(other.size()<2)return out;
    out.images=other.size();
    std::vector<int> second=other;
    auto priority=[&](int e){
        double dt=std::abs(s.images[e].exposure.t-A.t);
        return dt<.32?3.-.5*std::abs(std::log(dt/.07)):-dt;
    };
    std::sort(second.begin(),second.end(),[&](int i,int j){return priority(i)>priority(j);});
    if(second.size()>4)second.resize(4);
    double minimum=std::max(8.,far_speed+5.)*24.;
    double angle=std::fmod(row*.754877666,1.)*2*M_PI;
    for(int ib:second){
        auto &im=s.images[ib];auto &B=im.exposure;double dt=B.t-A.t;
        if(std::abs(dt)>1.60 || std::abs(dt)<.0022)continue;
        for(double radius:{1.75,2.45,3.30,5.4,9.0}){
            Vector h=heliocentric(ray,O,radius);
            Vector vc=north.cross(h).unit()*std::sqrt(mu/radius);
            Vector acceleration=h*(-mu/std::pow(radius,3));
            Vector cb=h+vc*dt+acceleration*(.5*dt*dt)-observer(B);
            if(cb.z<=.15)continue;
            double cx=cb.x/cb.z,cy=cb.y/cb.z;
            double width=std::hypot(.22,.26*std::sqrt(mu/radius)*std::abs(dt)/cb.z/as);
            double area=2*M_PI*width*width;
            std::vector<Choice> bank;
            s.nearby(im,cx,cy,2.55*width*as,2.55*width*as,[&](int j){
                double bx=s.xy[2*j],by=s.xy[2*j+1];
                double disp=std::hypot(bx-x,by-y)/as;
                if(disp<std::max(1.25,minimum*std::abs(dt)))return;
                double chi=(sq((bx-cx)/as)+sq((by-cy)/as))/sq(width);
                if(chi>6.5)return;
                double score=mark(s,im,j,mag,s.lever[ib]-s.lever[home])-.5*chi-std::log(area);
                if(score< -10.)return;
                bank.push_back({j,score,Vector{bx,by,1}.unit()});
            });
            if(bank.size()>40){
                std::nth_element(bank.begin(),bank.begin()+40,bank.end(),[](auto &a,auto &b){return a.score>b.score;});
                bank.resize(40);
            }
            for(auto &b:bank){
                Vector hb=heliocentric(b.ray,observer(B),radius);
                double best[3]={-40.,-40.,-40.},next[3]={-40.,-40.,-40.};
                int witness[3]={-1,-1,-1},follow[3]={-1,-1,-1};double span=0.;
                for(int ic:other){
                    if(ic==ib)continue;
                    auto &image=s.images[ic];auto &C=image.exposure;double t=C.t-A.t,w=t/dt;
                    if(std::min(std::abs(t),std::abs(t-dt))<.0022 || std::abs(w)>18.)continue;
                    Vector pred=h*(1.-w)+hb*w+acceleration*(.5*t*(t-dt))-observer(C);
                    if(pred.z<=.1)continue;
                    double u=pred.x/pred.z,v=pred.y/pred.z;
                    double separation=std::min(std::hypot(u-x,v-y),std::hypot(u-b.ray.x/b.ray.z,v-b.ray.y/b.ray.z))/as;
                    if(separation<1.25)continue;
                    double noise=sq(.085)+sq(1.05/s.sn[row])*sq(1.-w)+sq(1.05/s.sn[b.row])*sq(w);
                    double d2=std::abs(t*(t-dt));
                    // Short-arc range, radial motion and rotation of the
                    // topocentre all affect an extrapolated fast mover.
                    noise+=sq((.36+1.12/std::max(.5,pred.z))*d2)+sq(.08)+sq(.13*w);
                    if(noise>sq(4.7))continue;
                    double m=.53*mag+.47*s.mag[b.row];
                    double lev=s.lever[ic]-.53*s.lever[home]-.47*s.lever[ib];
                    for(int k=0;k<3;k++){
                        double shift=k?std::max(11.,6.*std::sqrt(noise)):0.;
                        double px=u+as*shift*std::cos(angle+k*2.4),py=v+as*shift*std::sin(angle+k*2.4);
                        double width=3.65*std::sqrt(noise+.24*.24);
                        double val=-40.;int hit=-1;
                        s.nearby(image,px,py,width*as,width*as,[&](int j){
                            double variance=noise+sq(1.05/s.sn[j]);
                            double chi=(sq((s.xy[2*j]-px)/as)+sq((s.xy[2*j+1]-py)/as))/variance;
                            if(chi>13.)return;
                            double z=mark(s,image,j,m,lev)-.5*chi-std::log(2*M_PI*variance);
                            if(z>val){val=z;hit=j;}
                        });
                        if(val>best[k]){
                            next[k]=best[k];follow[k]=witness[k];best[k]=val;witness[k]=hit;
                        }else if(val>next[k]){next[k]=val;follow[k]=hit;}
                    }
                    span=std::max(span,std::max(std::abs(t),std::abs(t-dt)));
                }
                for(int k=0;k<3;k++){
                    double z=b.score+best[k]+std::max(0.,.85*next[k]-1.5)
                            -std::log(5.*second.size()*std::max(1,int(other.size())-1));
                    if(z>out.odds[k]){
                        out.odds[k]=z;
                        if(!k){out.other[0]=b.row;out.other[1]=witness[k];out.other[2]=follow[k];out.span=span;}
                    }
                }
            }
        }
    }
    return out;
}
}
extern "C" int foreground_tracks(const original::Sky *sky,const int *rows,const float *speed,int n,
                                 const double *pole,const double *depth,double seconds,foreground::Result *out){
    try{
        int ne=sky->images.size();std::vector<std::vector<int>> visits(ne);
        for(int e=0;e<ne;e++)for(int j=0;j<ne;j++){
            double t=std::abs(sky->images[e].exposure.t-sky->images[j].exposure.t);
            if(t>=.0022 && t<3.7)visits[e].push_back(j);
        }
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};
        auto work=[&]{
            while(Clock::now()<end){
                int at=next.fetch_add(4);if(at>=n)break;
                for(int i=at;i<std::min(at+4,n);i++){
                    int row=rows[i];out[i]=foreground::search(*sky,row,visits[sky->ex[row]],pole,depth,speed[i]);
                }
            }
        };
        std::thread thread(work);work();thread.join();
        return std::min(n,next.load());
    }catch(...){return -1;}
}
