// Search the neighbourhood of a reported row for a different, fast-moving source.
namespace foreground_tracks {
struct Query {int row,phase;double range;};
struct Result {float log_odds,speed;int count,nights;};
struct Visit {int e;double t,ox,oy,oz,merit;};
struct Pair {int row,j;double g,vx,vy,score;};
struct Trial {int e;double odds;};
struct Finder {
    const original::Sky &sky;
    const double *depth,*pole;
    Result run(const Query &q,const std::vector<int> &window)const{
        Result out{};
        int row=q.row,a=sky.ex[row];
        if(window.size()<2||!std::isfinite(sky.mag[row]))return out;
        const auto &im=sky.images[a];const auto &obs=im.exposure;
        double x=sky.xy[2*row],y=sky.xy[2*row+1],mag=sky.mag[row];
        if(q.phase){
            unsigned h=unsigned(row)*2654435761u+unsigned(q.phase)*2246822519u;
            double angle=6.283185307179586*(double(h)/4294967296.);
            x+=(55.+19.*q.phase)*as*cos(angle);y+=(55.+19.*q.phase)*as*sin(angle);
        }
        double norm=sqrt(1.+x*x+y*y);
        double ex=pole[1]-pole[2]*y,ey=pole[2]*x-pole[0],ez=pole[0]*y-pole[1]*x;
        double ve=sqrt(ex*ex+ey*ey+ez*ez);
        ex=(ex-x*ez)/std::max(ve,1.e-5);ey=(ey-y*ez)/std::max(ve,1.e-5);
        std::vector<Visit> visits;
        for(int k:window){
            auto &f=sky.images[k];auto &e=f.exposure;double dt=e.t-obs.t;
            if(depth[k]+.8<mag||f.nx<=0)continue;
            double pad=(75.+2400.*std::abs(dt))*as;
            if(x<f.xmin-pad||x>f.xmax+pad||y<f.ymin-pad||y>f.ymax+pad)continue;
            double dz=e.oz-obs.oz;
            double cost=.29*std::abs(std::log(std::max(.003,std::abs(dt))/.09));
            visits.push_back({k,dt,(e.ox-obs.ox-x*dz)/as,(e.oy-obs.oy-y*dz)/as,e.oz,
                              std::min(2.,depth[k]-mag)+.22*(e.night!=obs.night)-cost});
        }
        if(visits.size()<2)return out;
        std::sort(visits.begin(),visits.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        if(visits.size()>20)visits.resize(20);
        std::vector<Pair> pairs;
        auto order=[](const Pair &a,const Pair &b){return a.score>b.score;};
        for(int j=0;j<std::min(9,int(visits.size()));j++){
            auto &s=visits[j];auto &image=sky.images[s.e];
            if(std::abs(s.t)>4.5)continue;
            std::vector<Pair> local;
            for(double radius:{2.45,3.6,7.2}){
                double g=norm/radius,den=std::max(.35,1.-g*s.oz);
                double vc=sqrt(mu/(radius*radius*radius))*norm*s.t/as;
                double dx=(vc*ex-g*s.ox)/den,dy=(vc*ey-g*s.oy)/den;
                double gx=.20*(1.5*vc*ex-g*s.ox+g*s.oz*dx)/den;
                double gy=.20*(1.5*vc*ey-g*s.oy+g*s.oz*dy)/den;
                double low=sq(.37)+sq(.22*vc/den);
                double xx=gx*gx+low,xy=gx*gy,yy=gy*gy+low,det=xx*yy-xy*xy;
                sky.ellipse(image,x+dx*as,y+dy*as,xx,xy,yy,[&](int r){
                    if((sky.flag[r]&1)||!std::isfinite(sky.mag[r]))return;
                    double ux=(sky.xy[2*r]-x)/as,uy=(sky.xy[2*r+1]-y)/as;
                    double vx=ux/s.t,vy=uy/s.t,rate=hypot(vx,vy);
                    // A separation resolved in the actual shutters and a
                    // rate outside the measured distant object's envelope.
                    double maxslow=1.45*norm/(std::max(15.,q.range)-1.)*
                                      (hypot(s.ox,s.oy)/std::abs(s.t)+.0172/as/std::sqrt(std::max(15.,q.range)));
                    if(rate<std::max(175.,maxslow+45.)||rate>3400.||hypot(ux,uy)<1.5)return;
                    double du=ux-dx,dv=uy-dy;
                    double chi=(yy*du*du-2*xy*du*dv+xx*dv*dv)/det;
                    if(chi>9.2)return;
                    double var=.13*.13+sq(1.086/std::max(4.5f,sky.sn[row]))+sq(1.086/std::max(4.5f,sky.sn[r]))
                               +sq(.30*(sky.lever[a]-sky.lever[s.e]));
                    double pm=sq(sky.mag[r]-mag)/var;if(pm>8.5)return;
                    int m=std::clamp(int((mag-10.)*4.),0,79);
                    double odds=-.5*(chi+pm)-std::log(15.75*sqrt(det*var)*std::max(1.e-6,image.rho[m]));
                    if(sky.population&&!sky.population[r])odds-=1.15;
                    local.push_back({r,j,g,vx,vy,odds});
                });
            }
            if(local.size()>25){std::nth_element(local.begin(),local.begin()+25,local.end(),order);local.resize(25);}
            pairs.insert(pairs.end(),local.begin(),local.end());
        }
        if(pairs.size()>100){std::nth_element(pairs.begin(),pairs.begin()+100,pairs.end(),order);pairs.resize(100);}
        for(auto &pair:pairs){
            const auto &b=visits[pair.j];int night0=obs.night,night1=sky.images[b.e].exposure.night;
            std::vector<Trial> hits;
            for(auto &c:visits){
                if(c.e==b.e)continue;
                double s=c.t/b.t;
                if(std::abs(s)>9.||hypot(pair.vx,pair.vy)*std::min(std::abs(c.t),std::abs(c.t-b.t))<1.3)continue;
                auto &image=sky.images[c.e];
                double den=std::max(.4,1.-pair.g*c.oz);
                double rx=(c.ox-s*b.ox)/den,ry=(c.oy-s*b.oy)/den;
                double dx=s*pair.vx*b.t*(1.-pair.g*b.oz)/den-pair.g*rx;
                double dy=s*pair.vy*b.t*(1.-pair.g*b.oz)/den-pair.g*ry;
                double gx=.22*pair.g*rx,gy=.22*pair.g*ry;
                double base=sq(.18)*(1.+s*s+sq(1.-s))+sq(.65*c.t*(c.t-b.t));
                double xx=base+gx*gx,xy=gx*gy,yy=base+gy*gy,det=xx*yy-xy*xy;
                if(sqrt(det)>110.)continue;
                double best=-100.;
                sky.ellipse(image,x+dx*as,y+dy*as,xx,xy,yy,[&](int r){
                    if((sky.flag[r]&1)||!std::isfinite(sky.mag[r]))return;
                    double u=(sky.xy[2*r]-x)/as-dx,v=(sky.xy[2*r+1]-y)/as-dy;
                    double chi=(yy*u*u-2.*xy*u*v+xx*v*v)/det;
                    if(chi>12.)return;
                    double dm=sky.mag[r]-.5*(mag+sky.mag[pair.row]);
                    double variance=.16*.16+sq(1.086/std::max(4.5f,sky.sn[r]))+
                            sq(.28*(sky.lever[c.e]-.5*(sky.lever[a]+sky.lever[b.e])));
                    chi+=dm*dm/variance;if(chi>15.)return;
                    int m=std::clamp(int((mag-10.)*4.),0,79);
                    double lr=-.5*chi-std::log(15.75*sqrt(det*variance)*std::max(1.e-6,image.rho[m]));
                    if(sky.population&&!sky.population[r])lr-=1.15;
                    best=std::max(best,lr);
                });
                if(best>1.)hits.push_back({c.e,best});
            }
            std::sort(hits.begin(),hits.end(),[](auto &a,auto &b){return a.odds>b.odds;});
            std::vector<double> used={0.,b.t};std::vector<int> nights={night0};
            if(night1!=night0)nights.push_back(night1);
            double value=pair.score-std::log(3.+visits.size()),rate=hypot(pair.vx,pair.vy);
            for(auto &h:hits){
                auto &other=sky.images[h.e].exposure;double t=other.t-obs.t;
                bool distinct=true;
                for(double u:used)if(rate*std::abs(t-u)<2.3)distinct=false;
                if(!distinct)continue;
                used.push_back(t);value+=std::min(9.,h.odds)-1.;
                if(std::find(nights.begin(),nights.end(),other.night)==nights.end())nights.push_back(other.night);
                if(used.size()>=6)break;
            }
            auto bound=std::minmax_element(used.begin(),used.end());
            if(used.size()<3||rate*(*bound.second-*bound.first)<8.)continue;
            if(value>out.log_odds)out={float(value),float(rate/24.),int(used.size()),int(nights.size())};
        }
        return out;
    }
};
}
extern "C" int survey_foregrounds(const original::Sky *sky,const foreground_tracks::Query *q,int n,
                    const double *depth,const double *pole,double seconds,int threads,foreground_tracks::Result *out){
    try{
        int ne=sky->images.size();std::vector<std::vector<int>> windows(ne);
        for(int i=0;i<ne;i++)for(int j=0;j<ne;j++){
            double dt=std::abs(sky->images[i].exposure.t-sky->images[j].exposure.t);
            if(dt>.0015&&dt<6.5)windows[i].push_back(j);
        }
        foreground_tracks::Finder finder{*sky,depth,pole};
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int i=0;i<threads;i++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int ix=next.fetch_add(1);if(ix>=n)break;
                out[ix]=finder.run(q[ix],windows[sky->ex[q[ix].row]]);
            }
        });
        for(auto &w:pool)w.join();return std::min(next.load(),n);
    }catch(...){return -1;}
}
