namespace pencil {
struct TileGrid {
    float xmin,xmax,ymin,ymax,density[60];
    int bx,by,nx,ny,offset;
};
constexpr double pitch=108./arc;
struct Pair {Point a,b,peer; float g;};
struct Hit {int id;float merit;};
VP double positional_noise(const Point &p,double vc){
    return sqrt(square(.115)+.72*fmax(0.,double(p.var)-square(.14)))/(arc*vc);
}
VP Edge measured(const Track &track,const Point &b,const Visit &e,double rho){
    constexpr double dg=.075,dh=.16;
    V m=track.equation(e,0.,0.).velocity(b),u=track.equation(e,dg,0.).velocity(b),
      l=track.equation(e,-dg,0.).velocity(b),h=track.equation(e,0.,dh).velocity(b),
      gh=track.equation(e,dg,dh).velocity(b);
    double dt=e.t-track.start.t;
    return {float(m.x),float(m.y),float((u.x-l.x)/(2.*dg)),float((u.y-l.y)/(2.*dg)),
            float((h.x-m.x)/dh),float((h.y-m.y)/dh),
            float((u.x+l.x-2.*m.x)/(2.*dg*dg)),float((u.y+l.y-2.*m.y)/(2.*dg*dg)),
            float((gh.x-u.x-h.x+m.x)/(dg*dh)),float((gh.y-u.y-h.y+m.y)/(dg*dh)),
            float(dt),float(positional_noise(b,track.vc)/fabs(dt)),float(rho),b.q,b.mag,
            float(e.lever),b.var,b.row,e.night};
}
struct Condition {
    Track track;Origin a;Edge b;Normal base;Posterior post;
    double mag,colour,mean_lever,colourvar;
    VP Condition(const Pair &q,const Visit *e,double py,double pz):
            track(q.a,e[q.a.ex],sqrt(1.+q.a.x*q.a.x+q.a.y*q.a.y)/q.g,py,pz){
        a={float(track.nominal),float(track.vc),float(track.hz),float(positional_noise(q.a,track.vc)),
           q.a.q,float(1./track.norm),q.a.row};
        b=measured(track,q.b,e[q.b.ex],0.);
        base=prior(b,.25,.32);post=base.fit();
        if(q.peer.row>=0&&fabs(e[q.peer.ex].t-track.start.t)>.0005){
            Edge peer=measured(track,q.peer,e[q.peer.ex],0.);
            double variance=velocity_variance(a,b,peer)+.001;
            Normal trial=base;
            equation(trial,b,peer,post.g,post.h,1./variance);
            auto next=trial.fit();
            // Allow a companion selected from a crowded burst to be unrelated.
            if(next.chi<base.fit().chi+10.){base=trial;post=next;}
        }
        double wa=1./q.a.var,wb=1./q.b.var,dl=e[q.b.ex].lever-track.start.lever;
        mag=(wa*q.a.mag+wb*q.b.mag)/(wa+wb);
        mean_lever=(wa*track.start.lever+wb*e[q.b.ex].lever)/(wa+wb);
        colourvar=1./(1./square(.28)+square(dl)/(q.a.var+q.b.var));
        colour=colourvar*(q.b.mag-q.a.mag)*dl/(q.a.var+q.b.var);
    }
    VP V forecast(const Visit &e,double g,double h)const{
        Transfer t=track.equation(e,g,h);
        V v=at(b,g,h);
        return {t.mean.x+v.x/t.factor,t.mean.y+v.y/t.factor};
    }
    VP void find(const Point *sources,const int *offset,const TileGrid &im,const Visit &e,Hit *best)const{
        best[0]={-1,-1.e18f};best[1]=best[0];
        double m=mag+colour*(e.lever-mean_lever);
        if(!im.nx||e.night==track.start.night||e.night==b.night||e.depth<m-.91||
           fabs(post.g)>.6||fabs(post.h)>.88)return;
        double g=post.g,h=post.h;
        V center=forecast(e,g,h);
        double dt=e.t-track.start.t,ratio=dt/b.dt;
        double coarse=.002+.00010*fabs(dt)+.000003*fabs(ratio);
        if(center.x+coarse<im.xmin||center.x-coarse>im.xmax||
           center.y+coarse<im.ymin||center.y-coarse>im.ymax)return;
        constexpr double eps=.002;
        auto gu=forecast(e,g+eps,h),hv=forecast(e,g,h+eps);
        V u{(gu.x-center.x)/eps,(gu.y-center.y)/eps},v{(hv.x-center.x)/eps,(hv.y-center.y)/eps};
        double cx=(post.c*u.x-post.b*v.x)/post.det,dx=(post.a*v.x-post.b*u.x)/post.det;
        double cy=(post.c*u.y-post.b*v.y)/post.det,dy=(post.a*v.y-post.b*u.y)/post.det;
        double phase=fabs(dt)*track.vc;
        double propagation=.19+arc*phase*(.00016+.0008*phase);
        double noise=square(propagation/arc)+square(a.noise*track.vc*(1.-ratio))+
                     square(track.vc*b.noise*dt);
        double xx=noise+cx*u.x+dx*v.x,xy=cx*u.y+dx*v.y,yy=noise+cy*u.y+dy*v.y;
        double det=fmax(noise*noise,xx*yy-xy*xy),wx=3.65*sqrt(xx),wy=3.65*sqrt(yy);
        if(center.x+wx<im.xmin||center.x-wx>im.xmax||center.y+wy<im.ymin||center.y-wy>im.ymax)return;
        int mb=int((m-10.)*3.);mb=mb<0?0:(mb>59?59:mb);
        double logarea=.5*log(det)+2.*log(arc),logbg=im.density[mb];
        // This visit would supply hundreds of equally admissible sources
        // for one pair. A closer visit can narrow that footprint first.
        if(logarea+logbg>4.85)return;
        int x0=int(floor((center.x-wx)/pitch))-im.bx,x1=int(floor((center.x+wx)/pitch))-im.bx;
        x0=x0<0?0:x0;x1=x1>=im.nx?im.nx-1:x1;
        double tilt=xy/xx,band=3.65*sqrt(det/xx)+.5*fabs(tilt)*pitch;
        double vm=.018+1./(1./track.p.var+1./b.variance)+colourvar*square(e.lever-mean_lever);
        for(int col=x0;col<=x1;++col){
            double mid=center.y+((im.bx+col+.5)*pitch-center.x)*tilt;
            int y0=int(floor(fmax(center.y-wy,mid-band)/pitch))-im.by;
            int y1=int(floor(fmin(center.y+wy,mid+band)/pitch))-im.by;
            y0=y0<0?0:y0;y1=y1>=im.ny?im.ny-1:y1;
            if(y0>y1)continue;
            int start=im.offset+col*im.ny;
            for(int k=offset[start+y0];k<offset[start+y1+1];++k){
                const Point &s=sources[k];
                double dm=s.mag-m,pm=dm*dm/(s.var+vm);
                if(pm>14.)continue;
                double x=s.x-center.x,y=s.y-center.y;
                double chi=x*x/xx+square(y-x*tilt)*xx/det;
                if(chi>13.3)continue;
                double val=s.q-.5*(chi+pm+log(vm+s.var))-logarea-logbg;
                if(val>best[1].merit){
                    Hit z{k,float(val)};
                    if(val>best[0].merit){best[1]=best[0];best[0]=z;}else best[1]=z;
                }
            }
        }
    }
};
}
#ifdef __CUDACC_RTC__
extern "C" __global__ void conditioned_visits(const pencil::Point *p,const pencil::TileGrid *grid,
          const int *offset,const pencil::Visit *e,int ne,const pencil::Pair *pairs,int n,
          double py,double pz,pencil::Hit *hits){
    int pair=blockIdx.x,visit=threadIdx.x+blockIdx.y*blockDim.x;
    if(pair>=n||visit>=ne)return;
    pencil::Condition cond(pairs[pair],e,py,pz);
    cond.find(p,offset,grid[visit],e[visit],hits+2*(pair*ne+visit));
}
#else
extern "C" void conditioned_host(const pencil::Point *p,const pencil::TileGrid *grid,
          const int *offset,const pencil::Visit *e,int ne,const pencil::Pair *pairs,int n,
          double py,double pz,int threads,pencil::Hit *hits){
    std::atomic<int> next{0};std::vector<std::thread> jobs;
    for(int th=0;th<threads;++th)jobs.emplace_back([&]{
        for(int i;(i=next.fetch_add(1))<n;){
            pencil::Condition c(pairs[i],e,py,pz);
            for(int j=0;j<ne;++j)c.find(p,offset,grid[j],e[j],hits+2*(i*ne+j));
        }
    });
    for(auto &job:jobs)job.join();
}
namespace pencil {
struct MeasuredThird {int id;Edge edge;Trial trial;double merit,leverage;};
void reports(const Point *p,const TileGrid *grid,const Visit *e,int ne,const Pair &pair,
             double py,double pz,const Hit *hits,Fit *out){
    for(int j=0;j<6;++j)out[j]=empty_fit();
    std::vector<Hit> found;
    for(int j=0;j<2*ne;++j)if(hits[j].id>=0)found.push_back(hits[j]);
    if(found.empty())return;
    std::sort(found.begin(),found.end(),[](auto a,auto b){return a.merit>b.merit;});
    Condition c(pair,e,py,pz);const auto &a=c.a;const auto &b=c.b;
    std::vector<MeasuredThird> thirds;
    std::vector<int> per_night(ne,0);
    for(auto hit:found){
        const Point &source=p[hit.id];auto &ex=e[source.ex];
        if(per_night[ex.night]>=6)continue;
        double rho=grid[source.ex].density[std::clamp(int((source.mag-10.)*3.),0,59)];
        Edge edge=measured(c.track,source,ex,rho);
        Trial trial;
        if(!triple(a,b,edge,c.base,c.post.g,c.post.h,trial))continue;
        ++per_night[ex.night];
        double span=fmax(fabs(b.dt),fmax(fabs(edge.dt),fabs(edge.dt-b.dt)));
        // Well-separated measurements with leverage on both range and
        // radial velocity give a smaller prediction on other nights.
        double info=.28*log(trial.p.det/c.post.det)+.38*log1p(span/120.);
        thirds.push_back({hit.id,edge,trial,trial.merit,info});
        if(thirds.size()>=80)break;
    }
    if(thirds.empty())return;
    double prior=a.mark+b.mark+.55*log1p(fabs(b.dt)/5.);
    for(size_t i=0;i<thirds.size();++i){
        auto &u=thirds[i];double score=prior+u.merit;
        if(c.track.start.night!=b.night){
            auto z=proposal(a,b,u.edge,-1,u.trial.p.g,u.trial.p.h,score);
            if(z.merit>out[0].merit)out[0]=z;
            z.merit+=u.leverage;
            if(z.merit>out[1].merit)out[1]=z;
        }
        for(size_t j=0;j<i;++j){
            auto &v=thirds[j];double dg=0.,h=0.;
            double added=fourth(a,b,u.edge,v.edge,u.trial,dg,h);
            if(added<-1.e10)continue;
            double merit=score+added;
            if(merit<=out[5].merit)continue;
            auto z=proposal(a,b,u.edge,v.edge.row,dg,h,merit,v.edge.dt);
            int k=5;
            while(k>2&&z.merit>out[k-1].merit){out[k]=out[k-1];--k;}
            out[k]=z;
        }
    }
}
}
extern "C" void conditioned_fits(const pencil::Point *p,const pencil::TileGrid *grid,const pencil::Visit *e,int ne,
        const pencil::Pair *pairs,int n,double py,double pz,int threads,const pencil::Hit *hits,pencil::Fit *out){
    std::atomic<int> next{0};std::vector<std::thread> jobs;
    for(int th=0;th<threads;++th)jobs.emplace_back([&]{
        for(int i;(i=next.fetch_add(1))<n;)
            pencil::reports(p,grid,e,ne,pairs[i],py,pz,hits+2*i*ne,out+6*i);
    });
    for(auto &job:jobs)job.join();
}
#endif
