// Build the common list of possible returns once for every first direction.
// All choices of the second, third and fourth nights share this list.
namespace pencil {
struct Field {
    Visit e;
    std::vector<int> ids,offset;
    int bx=0,by=0,nx=0,ny=0;
    double xmin=1e8,xmax=-1e8,ymin=1e8,ymax=-1e8;
    std::array<double,60> lograte{};
    static constexpr double cell=170./arc;
    Field(const Visit &v):e(v){}
    void build(const Point *p){
        if(ids.empty())return;
        std::array<int,60> freq{};
        for(int id:ids){
            const Point &a=p[id];xmin=std::min(xmin,double(a.x));xmax=std::max(xmax,double(a.x));
            ymin=std::min(ymin,double(a.y));ymax=std::max(ymax,double(a.y));
            ++freq[std::clamp(int((a.mag-10.)*3.),0,59)];
        }
        bx=int(floor(xmin/cell));by=int(floor(ymin/cell));
        nx=int(floor(xmax/cell))-bx+1;ny=int(floor(ymax/cell))-by+1;
        if(size_t(nx)*ny>2000000){nx=0;return;}
        auto key=[&](int j){return (int(floor(p[j].x/cell))-bx)*ny+int(floor(p[j].y/cell))-by;};
        offset.resize(nx*ny+1);
        for(int j:ids)++offset[key(j)+1];
        int occupied=0;
        for(size_t k=1;k<offset.size();++k){occupied+=offset[k]>0;offset[k]+=offset[k-1];}
        auto cursor=offset;std::vector<int> order(ids.size());
        for(int j:ids)order[cursor[key(j)]++]=j;
        ids.swap(order);
        double area=std::max(3600.,std::min((xmax-xmin)*(ymax-ymin)*arc*arc,
                occupied*square(cell*arc)/std::max(.12,1.-exp(-ids.size()/double(nx*ny)))));
        for(int k=0;k<60;++k){
            double count=freq[k]*.48+.22*(freq[std::max(0,k-1)]+freq[std::min(59,k+1)])
                              +.04*(freq[std::max(0,k-2)]+freq[std::min(59,k+2)]);
            lograte[k]=log(fmax(2.e-9,(3.*count+.55)/area));
        }
    }
};
struct Ranked {double merit;Edge edge;};
inline bool stronger(const Ranked &a,const Ranked &b){return a.merit>b.merit;}
struct Catalog {
    const Point *p;
    std::vector<Field> field;
    double pole_y,pole_z;
    Catalog(const Point *s,int n,const Visit *ex,int ne,double py,double pz):p(s),pole_y(py),pole_z(pz){
        for(int j=0;j<ne;++j)field.emplace_back(ex[j]);
        for(int j=0;j<n;++j)if(s[j].ex>=0&&s[j].ex<ne)field[s[j].ex].ids.push_back(j);
        for(auto &f:field)f.build(p);
    }
    void gather(int id,double radius,int family,Origin &origin,Edge *out,int &count,int cap)const{
        const Point &a=p[id];const auto &s=field[a.ex].e;
        Track track(a,s,radius,pole_y,pole_z,family==2);
        // Point variance includes intrinsic photometric scatter. For the
        // astrometric error take the source's flux error above that floor.
        double ast=sqrt(.115*.115+.72*fmax(0.,double(a.var)-square(.14)))/(arc*track.vc);
        origin={float(track.nominal),float(track.vc),float(track.hz),float(ast),a.q,float(1./track.norm),a.row};
        std::vector<Ranked> choices;choices.reserve(160);
        for(auto &im:field){
            const auto &e=im.e;double dt=e.t-s.t;
            if(!im.nx||e.night==s.night||fabs(dt)<.52||e.depth<a.mag-1.05)continue;
            auto mid=track.equation(e,0.,0.);
            double scale=fabs(1./mid.factor);
            // Test footprints before computing the tight *oriented* ellipse.
            double wide=.0015+.62*track.nominal+(family==2?2.3:1.04)*scale;
            if(mid.mean.x<im.xmin-wide||mid.mean.x>im.xmax+wide||
               mid.mean.y<im.ymin-wide||mid.mean.y>im.ymax+wide)continue;
            constexpr double dg=.085,dh=.16;
            auto lo=track.equation(e,-dg,0.),hi=track.equation(e,dg,0.);
            auto hr=track.equation(e,0.,dh),hr_hi=track.equation(e,dg,dh);
            double sw=family==0?.19:.29,sh=family==0?.20:.36;
            V gd{(hi.mean.x-lo.mean.x)*sw/(2.*dg),(hi.mean.y-lo.mean.y)*sw/(2.*dg)};
            V hd{(hr.mean.x-mid.mean.x)*sh/dh,(hr.mean.y-mid.mean.y)*sh/dh};
            double xx=square(gd.x)+square(hd.x)+square((family==2?.73:.24)*scale)+square(.34/arc);
            double yy=square(gd.y)+square(hd.y)+square((family==2?.73:.33)*scale)+square(.34/arc),xy=gd.x*gd.y+hd.x*hd.y;
            double det=xx*yy-xy*xy;if(!(det>0.))continue;
            double wx=3.05*sqrt(xx),wy=3.05*sqrt(yy);
            if(mid.mean.x<im.xmin-wx||mid.mean.x>im.xmax+wx||
               mid.mean.y<im.ymin-wy||mid.mean.y>im.ymax+wy)continue;
            int x0=std::max(0,int(floor((mid.mean.x-wx)/Field::cell))-im.bx);
            int x1=std::min(im.nx-1,int(floor((mid.mean.x+wx)/Field::cell))-im.bx);
            double slope=xy/xx,band=3.05*sqrt(det/xx)+.5*fabs(slope)*Field::cell;
            size_t before=choices.size();
            for(int x=x0;x<=x1;++x){
                double y=mid.mean.y+((im.bx+x+.5)*Field::cell-mid.mean.x)*slope;
                int y0=std::max(0,int(floor(fmax(y-band,mid.mean.y-wy)/Field::cell))-im.by);
                int y1=std::min(im.ny-1,int(floor(fmin(y+band,mid.mean.y+wy)/Field::cell))-im.by);
                if(y0>y1)continue;
                for(int k=im.offset[x*im.ny+y0];k<im.offset[x*im.ny+y1+1];++k){
                    const auto &b=p[im.ids[k]];
                    double photo=square(a.mag-b.mag)/(a.var+b.var+square(.27*(e.lever-s.lever)));
                    if(photo>11.5)continue;
                    double xoff=b.x-mid.mean.x,yoff=b.y-mid.mean.y;
                    double chi=xoff*xoff/xx+square(yoff-xoff*slope)*xx/det;
                    if(chi>9.3)continue;
                    V m=mid.velocity(b),u=hi.velocity(b),l=lo.velocity(b),h=hr.velocity(b),gh=hr_hi.velocity(b);
                    double rho=im.lograte[std::clamp(int((b.mag-10.)*3.),0,59)];
                    double noise=sqrt(square(.115)+.72*fmax(0.,double(b.var)-square(.14)))/(arc*track.vc*fabs(dt));
                    double mark=b.q-.28*photo;
                    Edge edge{float(m.x),float(m.y),float((u.x-l.x)/(2.*dg)),float((u.y-l.y)/(2.*dg)),
                        float((h.x-m.x)/dh),float((h.y-m.y)/dh),
                        float((u.x+l.x-2.*m.x)/(2.*dg*dg)),float((u.y+l.y-2.*m.y)/(2.*dg*dg)),
                        float((gh.x-u.x-h.x+m.x)/(dg*dh)),float((gh.y-u.y-h.y+m.y)/(dg*dh)),
                        float(dt),float(noise),float(rho),float(mark),b.mag,float(e.lever),b.var,b.row,e.night};
                    double rank=mark-.46*chi-.51*(rho+.5*log(det)+2.*log(arc));
                    choices.push_back({rank,edge});
                }
            }
            // Cap individual crowded exposures but retain a representation
            // of multiple visits and independent observing seasons.
            if(choices.size()>before+24){
                std::nth_element(choices.begin()+before,choices.begin()+before+24,choices.end(),stronger);
                choices.resize(before+24);
            }
        }
        std::sort(choices.begin(),choices.end(),stronger);
        std::vector<int> nights(field.size()+1,0);
        for(auto &c:choices)c.merit-=1.6*log1p(nights[c.edge.night]++);
        size_t n=std::min<size_t>(cap,choices.size());
        std::partial_sort(choices.begin(),choices.begin()+n,choices.end(),stronger);
        for(size_t i=0;i<n;++i)out[i]=choices[i].edge;
        count=int(n);
    }
};
}
extern "C" void *pencil_index(const pencil::Point *p,int n,const pencil::Visit *e,int ne,double py,double pz){
    try{return new pencil::Catalog(p,n,e,ne,py,pz);}catch(...){return nullptr;}
}
extern "C" void pencil_free(pencil::Catalog *c){delete c;}
extern "C" int pencil_edges(const pencil::Catalog *c,const int *ids,int n,double radius,int family,
        int cap,int threads,pencil::Origin *orig,pencil::Edge *out,int *counts){
    try{
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int i=0;i<threads;++i)workers.emplace_back([&]{
            for(int k;(k=next.fetch_add(1))<n;)c->gather(ids[k],radius,family,orig[k],out+k*cap,counts[k],cap);
        });
        for(auto &t:workers)t.join();
        return n;
    }catch(...){return -1;}
}
