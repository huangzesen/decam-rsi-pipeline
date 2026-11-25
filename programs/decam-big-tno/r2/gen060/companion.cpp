// Catalogue likelihood of companions on other nights. Coordinates and
// covariance here are in arcseconds on the region's tangent plane.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace {
constexpr double arc = 206264.80624709636;
constexpr double cell = 36.;
constexpr int bins = 42;
struct Point { float x,y,m,sn,mark; int e,id; };
struct Date { double t,ox,oy,oz,depth,color; int night,pad; };
struct View {
    int a,b;
    double at[2],mx[2],my[2],xx,xy,yy,g,sg,gx,gy,weight;
};
struct Hit {int row;float score,g,sigma;};
struct Seed {int row[3]; float score,g;};
constexpr double sq(double x){return x*x;}
struct Aperture:View {
    double det,wx,wy,tilt,band,inv,astro;
    explicit Aperture(const View &v):View(v),det(xx*yy-xy*xy),wx(3.*std::sqrt(xx)),
        wy(3.*std::sqrt(yy)),tilt(xy/xx),band(3.*std::sqrt(det/xx)+.5*std::abs(tilt)*cell),
        inv(xx/det),astro(weight-std::log(6.283185307*std::sqrt(det))){}
};
int mbin(double mag){return std::clamp(int((mag-12.)*2.5),0,bins-1);}
struct Field {
    std::vector<Point> p;
    std::vector<int> offsets,local;
    std::array<double,bins> mean{};
    double xlo=1e12,ylo=1e12,xhi=-1e12,yhi=-1e12;
    int nx=0,ny=0,hy=0;
    void build(const std::vector<Point> &input) {
        if(input.empty())return;
        for(auto v:input) {
            xlo=std::min(xlo,double(v.x)); ylo=std::min(ylo,double(v.y));
            xhi=std::max(xhi,double(v.x)); yhi=std::max(yhi,double(v.y));
        }
        nx=int((xhi-xlo)/cell)+1; ny=int((yhi-ylo)/cell)+1;
        if(size_t(nx)*ny>2200000){nx=ny=0;return;}
        offsets.resize(nx*ny+1);
        const int hx=(nx+9)/10;hy=(ny+9)/10;
        local.resize(hx*hy*bins);
        std::vector<int> coarse(hx*hy);
        auto coord=[&](const Point &v){return std::array<int,2>{
            int((v.x-xlo)/cell),int((v.y-ylo)/cell)};};
        for(auto v:input){
            auto ij=coord(v);
            offsets[ij[0]*ny+ij[1]+1]++;
            int h=ij[0]/10*hy+ij[1]/10,b=mbin(v.m);
            coarse[h]++;local[h*bins+b]++;mean[b]++;
        }
        int used=0;for(int c:coarse)if(c)used++;
        auto hist=mean;
        for(int b=0;b<bins;b++)mean[b]=(.5*hist[b]+.25*hist[std::max(0,b-1)]+
            .25*hist[std::min(b+1,bins-1)]+1.)/std::max(1,used);
        for(size_t i=1;i<offsets.size();i++)offsets[i]+=offsets[i-1];
        auto slots=offsets;p.resize(input.size());
        for(auto v:input){
            auto ij=coord(v);p[slots[ij[0]*ny+ij[1]]++]=v;
        }
    }
    double density(const Point &v) const {
        int i=int((v.x-xlo)/cell),j=int((v.y-ylo)/cell),b=mbin(v.m);
        int h=(i/10*hy+j/10)*bins;
        double count=.50*local[h+b]+.25*(local[h+std::max(0,b-1)]+local[h+std::min(b+1,bins-1)]);
        return (count+.8*mean[b]+.24)*2.5/(1.8*sq(cell*10.));
    }
    template<typename F> void visit(double x,double y,const Aperture &v,F use) const {
        double wx=v.wx,wy=v.wy;
        if(!nx||x+wx<xlo||x-wx>xhi||y+wy<ylo||y-wy>yhi)return;
        double tilt=v.tilt,band=v.band;
        int left=std::max(0,int(std::floor((x-wx-xlo)/cell)));
        int right=std::min(nx-1,int((x+wx-xlo)/cell));
        for(int i=left;i<=right;i++){
            double mid=y+tilt*(xlo+(i+.5)*cell-x);
            int lo=std::max(0,int(std::floor((std::max(y-wy,mid-band)-ylo)/cell)));
            int hi=std::min(ny-1,int(std::floor((std::min(y+wy,mid+band)-ylo)/cell)));
            if(hi<lo)continue;
            for(int k=offsets[i*ny+lo];k<offsets[i*ny+hi+1];k++){
                const Point &z=p[k];double dx=z.x-x,dy=z.y-y;
                double chi=dx*dx/v.xx+sq(dy-tilt*dx)*v.inv;
                if(chi<9.)use(z,dx,dy,chi);
            }
        }
    }
};

struct Discovery {
    const Date *dates;
    const Point *points;
    const std::vector<Field> &field;
    const std::vector<std::vector<Aperture>> &views;
    const std::vector<float> &background;
    float *scores;
    Hit *pairs;
    Seed *triples;
    void source(const Point &a) const {
        const auto &da=dates[a.e];std::array<Hit,10> hits;int n=0;
        double bestpair=-100.;
        for(const Aperture &v:views[a.e]){
            const auto &db=dates[v.b];
            if(a.m>db.depth+1.1)continue;
            const auto &f=field[v.b];
            double x=a.x+v.at[0]+v.mx[0]*a.x+v.my[0]*a.y;
            double y=a.y+v.at[1]+v.mx[1]*a.x+v.my[1]*a.y;
            double colorvar=sq(.28*(db.color-da.color));
            f.visit(x,y,v,[&](const Point &b,double dx,double dy,double chi){
                double dm=b.m-a.m;
                if(std::abs(dm)>1.3||sq(b.x-a.x)+sq(b.y-a.y)<sq(1.8))return;
                double vmag=.040+sq(1.086/a.sn)+sq(1.086/b.sn)+colorvar;
                double flux=-.5*sq(dm)/vmag-.5*std::log(6.28318*vmag);
                double merit=v.astro-.5*chi+flux+background[b.id];
                if(merit< -3.8)return;
                int existing=-1;
                for(int j=0;j<n;j++)if(dates[points[hits[j].row].e].night==db.night){
                    existing=j;break;
                }
                if(existing<0&&n==int(hits.size()))
                    existing=std::min_element(hits.begin(),hits.end(),[](auto u,auto z){return u.score<z.score;})-hits.begin();
                if(existing>=0&&merit<=hits[existing].score)return;
                int slot=existing>=0?existing:n++;
                hits[slot]={b.id,float(merit),float(v.g+v.gx*dx+v.gy*dy),float(v.sg)};
            });
        }
        if(!n)return;
        std::sort(hits.begin(),hits.begin()+n,[](auto p,auto q){return p.score>q.score;});
        const double trials=std::log1p(.18*views[a.e].size());
        for(int j=0;j<std::min(n,3);j++){
            auto &pair=pairs[3*a.id+j];pair=hits[j];pair.score+=a.mark*.45-trials;
        }
        bestpair=hits[0].score-trials;
        // Absolute likelihood is also informative below the threshold for
        // selecting an isolated pair: two such nights can constrain range.
        double best3=-1e10;
        for(int j=1;j<n;j++)for(int i=0;i<j;i++){
            auto b=hits[i],c=hits[j];
            if(std::abs(dates[points[b.row].e].t-da.t)<std::abs(dates[points[c.row].e].t-da.t))std::swap(b,c);
            auto &pb=points[b.row],&pc=points[c.row];
            auto &db=dates[pb.e],&dc=dates[pc.e];
            double dt=dc.t-da.t,u=dt/(db.t-da.t);
            // Inverse range makes the heliocentric positions linear in
            // time. The remaining uncertainty along the parallax vector
            // is common to both coordinates.
            double rx=pc.x-a.x-u*(pb.x-a.x),ry=pc.y-a.y-u*(pb.y-a.y);
            double px=(dc.ox-da.ox-u*(db.ox-da.ox))*arc-
                (pc.x*dc.oz-a.x*da.oz-u*(pb.x*db.oz-a.x*da.oz));
            double py=(dc.oy-da.oy-u*(db.oy-da.oy))*arc-
                (pc.y*dc.oz-a.y*da.oz-u*(pb.y*db.oz-a.y*da.oz));
            double g=std::clamp(double(b.g),.00015,.061);
            double floor=.18*.18*(1.+u*u+sq(1.-u))+.04;
            double sr=.34*std::sqrt(.0002963*g*g*g);
            double hx=(pc.x-pb.x)*dt*sr,hy=(pc.y-pb.y)*dt*sr;
            double xx=floor+sq(b.sigma*px)+hx*hx,xy=sq(b.sigma)*px*py+hx*hy;
            double yy=floor+sq(b.sigma*py)+hy*hy,det=xx*yy-xy*xy;
            rx+=g*px;ry+=g*py;
            double wx=(yy*rx-xy*ry)/det,wy=(xx*ry-xy*rx)/det;
            double chi=rx*wx+ry*wy;if(chi>18.)continue;
            double gamma=g-sq(b.sigma)*(px*wx+py*wy);
            if(gamma<=.0001||gamma>.066)continue;
            double vmag=.037+sq(1.086/pc.sn)+.25*(sq(1.086/a.sn)+sq(1.086/pb.sn))+
                         sq(.28*(dc.color-.5*(da.color+db.color)));
            double photo=-.5*sq(pc.m-.5*(a.m+pb.m))/vmag-.5*std::log(6.28318*vmag);
            double score=b.score-.5*chi-std::log(6.28318*std::sqrt(det))+
                photo+background[pc.id]-2.*trials;
            if(score>best3){
                best3=score;triples[a.id]={{a.id,b.row,c.row},float(score+.45*a.mark),float(gamma)};
            }
        }
        scores[a.id]=std::max(0.,std::max(bestpair-2.5,.8*(best3-7.)));
    }
};
}

extern "C" int night_companion_alternatives(const Point *points,int n,const Date *dates,int ne,
    const View *maps,int nv,int threads,float *score,Hit *pair,Seed *triple) {
    try {
        std::vector<Field> fields(ne);std::vector<std::vector<Point>> parts(ne);
        std::vector<std::vector<Aperture>> views(ne);
        for(int i=0;i<n;i++)parts[points[i].e].push_back(points[i]);
        for(int e=0;e<ne;e++){
            fields[e].build(parts[e]);std::vector<Point>().swap(parts[e]);
        }
        std::vector<float> bg(n);
        for(auto &f:fields)for(auto &p:f.p)bg[p.id]=.45*p.mark-std::log(f.density(p));
        for(int i=0;i<nv;i++)views[maps[i].a].emplace_back(maps[i]);
        Discovery search{dates,points,fields,views,bg,score,pair,triple};
        std::atomic<int> next{0};std::vector<std::thread> workers;
        for(int k=0;k<threads;k++)workers.emplace_back([&]{
            for(int start;(start=next.fetch_add(192))<n;)
                for(int j=start;j<std::min(n,start+192);j++)search.source(points[j]);
        });
        for(auto &w:workers)w.join();
        return 0;
    }catch(...){return -1;}
}
