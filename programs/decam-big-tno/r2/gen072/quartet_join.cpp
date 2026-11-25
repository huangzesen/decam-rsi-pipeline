// Join angular displacements from two pairs of shutters in common orbit coordinates.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "quartet_fit.hpp"

namespace quartet {
struct Point {float x,y,mag,sigma,quality; int row;float rawx,rawy;int exposure;};
struct Pair {
    float pos[4], mag, variance, quality;
    int a,b;
    float native[4];int e0,e1;
};
struct Mapping {double center[4], linear[4][4], polynomial[15][4];};
struct Signature {float z[4];};
struct Seed {int rows[4]; float gamma,score;};
struct Image {
    std::vector<Point> pts;
    std::vector<int> start;
    double x0=0., width=50.;
    int nx=0;
    int ix(double x) const {return std::clamp(int(std::floor((x-x0)/width)),0,nx-1);}
    void build(const Point* lo,const Point* hi) {
        if(lo==hi)return;
        pts.assign(lo,hi);
        auto bounds=std::minmax_element(lo,hi,[](auto &a,auto &b){return a.x<b.x;});
        x0=bounds.first->x-.1;
        width=std::max(18.,(bounds.second->x-x0)/std::max(12.,1.8*std::sqrt(double(pts.size()))));
        nx=1+int((bounds.second->x-x0)/width);
        std::sort(pts.begin(),pts.end(),[&](auto &a,auto &b){
            int u=ix(a.x),v=ix(b.x);
            return u!=v?u<v:a.y<b.y;
        });
        start.assign(nx+1,0);
        for(auto &p:pts)++start[ix(p.x)+1];
        std::partial_sum(start.begin(),start.end(),start.begin());
    }
};
struct Levels {
    std::array<Image,4> grid;
    double area=1.;
    void build(const Point *lo,const Point *hi){
        grid[0].build(lo,hi);
        if(lo==hi)return;
        auto yy=std::minmax_element(lo,hi,[](auto &a,auto &b){return a.y<b.y;});
        area=std::max(10000.,.78*(yy.second->y-yy.first->y)*(grid[0].nx*grid[0].width));
        std::vector<Point> pts(lo,hi);
        for(int k=1;k<4;k++){
            size_t n=std::min(pts.size(),std::max<size_t>(60,pts.size()/3));
            if(n<pts.size()){
                std::nth_element(pts.begin(),pts.begin()+n,pts.end(),[](auto &a,auto &b){return a.quality>b.quality;});
                pts.resize(n);
            }
            grid[k].build(pts.data(),pts.data()+pts.size());
        }
    }
};
struct Field {std::vector<Levels> im;};

static inline double sq(double x){return x*x;}
static void generate(const Image &first,const Image &other,const double *model,
                     double color,int cap,std::vector<Pair> &out){
    if(first.pts.empty()||other.pts.empty())return;
    const double xx=model[8],xy=model[9],yy=model[10];
    const double det=xx*yy-xy*xy;
    if(!(det>0.)||!std::isfinite(det))return;
    double sx=std::sqrt(xx),slope=xy/xx,conditional=std::sqrt(det/xx);
    const double limit=3.15, reach=limit*sx;
    auto better=[](auto &a,auto &b){return a.quality>b.quality;};
    // Reservoirs bounded per source allow an independent second chord to
    // resolve an ambiguous local pairing in a crowded range interval.
    size_t per=std::clamp<size_t>(size_t(9*cap/std::max<size_t>(1,first.pts.size())),8,100);
    std::vector<Pair> local;
    for(auto &p:first.pts){
        double x=p.x-model[0],y=p.y-model[1];
        double mx=p.x+model[2]+x*model[4]+y*model[6];
        double my=p.y+model[3]+x*model[5]+y*model[7];
        if(mx+reach<other.x0||mx-reach>other.x0+other.width*other.nx)continue;
        int l=other.ix(mx-reach),h=other.ix(mx+reach);
        local.clear();
        for(int ix=l;ix<=h;ix++){
            double xlo=std::max(-reach,other.x0+ix*other.width-mx);
            double xhi=std::min(reach,other.x0+(ix+1)*other.width-mx);
            double ylo=my+std::min(slope*xlo,slope*xhi)-limit*conditional;
            double yhi=my+std::max(slope*xlo,slope*xhi)+limit*conditional;
            auto end=other.pts.begin()+other.start[ix+1];
            auto it=std::lower_bound(other.pts.begin()+other.start[ix],end,ylo,
                                     [](const Point &v,double val){return v.y<val;});
            for(;it!=end&&it->y<=yhi;++it){
                const auto &q=*it;
                double dx=q.x-mx,dy=q.y-my;
                double chi=dx*dx/xx+sq((dy-slope*dx)/conditional);
                if(chi>limit*limit)continue;
                double var=sq(p.sigma)+sq(q.sigma)+.036+sq(.28*color);
                double dm=sq(p.mag-q.mag)/var;
                if(dm>12.25)continue;
                float merit=float(.54*(p.quality+q.quality)-.34*chi-.44*dm);
                local.push_back({{p.x,p.y,q.x-p.x,q.y-p.y},.5f*(p.mag+q.mag),
                                 float(.25*var+.02),merit,p.row,q.row,
                                 {p.rawx,p.rawy,q.rawx-p.rawx,q.rawy-p.rawy},p.exposure,q.exposure});
            }
        }
        if(local.size()>per){
            std::nth_element(local.begin(),local.begin()+per,local.end(),better);local.resize(per);
        }
        out.insert(out.end(),local.begin(),local.end());
        if(int(out.size())>3*cap){
            std::nth_element(out.begin(),out.begin()+cap,out.end(),better);out.resize(cap);
        }
    }
    if(int(out.size())>cap){
        std::nth_element(out.begin(),out.begin()+cap,out.end(),better);out.resize(cap);
    }
}

static inline Signature transform(const Pair &p,const Mapping &m){
    double s[4]={},w[4]={};
    for(int i=0;i<4;i++){
        double d=p.pos[i]-m.center[i];
        for(int k=0;k<4;k++)s[k]+=d*m.linear[i][k];
    }
    for(int k=0;k<4;k++)w[k]=m.polynomial[0][k];
    for(int i=0;i<4;i++)for(int k=0;k<4;k++)w[k]+=s[i]*m.polynomial[1+i][k];
    int t=5;
    for(int i=0;i<4;i++)for(int j=i;j<4;j++,t++){
        double d=s[i]*s[j];for(int k=0;k<4;k++)w[k]+=d*m.polynomial[t][k];
    }
    return {{float(w[0]),float(w[1]),float(w[2]),float(w[3])}};
}
inline uint64_t key(int x,int y){return (uint64_t(uint32_t(x))<<32)|uint32_t(y);}
struct Entry {Signature s;int next;};
static int match(const Pair *a,int na,const Pair *b,int nb,const Mapping &first,
                 const Mapping &second,const quartet_fit::Shutter *ex,double gamma,double color,int cap,Seed *out,long long *stat){
    constexpr double width=4.;
    std::unordered_map<uint64_t,int> cells;cells.reserve(nb);
    std::vector<Entry> entries(nb);
    auto bin=[](double x){return int(std::floor(std::clamp(x,-1.e8,1.e8)/width));};
    for(int i=0;i<nb;i++){
        auto z=transform(b[i],second);
        auto pos=cells.emplace(key(bin(z.z[0]),bin(z.z[1])),-1).first;
        entries[i]={z,pos->second};pos->second=i;
    }
    std::vector<Seed> hits;
    auto better=[](const Seed &u,const Seed &v){return u.score>v.score;};
    for(int i=0;i<na;i++){
        auto z=transform(a[i],first);
        int x=bin(z.z[0]),y=bin(z.z[1]);
        for(int u=x-1;u<=x+1;u++)for(int v=y-1;v<=y+1;v++){
            auto it=cells.find(key(u,v));if(it==cells.end())continue;
            for(int j=it->second;j>=0;j=entries[j].next){
                double chi=0.;
                for(int k=0;k<4;k++)chi+=sq(z.z[k]-entries[j].s.z[k]);
                if(chi>16.)continue;
                double var=a[i].variance+b[j].variance+.025+sq(.28*color);
                double phot=sq(a[i].mag-b[j].mag)/var;
                if(phot>11.)continue;
                ++stat[0];
                quartet_fit::State orbit;double rms;
                quartet_fit::Shutter epochs[]={ex[a[i].e0],ex[a[i].e1],ex[b[j].e0],ex[b[j].e1]};
                if(!quartet_fit::solve(a[i].native,b[j].native,epochs,gamma,orbit,rms))continue;
                hits.push_back({{a[i].a,a[i].b,b[j].a,b[j].b},float(orbit.g),
                                float(a[i].quality+b[j].quality-.32*chi-.48*phot-1.6*sq(rms/.22))});
            }
        }
        if(int(hits.size())>4*cap){
            std::nth_element(hits.begin(),hits.begin()+cap,hits.end(),better);hits.resize(cap);
        }
    }
    if(int(hits.size())>cap){
        std::nth_element(hits.begin(),hits.begin()+cap,hits.end(),better);hits.resize(cap);
    }
    std::copy(hits.begin(),hits.end(),out);return int(hits.size());
}
}
extern "C" void* quartet_field(const quartet::Point *p,const int *edges,int ne){
    try{auto f=std::make_unique<quartet::Field>();f->im.resize(ne);
        for(int e=0;e<ne;e++)f->im[e].build(p+edges[e],p+edges[e+1]);
        return f.release();}
    catch(...){return nullptr;}
}
extern "C" void quartet_free(void *f){delete static_cast<quartet::Field*>(f);}
extern "C" int quartet_pairs(void *f,int a,int b,const double *gate,double color,int cap,quartet::Pair *out){
    try{auto &im=static_cast<quartet::Field*>(f)->im;std::vector<quartet::Pair> pairs;
        auto &left=im.at(a);auto &right=im.at(b);
        double area=31.2*sqrt(std::max(.0001,gate[8]*gate[10]-quartet::sq(gate[9])));
        double probability=std::min(1.,area/right.area);
        int u=0,v=0;
        while(probability*left.grid[u].pts.size()*right.grid[v].pts.size()>150000. && (u<3||v<3)){
            if(u<3&&(v==3||left.grid[u].pts.size()>right.grid[v].pts.size()))++u;
            else ++v;
        }
        quartet::generate(left.grid[u],right.grid[v],gate,color,cap,pairs);
        std::copy(pairs.begin(),pairs.end(),out);return int(pairs.size());}catch(...){return -1;}
}
extern "C" int quartet_matches(const quartet::Pair *a,int na,const quartet::Pair *b,int nb,
            const quartet::Mapping *models,const quartet_fit::Shutter *ex,double gamma,double color,int cap,quartet::Seed *out,long long *stat){
    try{return quartet::match(a,na,b,nb,models[0],models[1],ex,gamma,color,cap,out,stat);}
    catch(...){return -1;}
}
