// Three- and four-night paths in the graph of measured independent-night pairs.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
namespace night_chains {
constexpr double as=4.84813681109536e-6,mu=.0002963093;
struct Point {float x,y,m;int e;};
struct Date {double t,ox,oy,oz;int night,pad;};
struct Unit {int a,b;float score,g;};
struct Edge {int other;float g,score;};
struct Hypothesis {int row[4];float g,merit;};
struct V {double x,y;V operator+(V v)const{return {x+v.x,y+v.y};}
    V operator-(V v)const{return {x-v.x,y-v.y};}V operator*(double a)const{return {x*a,y*a};}};
double dot(V a,V b){return a.x*b.x+a.y*b.y;}
V pos(Point p){return {p.x,p.y};}
V obs(Point p,const Date &d){return {d.ox-p.x*d.oz,d.oy-p.y*d.oz};}
double sq(double x){return x*x;}
void trim(std::vector<Hypothesis> &h,int cap){
    std::sort(h.begin(),h.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    std::unordered_set<uint64_t> seen;int n=0;
    for(const auto &v:h){
        std::array<int,4> ids{};std::copy_n(v.row,4,ids.begin());std::sort(ids.begin(),ids.end());
        uint64_t k=1469598103934665603ull;for(int r:ids)k=(k^uint32_t(r))*1099511628211ull;
        if(!seen.insert(k).second)continue;h[n++]=v;if(n==cap)break;
    }
    h.resize(n);
}
struct Graph {
    const Point *p;const Date *date;const float *signal;
    std::vector<int> starts,pivots;std::vector<Edge> edges;
    Graph(const Point *rows,int n,const Date *times,const float *sn,const Unit *pairs,int np):
        p(rows),date(times),signal(sn),starts(n+1){
        for(int i=0;i<np;i++){++starts[pairs[i].a+1];++starts[pairs[i].b+1];}
        for(int i=0;i<n;i++){if(starts[i+1]>=2)pivots.push_back(i);starts[i+1]+=starts[i];}
        auto at=starts;edges.resize(2*np);
        for(int i=0;i<np;i++){
            auto &u=pairs[i];edges[at[u.a]++]={u.b,u.g,u.score};edges[at[u.b]++]={u.a,u.g,u.score};
        }
        std::mt19937 rng(519);std::shuffle(pivots.begin(),pivots.end(),rng);
    }
    double variance(int r)const{return sq(as)*(.0064+sq(1.07/std::max(4.5f,signal[r])));}
    void test(int pivot,Edge l,Edge r,std::vector<Hypothesis> &tri,std::vector<Hypothesis> &quad)const{
        std::array<int,3> ids{pivot,l.other,r.other};
        std::sort(ids.begin(),ids.end(),[&](int i,int j){return date[p[i].e].t<date[p[j].e].t;});
        auto [i,j,k]=ids;auto a=p[i],b=p[j],c=p[k];
        auto u=date[a.e],v=date[b.e],w=date[c.e];
        if(u.night==v.night||v.night==w.night)return;
        double span=w.t-u.t,mid=(v.t-u.t)/span;
        if(!(span>1.&&span<200.))return;
        double sg=.11*(l.g+r.g),g0=.5*(l.g+r.g);
        if(!(sg>0.)||std::abs(l.g-r.g)>2.6*sg)return;
        V pa=pos(a),pc=pos(c),oa=obs(a,u),oc=obs(c,w);
        V bend=(oc-oa)*mid-(obs(b,v)-oa),res=pos(b)-(pa+(pc-pa)*mid);
        double noise=variance(j)+sq(1.-mid)*variance(i)+sq(mid)*variance(k)+sq(as*(.05+.00001*span*span));
        double info=dot(bend,bend)/noise+1./sq(sg);
        double g=(dot(res,bend)/noise+g0/sq(sg))/info;
        V dev=res-bend*g;double chi=dot(dev,dev)/noise+sq((g-g0)/sg);
        if(chi>15.||g<1./9000.||g>1./13.)return;
        V rate=(pc-pa+(oc-oa)*g)*(1./span);
        if(dot(rate,rate)>3.1*mu*g*g*g+9.*(variance(i)+variance(k))/sq(span))return;
        double mag=(a.m+b.m+c.m)/3.,phot=(sq(a.m-mag)+sq(b.m-mag)+sq(c.m-mag))/sq(.31);
        if(phot>13.)return;
        double merit=.38*(l.score+r.score)-.5*chi-.38*phot+.47*std::log1p(span/5.)+
                     .55*std::log1p(sg*std::sqrt(info))-.26*std::log1p(starts[pivot+1]-starts[pivot]);
        Hypothesis third{{i,j,k,-1},float(g),float(merit)};
        tri.push_back(third);
        Hypothesis best=third;best.merit=-1.e20f;
        for(int id:ids)for(int z=starts[id];z<std::min(starts[id+1],starts[id]+14);z++){
            int rr=edges[z].other;auto f=p[rr];auto d=date[f.e];
            if(d.night==u.night||d.night==v.night||d.night==w.night||std::abs(f.m-mag)>.85)continue;
            double t=(d.t-u.t)/span;if(std::abs(t-.5)>5.5)continue;
            V lever=(oc-oa)*t-(obs(f,d)-oa),delta=pos(f)-(pa+(pc-pa)*t+lever*g);
            double var=variance(rr)+sq(1.-t)*variance(i)+sq(t)*variance(k)+sq(as*(.075+.00003*sq(d.t-v.t)));
            double leverage=dot(lever,lever),den=var+leverage/info;
            double dg=dot(delta,lever)/(var*info+leverage);
            V perpendicular=delta-lever*dg;
            double dist=dot(perpendicular,perpendicular)/var+dg*dg*info;
            if(dist>13.||std::hypot(delta.x,delta.y)>24.*as)continue;
            double quality=merit+.28*edges[z].score-.5*dist-.44*sq((f.m-mag)/.3)-
                           .44*std::log(.05+std::sqrt(var*den)/sq(as));
            if(quality>best.merit)best={{i,j,k,rr},float(g+dg),float(quality)};
        }
        if(best.row[3]>=0)quad.push_back(best);
    }
};
}
extern "C" int shared_night_chains(const night_chains::Point *target,int nt,const night_chains::Date *dates,
        const float *signal,const night_chains::Unit *bank,int n,int threads,double seconds,int cap,
        night_chains::Hypothesis *out,long long *stats){
    using namespace night_chains;using Clock=std::chrono::steady_clock;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        Graph sky(target,nt,dates,signal,bank,n);std::atomic<int> next{0};
        std::vector<std::vector<Hypothesis>> threes(threads),fours(threads);
        std::vector<std::thread> workers;
        for(int z=0;z<threads;z++)workers.emplace_back([&,z]{
            auto &tri=threes[z],&quad=fours[z];
            while(Clock::now()<end){
                int lo=next.fetch_add(32);if(lo>=int(sky.pivots.size()))break;
                for(int p=lo;p<std::min(lo+32,int(sky.pivots.size()));p++){
                    int pivot=sky.pivots[p],start=sky.starts[pivot],stop=std::min(sky.starts[pivot+1],start+18);
                    for(int i=start;i<stop;i++)for(int j=start;j<i;j++)sky.test(pivot,sky.edges[i],sky.edges[j],tri,quad);
                }
                if(int(tri.size())>3*cap)trim(tri,cap);
                if(int(quad.size())>3*cap)trim(quad,cap);
            }
        });
        for(auto &w:workers)w.join();
        std::vector<Hypothesis> tri,quad;
        for(int z=0;z<threads;z++){tri.insert(tri.end(),threes[z].begin(),threes[z].end());quad.insert(quad.end(),fours[z].begin(),fours[z].end());}
        trim(quad,cap/2);trim(tri,cap-quad.size());
        stats[0]=std::min<int>(next,sky.pivots.size());stats[1]=sky.pivots.size();stats[2]=quad.size();stats[3]=tri.size();
        std::copy(quad.begin(),quad.end(),out);std::copy(tri.begin(),tri.end(),out+quad.size());return quad.size()+tri.size();
    }catch(...){return -1;}
}
