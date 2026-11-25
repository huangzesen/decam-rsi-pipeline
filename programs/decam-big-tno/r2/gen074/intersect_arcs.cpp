// Join independently measured triples at one shared detection. The difference
// of their two heliocentric angular velocities also constrains inverse range.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>
#include "night_screen.cpp"

namespace intersection {
struct Triple { int row[3]; float gamma,score; int serial; };
struct Point { float x,y; int e,night; };
struct Trial { int row[5]; float gamma,score; };
struct Motion { double g,precision,slope[2],baseline[2],variance; };
using quartet_fit::Shutter;
constexpr double as=1./quartet_fit::arc;
constexpr double mu=quartet_fit::mu;

Motion measure(const Triple &s,const Point *points,const Shutter *shutters){
    double t[3],q[3][2],b[3][2],tm=0.,a[2]={},p[2]={};
    for(int i=0;i<3;i++){
        const Point &v=points[s.row[i]];const Shutter &e=shutters[v.e];
        t[i]=e.t;q[i][0]=v.x;q[i][1]=v.y;
        b[i][0]=e.ox-v.x*e.oz;b[i][1]=e.oy-v.y*e.oz;
        tm+=t[i]/3.;
        for(int k=0;k<2;k++){a[k]+=q[i][k]/3.;p[k]+=b[i][k]/3.;}
    }
    Motion m{};
    double tt=0.;
    for(int i=0;i<3;i++){t[i]-=tm;tt+=t[i]*t[i];}
    tt=std::max(1.,tt);
    for(int k=0;k<2;k++)for(int i=0;i<3;i++){
        q[i][k]-=a[k];b[i][k]-=p[k];
        m.slope[k]+=t[i]*q[i][k]/tt;m.baseline[k]+=t[i]*b[i][k]/tt;
    }
    double prior=s.gamma,vc=std::sqrt(mu*prior*prior*prior);
    double scatter=.37*as+.56*vc*vc*tt;
    double w=1./std::pow(.42*prior,2),rhs=prior*w;
    for(int k=0;k<2;k++)for(int i=0;i<3;i++){
        double u=q[i][k]-t[i]*m.slope[k],v=b[i][k]-t[i]*m.baseline[k];
        rhs-=v*u/(scatter*scatter);w+=v*v/(scatter*scatter);
    }
    m.g=std::max(.35*prior,std::min(2.4*prior,rhs/w));m.precision=w;
    m.variance=std::pow(.43*as/std::sqrt(tt)+.04*vc+.85*vc*vc*std::sqrt(tt),2);
    return m;
}

double joint(const Motion &a,const Motion &b,double &g){
    double w=a.precision+b.precision,rhs=a.g*a.precision+b.g*b.precision;
    const double variance=a.variance+b.variance;
    for(int k=0;k<2;k++){
        double v=a.slope[k]-b.slope[k],p=a.baseline[k]-b.baseline[k];
        w+=p*p/variance;rhs-=v*p/variance;
    }
    g=rhs/w;
    if(!(g>.000008 && g<.087))return 1.e25;
    double chi=std::pow(g-a.g,2)*a.precision+std::pow(g-b.g,2)*b.precision;
    for(int k=0;k<2;k++){
        double v=a.slope[k]-b.slope[k]+g*(a.baseline[k]-b.baseline[k]);
        chi+=v*v/variance;
    }
    return chi;
}
}

extern "C" int intersect_triples(const intersection::Triple *seeds,int count,const intersection::Point *points,
        const quartet_fit::Shutter *vis,int previous,double seconds,int cap,intersection::Trial *out,int64_t *stats){
    using namespace intersection;
    auto start=std::chrono::steady_clock::now();
    std::vector<Motion> motions;motions.reserve(count);
    std::vector<uint64_t> incidence;incidence.reserve(3*count);
    for(int i=0;i<count;i++){
        motions.push_back(measure(seeds[i],points,vis));
        for(int id:seeds[i].row)incidence.push_back((uint64_t(uint32_t(id))<<32)|uint32_t(i));
    }
    std::sort(incidence.begin(),incidence.end());
    auto cmp=[](const Trial &a,const Trial &b){return a.score>b.score;};
    std::priority_queue<Trial,std::vector<Trial>,decltype(cmp)> hits(cmp);
    int64_t pairs=0,fits=0,passed=0;
    for(size_t lo=0;lo<incidence.size();){
        int shared=int(incidence[lo]>>32);
        size_t hi=lo+1;
        while(hi<incidence.size() && int(incidence[hi]>>32)==shared)++hi;
        if(hi-lo>=2){
            if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>seconds)break;
            // Rank the exceptional very high multiplicities by measured arc
            // quality. Several independent surrounding observing nights survive.
            if(hi-lo>64){
                std::nth_element(incidence.begin()+lo,incidence.begin()+lo+64,incidence.begin()+hi,
                    [&](uint64_t a,uint64_t b){return seeds[uint32_t(a)].score>seeds[uint32_t(b)].score;});
            }
            size_t end=std::min(hi,lo+64);
            for(size_t l=lo;l<end;l++)for(size_t k=lo;k<l;k++){
                int a=uint32_t(incidence[k]),b=uint32_t(incidence[l]);
                if(seeds[a].serial<=previous && seeds[b].serial<=previous)continue;
                Trial trial{};int n=0;
                for(int row:seeds[a].row)trial.row[n++]=row;
                for(int row:seeds[b].row)if(row!=shared && n<5)trial.row[n++]=row;
                if(n!=5)continue;
                bool distinct=true;
                for(int i=1;i<5;i++)for(int j=0;j<i;j++)
                    if(points[trial.row[i]].night==points[trial.row[j]].night)distinct=false;
                if(!distinct)continue;
                ++pairs;double g;double chi=joint(motions[a],motions[b],g);
                if(chi>38.)continue;
                ++fits;
                night_screen::Sample s{};s.gamma=g;
                for(int i=0;i<5;i++){
                    const Point &p=points[trial.row[i]];
                    s.xy[i][0]=p.x;s.xy[i][1]=p.y;s.exposure[i]=p.e;
                }
                auto fit=night_screen::evaluate(s,vis);
                if(!(fit.gamma>0. && fit.rms<.46))continue;
                ++passed;
                trial.gamma=fit.gamma;
                trial.score=.56f*(seeds[a].score+seeds[b].score)-.13f*chi-2.6f*std::pow(fit.rms/.25f,2);
                if(int(hits.size())==cap){if(trial.score<=hits.top().score)continue;hits.pop();}
                hits.push(trial);
            }
        }
        lo=hi;
    }
    int n=hits.size();
    for(int i=n-1;i>=0;i--){out[i]=hits.top();hits.pop();}
    stats[0]=pairs;stats[1]=fits;stats[2]=passed;
    return n;
}
