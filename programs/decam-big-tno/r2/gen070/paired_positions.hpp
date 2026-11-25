// Enumerate measured alternatives at two withheld epochs. Their astrometric
// errors are correlated because both directions depend on the same orbit.
namespace paired_positions {
struct Query {
    int row[2];
    double offset[4],cov[16],m[2],phot[3];
};
struct Value {float same,one,any;int pairs;};
struct Peak {double dx,dy,dm,variance,mark,marginal;bool member;};
using cadence_ranging::Ellipse;
using encounter::V;

static std::vector<Peak> peaks(const original::Sky &sky,const Query &q,int end,Ellipse S){
    int row=q.row[end],e=sky.ex[row];const auto &im=sky.images[e];
    int dim=end*2;double x=sky.xy[2*row]+q.offset[dim]*as,y=sky.xy[2*row+1]+q.offset[dim+1]*as;
    double wx=4.6*std::sqrt(S.xx),wy=4.6*std::sqrt(S.yy);
    std::vector<Peak> out;
    auto inspect=[&](int j){
        double dx=(sky.xy[2*j]-x)/as,dy=(sky.xy[2*j+1]-y)/as;
        double chi=dot({dx,dy},S.solve({dx,dy}));
        double dm=sky.mag[j]-q.m[end],sn=std::max(4.5f,sky.sn[j]);
        double variance=sq(.095)+sq(1.086/sn);
        double mv=q.phot[2*end]+variance;
        if(!std::isfinite(dm+chi)||((chi>21.||sq(dm)>15.*mv||sky.flag[j]&1)&&j!=row))return;
        bool same=j==row||sq((sky.xy[2*j]-sky.xy[2*row])/as)+sq((sky.xy[2*j+1]-sky.xy[2*row+1])/as)<sq(.48);
        int b=std::clamp(int((sky.mag[j]-10.)*4.),0,79);
        double rho=im.rho[b],fraction=1.;
        if(sky.population){
            int tier=sky.population[j];
            if(tier){
                rho=im.transient[tier-1][b];fraction=tier==2?.78:.15;
            }else{
                rho=std::max(.00002,im.rho[b]-im.transient[0][b]-im.transient[1][b]);fraction=.07;
            }
        }
        double z=(sky.sharp[j]-1.)/std::hypot(.22,3.1/sn);
        double mark=std::log(fraction/std::max(.000002,rho))-.42*std::min(10.,z*z)-
                    .74*bool(sky.flag[j]&2)-.34*bool(sky.flag[j]&4);
        out.push_back({dx,dy,dm,variance,mark,mark-.5*(chi+sq(dm)/mv+std::log(mv)),same});
    };
    inspect(row);
    sky.nearby(im,x,y,wx*as,wy*as,[&](int j){if(j!=row)inspect(j);});
    // Keep the assigned peak and rank the remote alternatives by their
    // one-visit likelihood before the joint covariance test.
    if(out.size()>120){
        std::nth_element(out.begin()+1,out.begin()+120,out.end(),[](auto &a,auto &b){return a.marginal>b.marginal;});
        out.resize(120);
    }
    return out;
}
static double logadd(double a,double b){
    if(a<b)std::swap(a,b);
    return a+std::log1p(std::exp(b-a));
}
static Value compare(const original::Sky &sky,const Query &q){
    Ellipse A{q.cov[0],q.cov[1],q.cov[5]},B{q.cov[10],q.cov[11],q.cov[15]};
    if(!(A.det()>0.&&B.det()>0.))return {};
    V u=A.solve({q.cov[8],q.cov[9]}),v=A.solve({q.cov[12],q.cov[13]});
    Ellipse C{B.xx-dot(u,{q.cov[8],q.cov[9]}),B.xy-dot(u,{q.cov[12],q.cov[13]}),
              B.yy-dot(v,{q.cov[12],q.cov[13]})};
    if(!(C.det()>0.)||C.xx<=0.||C.yy<=0.)return {};
    auto first=peaks(sky,q,0,A),second=peaks(sky,q,1,B);
    if(first.empty()||second.empty())return {};
    double mass[]={-1.e40,-1.e40,-1.e40};int tests=0;
    for(auto &a:first){
        V here{a.dx,a.dy},centre{dot(u,here),dot(v,here)};
        double chi=dot(here,A.solve(here)),pm=q.phot[0]+a.variance;
        double phot=a.dm*a.dm/pm,colour=q.phot[1]/pm;
        for(auto &b:second){
            V d=V{b.dx,b.dy}-centre;double ast=chi+dot(d,C.solve(d));
            if(ast>42.&&!(a.member&&b.member))continue;
            double qm=q.phot[2]+b.variance-colour*q.phot[1];
            if(!(qm>0.))continue;
            double value=a.mark+b.mark-.5*(ast+phot+sq(b.dm-colour*a.dm)/qm+std::log(pm*qm));
            int changed=int(!a.member)+int(!b.member);
            mass[changed]=logadd(mass[changed],value);++tests;
        }
    }
    if(mass[0]<-1.e30)return {};
    return {float(mass[0]),float(logadd(mass[0],mass[1])),float(logadd(logadd(mass[0],mass[1]),mass[2])),tests};
}
}
extern "C" int paired_assignments(const original::Sky *sky,const paired_positions::Query *q,int n,
                                  double seconds,paired_positions::Value *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};
        auto work=[&]{
            while(Clock::now()<end){
                int at=next.fetch_add(8);if(at>=n)break;
                for(int i=at;i<std::min(at+8,n);i++)out[i]=paired_positions::compare(*sky,q[i]);
            }
        };
        std::thread thread(work);work();thread.join();return std::min(n,next.load());
    }catch(...){return -1;}
}
