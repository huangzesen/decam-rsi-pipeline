// The errors in forecasts at two missing nights share the uncertain orbital
// elements. Two peaks which are uninformative on their own can fix the orbit.
namespace joint_returns {
using original::Sky;
using original::PairedReturns;
using original::Core;
using original::Recovery;

struct Ellipse {
    double xx,xy,yy;
    double det()const{return xx*yy-xy*xy;}
    double chi(double x,double y)const{return (yy*x*x-2.*xy*x*y+xx*y*y)/det();}
};
struct Forecast {int e,night;double x,y,J[2][6],JV[2][6];Ellipse cov;};
struct Peak {int row,view;double x,y,term,merit,noise;};
struct Choice {int a,b;double merit;};

bool bound_fit(const std::vector<Sample> &points,ExactState &state,std::vector<double> &residual){
    const ExactState initial=state;
    if(measured_fit(points,state,residual,8))return true;
    if(!(initial.g>1./9000.&&initial.g<1./17.))return false;
    // A weakly measured radial component can carry the unconstrained
    // optimum above escape speed. Profile the measured seed's two range
    // coordinates separately before rejecting the physical family.
    double best=sq(.34)*points.size();bool accepted=false;
    ExactState solution=initial;std::vector<double> errors;
    for(int fixed=1;fixed<=3;fixed++){
        ExactState fit=initial;
        if(!measured_fit(points,fit,residual,7,nullptr,fixed&1,fixed&2))continue;
        double rss=0.;for(double x:residual)rss+=x*x;
        if(rss>=best)continue;
        best=rss;solution=fit;errors=residual;accepted=true;
    }
    if(accepted){state=solution;residual.swap(errors);}
    return accepted;
}
struct Images {
    const Sky &sky;
    const PairedReturns *rare;
    const double *depth;
    std::vector<Forecast> views;
    std::vector<Peak> peaks,local;
    std::vector<Choice> choices;
    std::vector<Sample> core;
    double cm=0.,color=0.,lever=0.,colorsigma=.28,brightness_var=.02;
    int nights[12],nn=0;

    bool prior(ExactState &o,double V[6][6]){
        Kepler orbit(o);original::OrbitDerivatives solve(orbit);
        double N[6][6]{},m[12]{},sw=0.,rhs=0.,slope=1./sq(.29);
        N[0][0]=N[1][1]=1.e-12;N[2][2]=6.;N[3][3]=N[4][4]=3.;N[5][5]=6.;
        for(size_t k=0;k<core.size();k++)m[k]=sky.mag[core[k].row];
        std::sort(m,m+core.size());double median=.5*(m[core.size()/2]+m[(core.size()-1)/2]);
        cm=lever=0.;
        for(const Sample &v:core){
            int row=v.row,e=sky.ex[row];double x,y,f,s,anom[3],J[2][6];
            orbit.apparent(v,x,y,f,s,anom);solve.design(v,x,y,f,s,anom,J);
            double error=std::hypot(x-v.x,y-v.y)/as;
            if(!(error<1.05))return false;
            double wt=1./(.012+sq(1.07/sky.sn[row])+.16*sq(error));
            for(int j=0;j<6;j++)for(int l=0;l<6;l++)
                N[j][l]+=wt*(J[0][j]*J[0][l]+J[1][j]*J[1][l]);
            double w=1./(.030+sq(1.09/sky.sn[row])+sq(.50*(sky.mag[row]-median)));
            cm+=w*sky.mag[row];lever+=w*sky.lever[e];sw+=w;
        }
        cm/=sw;lever/=sw;
        for(const Sample &v:core){
            double w=1./(.030+sq(1.09/sky.sn[v.row])+sq(.50*(sky.mag[v.row]-median)));
            double l=sky.lever[sky.ex[v.row]]-lever;
            slope+=w*l*l;rhs+=w*l*(sky.mag[v.row]-cm);
        }
        color=std::clamp(rhs/slope,-.56,.62);colorsigma=std::sqrt(1./slope);
        brightness_var=.004+1./sw;
        return original::invert(N,V);
    }
    void forecast(const ExactState &o,const double V[6][6]){
        Kepler orbit(o);original::OrbitDerivatives deriv(orbit);
        double mug=mu*std::pow(o.g/std::sqrt(1.+sq(o.x)+sq(o.y)),3);
        for(int e=0;e<int(sky.images.size());e++){
            const auto &im=sky.images[e];const auto &z=im.exposure;
            if(!im.nx||std::binary_search(nights,nights+nn,z.night))continue;
            double m=cm+(sky.lever[e]-lever)*color;
            if(!std::isfinite(m)||(depth&&m>depth[e]+1.1))continue;
            int mb=std::clamp(int((m-10.)*4.),0,79);
            double t=z.t-o.ref,ph=mug*t*t,f=1.-.5*ph*(1.-t*o.h),s=t*(1.-ph/6.);
            double den=f+s*o.h-o.g*z.oz;
            double x=(f*o.x+s*o.vx-o.g*z.ox)/den,y=(f*o.y+s*o.vy-o.g*z.oy)/den;
            double pad=.014+.13*ph*ph;
            if(x<im.xmin-pad||x>im.xmax+pad||y<im.ymin-pad||y>im.ymax+pad)continue;
            Forecast view{};view.e=e;view.night=z.night;
            Sample epoch{-1,0.,0.,z.t,z.ox,z.oy,z.oz};double anom[3];
            orbit.apparent(epoch,x,y,f,s,anom);view.x=x;view.y=y;
            deriv.design(epoch,x,y,f,s,anom,view.J);
            for(int a=0;a<2;a++)for(int j=0;j<6;j++)for(int k=0;k<6;k++)
                view.JV[a][j]+=view.J[a][k]*V[k][j];
            auto &C=view.cov;
            for(int j=0;j<6;j++){
                C.xx+=view.JV[0][j]*view.J[0][j];
                C.xy+=view.JV[0][j]*view.J[1][j];
                C.yy+=view.JV[1][j]*view.J[1][j];
            }
            Ellipse gate{C.xx+.065,C.xy,C.yy+.065};double det=gate.det();
            if(!(det>0.&&gate.xx>0.&&gate.yy>0.)||gate.xx+gate.yy>sq(3200.))continue;
            double area=std::sqrt(det),wx=3.15*as*std::sqrt(gate.xx),wy=3.15*as*std::sqrt(gate.yy);
            if(x+wx<im.xmin||x-wx>im.xmax||y+wy<im.ymin||y-wy>im.ymax)continue;
            bool sparse=rare&&area*im.rho[mb]>2.8;
            double rho=sparse?rare->images[e].rare_density[mb]:im.rho[mb];
            if(area*rho>24.||area>300000.)continue;
            const int index=views.size();
            local.clear();
            auto consider=[&](int row){
                if((sky.flag[row]&1)||!std::isfinite(sky.mag[row]))return;
                double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
                double variance=sq(.108)+sq(1.10/sky.sn[row]);
                Ellipse S{C.xx+variance,C.xy,C.yy+variance};
                double chi=S.chi(dx,dy);if(!(chi<11.5))return;
                double dm=sky.mag[row]-m,vm=.025+brightness_var+sq(1.086/sky.sn[row])+sq(colorsigma*(sky.lever[e]-lever));
                if(dm*dm>12.*vm)return;
                int tier=sky.population?sky.population[row]:0;
                double null=rho,eff=sparse?.26:1.;
                if(!sparse && sky.population){
                    if(tier){null=im.transient[tier-1][mb];eff=tier==2?.69:.20;}
                    else{null=std::max(rho*.04,rho-im.transient[0][mb]-im.transient[1][mb]);eff=.11;}
                }
                double sh=(sky.sharp[row]-1.04)/std::hypot(.20,3.5/sky.sn[row]);
                double term=std::log(eff/(std::max(1.e-7,null)*2.*M_PI*std::sqrt(2.*M_PI*vm)))-
                            .5*sq(dm)/vm-.28*std::min(10.,sh*sh)-.65*bool(sky.flag[row]&2)-.26*bool(sky.flag[row]&4);
                double odds=term-.5*(chi+std::log(S.det()));
                if(odds> -7.0)local.push_back({row,index,dx,dy,term,odds,variance});
            };
            if(sparse){
                const auto &rows=rare->images[e].rare;
                auto lo=std::lower_bound(rows.begin(),rows.end(),y-wy,[&](int row,double cut){return sky.xy[row*2+1]<cut;});
                for(auto it=lo;it!=rows.end()&&sky.xy[*it*2+1]<=y+wy;++it)
                    if(std::abs(sky.xy[*it*2]-x)<=wx)consider(*it);
            }else sky.ellipse(im,x,y,gate.xx,gate.xy,gate.yy,consider);
            if(local.empty())continue;
            views.push_back(view);
            if(local.size()>12){
                std::nth_element(local.begin(),local.begin()+12,local.end(),[](auto &a,auto &b){return a.merit>b.merit;});
                local.resize(12);
            }
            peaks.insert(peaks.end(),local.begin(),local.end());
        }
    }
    double pair_odds(const Peak &a,const Peak &b)const{
        const auto &one=views[a.view],&two=views[b.view];
        const auto &s=one.cov;
        Ellipse c{s.xx+a.noise,s.xy,s.yy+a.noise};double det=c.det();
        double u[2][2]{},k[2][2];
        for(int i=0;i<2;i++)for(int j=0;j<2;j++){
            for(int l=0;l<6;l++)u[i][j]+=two.JV[i][l]*one.J[j][l];
        }
        for(int i=0;i<2;i++){
            k[i][0]=(u[i][0]*c.yy-u[i][1]*c.xy)/det;
            k[i][1]=(u[i][1]*c.xx-u[i][0]*c.xy)/det;
        }
        double dx=b.x-k[0][0]*a.x-k[0][1]*a.y,dy=b.y-k[1][0]*a.x-k[1][1]*a.y;
        const auto &t=two.cov;
        Ellipse cond{t.xx+b.noise-k[0][0]*u[0][0]-k[0][1]*u[0][1],
                     t.xy-k[0][0]*u[1][0]-k[0][1]*u[1][1],
                     t.yy+b.noise-k[1][0]*u[1][0]-k[1][1]*u[1][1]};
        if(!(cond.det()>0.&&cond.xx>0.&&cond.yy>0.))return -100.;
        double chi=cond.chi(dx,dy);
        if(chi>15.)return -100.;
        return a.merit+b.term-.5*(chi+std::log(cond.det()));
    }
    int extend(const Core &input,Recovery *out){
        views.clear();peaks.clear();core.clear();choices.clear();
        if(input.count<3||input.count>12)return 0;
        for(int i=0;i<input.count;i++){
            int row=input.rows[i];if(row<0||row>=sky.size||!std::isfinite(sky.mag[row]))return 0;
            core.push_back(sky.sample(row));nights[i]=sky.images[sky.ex[row]].exposure.night;
        }
        std::sort(nights,nights+input.count);nn=std::unique(nights,nights+input.count)-nights;
        if(nn<3||nn>4)return 0;
        ExactState state{input.x[0],input.x[1],input.x[2],input.x[3],input.x[4],input.x[5],input.ref};
        std::vector<double> residual;
        if(!bound_fit(core,state,residual))return 0;
        double V[6][6]{};
        if(!prior(state,V))return 0;
        forecast(state,V);
        if(peaks.empty())return 0;
        std::sort(peaks.begin(),peaks.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        std::vector<Peak> unique;std::unordered_map<int,int> night_count;
        for(const auto &p:peaks){
            if(night_count[views[p.view].night]++>=11)continue;
            unique.push_back(p);
            if(unique.size()>=96)break;
        }
        peaks.swap(unique);
        for(int i=0;i<int(peaks.size());i++){
            const auto &a=peaks[i];
            if(nn==4 && a.merit>.20)choices.push_back({i,-1,a.merit});
            for(int j=0;j<i;j++){
                const auto &b=peaks[j];if(views[a.view].night==views[b.view].night)continue;
                double common=pair_odds(b,a);
                // Compete against a genuine single return plus a field source.
                if(common>1.6 && common>std::max(a.merit,b.merit)-.20)choices.push_back({i,j,common});
            }
        }
        int trials=std::min(10,int(choices.size()));
        std::partial_sort(choices.begin(),choices.begin()+trials,choices.end(),[](auto &a,auto &b){return a.merit>b.merit;});
        std::vector<Recovery> records;
        for(int k=0;k<trials;k++){
            const auto &c=choices[k];auto observations=core;
            observations.push_back(sky.sample(peaks[c.a].row));
            if(c.b>=0)observations.push_back(sky.sample(peaks[c.b].row));
            ExactState fit=state;
            if(!bound_fit(observations,fit,residual))continue;
            double rss=0.;for(double r:residual)rss+=r*r;
            if(*std::max_element(residual.begin(),residual.end())>.74||rss>sq(.31)*observations.size())continue;
            Recovery record{};auto &r=record.orbit;
            r.count=observations.size();r.nights=nn+1+(c.b>=0);record.extra=r.count-core.size();
            for(int l=0;l<r.count;l++){r.row[l]=observations[l].row;r.err[l]=residual[l];}
            record.logbf=c.merit-.50*rss/(r.count*sq(.22))-.65*std::log1p(double(choices.size())/4.);
            r.merit=input.merit+record.logbf+2.*(r.nights-4);
            double par[]={fit.x,fit.y,fit.g,fit.vx,fit.vy,fit.h};std::copy_n(par,6,r.x);r.ref=fit.ref;
            if(rare)r.merit+=.60*rare->priority(r);
            records.push_back(record);
        }
        if(records.empty())return 0;
        std::sort(records.begin(),records.end(),[](auto &a,auto &b){return a.orbit.merit>b.orbit.merit;});
        out[0]=records[0];int count=1;
        for(size_t k=1;k<records.size();k++){
            const auto &r=records[k].orbit,&best=out[0].orbit;
            if(r.merit+3.5<best.merit)break;
            if(std::find(best.row+core.size(),best.row+best.count,r.row[r.count-1])==best.row+best.count){
                out[count++]=records[k];break;
            }
        }
        return count;
    }
};
}

extern "C" int joint_return_fits(const original::Sky *sky,const original::PairedReturns *rare,
        const double *depth,const original::Core *seeds,int n,int threads,double seconds,int cap,
        original::Recovery *out,int *tested){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> workers;
        std::vector<std::vector<original::Recovery>> banks(threads);
        for(int j=0;j<threads;j++)workers.emplace_back([&,j]{
            joint_returns::Images field{*sky,rare,depth};
            while(!expired(end)){
                int i=next.fetch_add(1);if(i>=n)break;
                original::Recovery hit[2];int count=field.extend(seeds[i],hit);
                banks[j].insert(banks[j].end(),hit,hit+count);
                if(int(banks[j].size())>2*cap/threads+32)visit_triplets::distinct_recoveries(banks[j],cap/threads+1);
            }
        });
        for(auto &w:workers)w.join();
        std::vector<original::Recovery> all;
        for(auto &bank:banks)all.insert(all.end(),bank.begin(),bank.end());
        visit_triplets::distinct_recoveries(all,cap);std::copy(all.begin(),all.end(),out);
        *tested=std::min(n,next.load());return all.size();
    }catch(...){return -1;}
}
