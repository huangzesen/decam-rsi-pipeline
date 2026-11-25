// Predict each independent night from the others before scheduling expensive
// image validation. All trial trajectories compete against the same sky.
namespace recovery_order {
struct Night {
    int row,day;
    double jac[2][6],res[2],variance,mag,lever,mvar,background,shape;
};

double predictive(const original::Sky &sky,const Certified &fit){
    if(fit.count<5||fit.count>64)return 0.;
    std::array<Night,64> rows{};
    int count=0;
    for(int k=0;k<fit.count;++k){
        int row=fit.row[k];
        if(row<0||row>=sky.size||!std::isfinite(sky.mag[row]))return 0.;
        int day=sky.images[sky.ex[row]].exposure.night;
        int i=0;while(i<count&&rows[i].day!=day)++i;
        if(i==count){++count;rows[i].day=day;rows[i].row=row;}
        else if(sky.sn[row]>sky.sn[rows[i].row])rows[i].row=row;
    }
    if(count<5)return 0.;
    const double *a=fit.x;
    ExactState state{a[0],a[1],a[2],a[3],a[4],a[5],fit.ref};
    if(!(state.g>1./20000.&&state.g<1./11.))return 0.;
    Kepler orbit(state);original::OrbitDerivatives derivative(orbit);
    double N[6][6]{},rhs[6]{},gradient[6]{},inverse[6][6]{};
    N[0][0]=N[1][1]=1.e-12;N[2][2]=5.;N[3][3]=N[4][4]=2.5;N[5][5]=7.;
    double sw=0.,sl=0.,sll=1./sq(.29),sy=0.,sly=0.;
    for(int i=0;i<count;++i){
        auto &n=rows[i];int r=n.row;
        auto p=sky.sample(r);double x,y,f,s,anom[3];
        orbit.apparent(p,x,y,f,s,anom);derivative.design(p,x,y,f,s,anom,n.jac);
        n.res[0]=(p.x-x)/as;n.res[1]=(p.y-y)/as;
        n.variance=sq(.105)+sq(1.06/sky.sn[r]);
        for(int k=0;k<2;++k)for(int u=0;u<6;++u){
            rhs[u]+=n.jac[k][u]*n.res[k]/n.variance;
            for(int v=0;v<6;++v)N[u][v]+=n.jac[k][u]*n.jac[k][v]/n.variance;
        }
        auto &im=sky.images[sky.ex[r]];
        n.mag=sky.mag[r]-sky.mag[rows[0].row];n.lever=sky.lever[sky.ex[r]];
        n.mvar=sq(.155)+sq(1.086/sky.sn[r]);double w=1./n.mvar;
        sw+=w;sl+=w*n.lever;sll+=w*sq(n.lever);sy+=w*n.mag;sly+=w*n.mag*n.lever;
        int mb=std::clamp(int((sky.mag[r]-10.)*4.),0,79);
        int tier=sky.population?sky.population[r]:0;
        double rho=im.rho[mb],fraction=1.;
        if(sky.population){
            if(tier){rho=im.transient[tier-1][mb];fraction=tier==2?.69:.20;}
            else{rho=std::max(rho*.04,rho-im.transient[0][mb]-im.transient[1][mb]);fraction=.11;}
        }
        n.background=std::log(std::max(2.e-8,rho)/fraction);
        n.shape=-.32*std::min(10.,sq((sky.sharp[r]-1.04)/std::hypot(.20,3.5/sky.sn[r])))
                -.55*bool(sky.flag[r]&2)-.23*bool(sky.flag[r]&4);
    }
    if(!original::invert(N,inverse))return 0.;
    for(int i=0;i<6;++i)for(int j=0;j<6;++j)gradient[i]+=inverse[i][j]*rhs[j];
    double scores[64];
    for(int i=0;i<count;++i){
        auto &n=rows[i];double innovation[2]={n.res[0],n.res[1]},P[2][2]{};
        for(int k=0;k<2;++k)for(int j=0;j<6;++j){
            innovation[k]-=n.jac[k][j]*gradient[j];
            double v=0.;for(int u=0;u<6;++u)v+=n.jac[k][u]*inverse[u][j];
            for(int q=0;q<2;++q)P[k][q]+=v*n.jac[q][j]/n.variance;
        }
        // Determinant lemma for removal of both coordinates in this night.
        double xx=std::max(1.e-10,1.-P[0][0]),yy=std::max(1.e-10,1.-P[1][1]),xy=-.5*(P[0][1]+P[1][0]);
        double det=std::clamp(xx*yy-xy*xy,1.e-10,1.);
        double chi=(yy*sq(innovation[0])-2.*xy*innovation[0]*innovation[1]+xx*sq(innovation[1]))/(det*n.variance);
        double w=1./n.mvar,l=n.lever,aa=sw-w,ab=sl-w*l,bb=sll-w*l*l;
        double determinant=aa*bb-ab*ab,b1=sy-w*n.mag,b2=sly-w*l*n.mag;
        double mu=((bb-ab*l)*b1+(aa*l-ab)*b2)/determinant;
        double vm=n.mvar+(bb+aa*l*l-2.*ab*l)/determinant;
        double phot=-.5*(std::min(18.,sq(n.mag-mu)/vm)+std::log(2.*M_PI*vm));
        double odds=phot+.5*std::log(det)-.5*std::min(30.,std::max(0.,chi))-
                    std::log(2.*M_PI*n.variance)-n.background+n.shape;
        scores[i]=std::isfinite(odds)?std::clamp(odds,-16.,16.):-10.;
    }
    std::sort(scores,scores+count);
    // The weakest return competes against a shorter real arc plus noise.
    // The remaining visits supply evidence when two separate nights are weak.
    return .44*scores[0]+.25*scores[1]+.31*scores[count/2]+.9*std::log1p(count-5);
}
}
extern "C" void recovery_support(const original::Sky *sky,const original::Recovery *v,int n,
                                 int threads,double *out){
    std::atomic<int> next{0};std::vector<std::thread> workers;
    for(int j=0;j<threads;++j)workers.emplace_back([&]{
        for(int i;(i=next.fetch_add(1))<n;)out[i]=recovery_order::predictive(*sky,v[i].orbit);
    });
    for(auto &w:workers)w.join();
}
