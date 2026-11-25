// Posterior sky ellipse for a position, two angular rates and one parallax.
// Eliminating the common linear trend leaves one scalar distance uncertainty.
struct ParallaxForecast {
    const Sky &sky;
    V intercept{},slope{};
    double epoch,weight=0.,tt=0.,par_info=0.;
    ParallaxForecast(const Sky &s,const int *rows,int n,const State &state):sky(s),epoch(state.epoch){
        for(int j=0;j<n;++j){
            int r=rows[j];double w=1./variance(s,r);weight+=w;
            intercept=intercept+observer(s.images[s.ex[r]].exposure,position(s,r))*w;
        }
        intercept=intercept*(1./weight);
        for(int j=0;j<n;++j){
            int r=rows[j];const auto &e=s.images[s.ex[r]].exposure;
            double w=1./variance(s,r),dt=e.t-epoch;tt+=w*dt*dt;
            V q=observer(e,position(s,r))-intercept;
            slope=slope+q*(w*dt);par_info+=dot(q,q)*w;
        }
        if(tt>0.){par_info-=dot(slope,slope)/tt;slope=slope*(1./tt);}
    }
    int match(int ex,const State &state,double mag,double &merit)const{
        const auto &im=sky.images[ex];const auto &e=im.exposure;
        if(!(tt>0.&&par_info>1.e-6))return -1;
        V predicted=apparent(state,e);double dt=e.t-epoch;
        V b=observer(e,predicted)-intercept-slope*dt;
        double intercept_var=1./weight+dt*dt/tt+.016;
        double xx=intercept_var+b.x*b.x/par_info,xy=b.x*b.y/par_info,yy=intercept_var+b.y*b.y/par_info;
        double gate0=xx+.060,gate1=yy+.060;
        double area=std::sqrt(std::max(1.e-8,gate0*gate1-xy*xy));
        int bin=std::clamp(int((mag-10.)*4.),0,79);
        if(area*im.rho[bin]>1.35||gate0+gate1>sq(90.))return -1;
        if(!covered(im,predicted,3.2*as*std::sqrt(std::max(gate0,gate1))))return -1;
        int best=-1;merit=-1.e10;
        sky.ellipse(im,predicted.x,predicted.y,gate0,xy,gate1,[&](int row){
            if(!usable(sky,row,mag,1.05))return;
            V delta=(position(sky,row)-predicted)*(1./as);
            double v=variance(sky,row),a=xx+v,c=yy+v,det=a*c-xy*xy;
            double chi=(c*sq(delta.x)-2.*xy*delta.x*delta.y+a*sq(delta.y))/det;
            if(chi>11.8)return;
            double vm=.06+sq(1.086/sky.sn[row]),dm=sky.mag[row]-mag;
            double odds=-std::log(2.*M_PI*std::max(.00001,im.rho[bin]))-
                       .5*(chi+std::log(det)+sq(dm)/vm+std::log(2.*M_PI*vm));
            if(odds>merit){merit=odds;best=row;}
        });
        return best;
    }
};
