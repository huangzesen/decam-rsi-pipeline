// A catalogue peak has both a flux and a position. Subtract the light
// and first spatial derivative of a steady reference PSF from that peak.
// This header is included inside the original-catalogue namespace.
struct ReferenceLight {
    double x=0., y=0., flux=0., pos_sigma=0., flux_sigma=0.;
    int nights=0,stratum=-1;
};
struct LightSample {double x,y,mag,t,sn,seeing,band;int day;};
inline double middle(std::vector<double> &v){
    size_t k=v.size()/2;
    std::nth_element(v.begin(),v.begin()+k,v.end());
    return v.size()%2?v[k]:.5*(v[k]+*std::max_element(v.begin(),v.begin()+k));
}
inline int sky_side(const Sky &sky,int row){
    int x=int(std::floor(sky.xy[2*row]/(.057*3600.*as)));
    int y=int(std::floor(sky.xy[2*row+1]/(.057*3600.*as)));
    return (unsigned(x)*2654435761U+unsigned(y)*2246822519U)&1U;
}
static ReferenceLight reference_light(const Sky &sky,int row,bool withhold=false){
    ReferenceLight out;
    int home=sky.ex[row];const auto &frame=sky.images[home];
    double x=sky.xy[2*row],y=sky.xy[2*row+1],when=frame.exposure.t;
    double radius=std::clamp(1.65*sky.seeing[home],1.40,2.35);
    std::vector<LightSample> sample;
    for(int j=0;j<int(sky.images.size());j++){
        const auto &im=sky.images[j];double dt=im.exposure.t-when;
        if(std::abs(dt)<6. || !im.nx || (withhold && sky.lever[j]==sky.lever[home]) ||
           x<im.xmin-radius*as || x>im.xmax+radius*as ||
           y<im.ymin-radius*as || y>im.ymax+radius*as)continue;
        int best=-1;double merit=1.e9;
        sky.nearby(im,x,y,radius*as,radius*as,[&](int i){
            if(sky.flag[i]&1 || !std::isfinite(sky.mag[i]))return;
            double d=sq((sky.xy[2*i]-x)/as)+sq((sky.xy[2*i+1]-y)/as);
            if(d>radius*radius)return;
            double q=d+.18*sq(std::max(0.,5.5-double(sky.sn[i])));
            if(q<merit){merit=q;best=i;}
        });
        if(best>=0)sample.push_back({(sky.xy[2*best]-x)/as,(sky.xy[2*best+1]-y)/as,
                   sky.mag[best],dt/365.25,sky.sn[best],sky.seeing[j],sky.lever[j],im.exposure.night});
    }
    if(sample.size()<2)return out;
    double home_band=sky.lever[home],transfer_error=0.,transfer_shift=0.;
    std::vector<int> direct;
    for(const auto &z:sample)if(z.band==home_band)direct.push_back(z.day);
    std::sort(direct.begin(),direct.end());
    if(std::unique(direct.begin(),direct.end())-direct.begin()>=2){
        sample.erase(std::remove_if(sample.begin(),sample.end(),[&](const auto &z){return z.band!=home_band;}),
                     sample.end());
    }else{
        // The catalogues are already on the same nominal colour system.
        // Independent passbands constrain the residual stellar slope. Use
        // one flux per band and date so a dither sequence cannot dominate.
        if(sample.size()<4)return out;
        std::sort(sample.begin(),sample.end(),[](const auto &a,const auto &b){
            if(a.band!=b.band)return a.band<b.band;
            if(a.day!=b.day)return a.day<b.day;
            return a.sn>b.sn;
        });
        sample.erase(std::unique(sample.begin(),sample.end(),[](const auto&a,const auto&b){
                                    return a.band==b.band&&a.day==b.day;}),sample.end());
        struct Bin {double l,mag,var;};std::vector<Bin> bins;
        for(size_t j=0;j<sample.size();){
            size_t k=j;std::vector<double> mags,errors;
            do{mags.push_back(sample[k].mag);errors.push_back(1.086/sample[k].sn);k++;}
            while(k<sample.size()&&sample[k].band==sample[j].band);
            double m=middle(mags),err=middle(errors);
            for(double &z:mags)z=std::abs(z-m);
            double scatter=1.48*middle(mags);
            bins.push_back({sample[j].band-home_band,m,.095*.095+(err*err+scatter*scatter)/std::min(9.,double(k-j))});
            j=k;
        }
        if(bins.size()<2 || sample.size()<4 || bins.back().l-bins.front().l<.25)return out;
        double center=bins.front().mag,slope=0.,a,b,c,det;
        for(int it=0;it<4;it++){
            a=0.;b=0.;c=1./sq(.78);double ax=0.,bx=0.;
            for(const auto &z:bins){
                double res=z.mag-center-slope*z.l,w=1./(z.var+(it?.18*res*res:0.));
                a+=w;b+=w*z.l;c+=w*z.l*z.l;ax+=w*z.mag;bx+=w*z.l*z.mag;
            }
            det=a*c-b*b;if(!(det>0.))return out;
            center=(c*ax-b*bx)/det;slope=(a*bx-b*ax)/det;
        }
        double scatter=0.;
        for(const auto &z:bins)scatter+=sq(z.mag-center-slope*z.l);
        double extrap=std::max({0.,bins.front().l,-bins.back().l});
        transfer_error=std::sqrt(.105*.105+c/det+.40*scatter/bins.size()+sq(.23*extrap));
        if(transfer_error>.31||std::abs(slope)>1.85)return out;
        std::array<double,6> levers={1.,0.,-.5,-.9,-1.,.2};
        int band_index=int(std::min_element(levers.begin(),levers.end(),[&](double x,double y){
                    return std::abs(x-home_band)<std::abs(y-home_band);})-levers.begin());
        int geometry=bins.front().l>0.?0:bins.back().l<0.?2:1;
        int colour=std::clamp(int(std::floor((slope+.75)/.35)),0,4);
        out.stratum=5*(3*band_index+geometry)+colour;
        if(!withhold){
            const auto &model=sky.transfer[sky_side(sky,row)][out.stratum];
            transfer_shift=model.shift;
            transfer_error=std::hypot(transfer_error,model.scatter);
        }
        for(auto &z:sample)z.mag-=slope*(z.band-home_band);
    }
    // One independent measurement per night, favouring reference seeing
    // comparable to the detection and a well-measured centroid.
    auto quality=[&](const auto &a){return .7*std::abs(std::log(a.seeing/sky.seeing[home]))+
                                             2.5/std::max(4.5,a.sn);};
    std::sort(sample.begin(),sample.end(),[&](const auto&a,const auto&b){
        return a.day==b.day?quality(a)<quality(b):a.day<b.day;
    });
    sample.erase(std::unique(sample.begin(),sample.end(),
                       [](const auto&a,const auto&b){return a.day==b.day;}),sample.end());
    if(sample.size()<(transfer_error>0.?4U:2U))return out;
    auto times=std::minmax_element(sample.begin(),sample.end(),[](const auto &a,const auto &b){return a.t<b.t;});
    if(times.second->t-times.first->t<3./365.25)return out;
    std::vector<double> xx,yy,mm,tmp;
    for(const auto &z:sample){xx.push_back(z.x);yy.push_back(z.y);mm.push_back(z.mag);}
    double cx=middle(xx),cy=middle(yy),cm=middle(mm),vx=0.,vy=0.;
    double a=0.,b=0.,c=0.,det=0.,ww=0.;
    for(int it=0;it<4;it++){
        a=0.;b=0.;c=1./sq(.12);double dx=0.,dy=0.,tx=0.,ty=0.,fm=0.; ww=0.;
        for(const auto &z:sample){
            double pos=sq(z.x-cx-vx*z.t)+sq(z.y-cy-vy*z.t);
            double dm=z.mag-cm,sg=std::hypot(.115,1.06/z.sn);
            double w=1./(sg*sg+pos*.40)/(1.+sq(dm/.54));
            a+=w;b+=w*z.t;c+=w*z.t*z.t;dx+=w*z.x;dy+=w*z.y;tx+=w*z.t*z.x;ty+=w*z.t*z.y;
            double wt=1./(.018+sq(1.086/z.sn)+.55*dm*dm+.15*pos);
            fm+=wt*z.mag;ww+=wt;
        }
        det=a*c-b*b;
        if(!(det>0.)||ww<=0.)return out;
        cx=(c*dx-b*tx)/det;cy=(c*dy-b*ty)/det;
        vx=(a*tx-b*dx)/det;vy=(a*ty-b*dy)/det;cm=fm/ww;
    }
    xx.clear();mm.clear();
    for(const auto &z:sample){
        double d=std::hypot(z.x-cx-z.t*vx,z.y-cy-z.t*vy);
        if(d<.65&&std::abs(z.mag-cm)<.75){
            xx.push_back(d);mm.push_back(std::abs(z.mag-cm));
        }
    }
    if(xx.size()<2 || xx.size()*2<sample.size())return out;
    double pos_scatter=middle(xx),phot_scatter=1.48*middle(mm);
    if(pos_scatter>.45||phot_scatter>.49)return out;
    double fraction=std::exp(-.92103404*(cm+transfer_shift-sky.mag[row]));
    if(!(fraction>.045 && fraction<8.))return out;
    out.x=cx;out.y=cy;out.flux=fraction;
    out.pos_sigma=std::sqrt(.065*.065+c/det+.42*pos_scatter*pos_scatter+sq(.35*transfer_error));
    out.flux_sigma=std::sqrt(.085*.085+1./ww+sq(.72*phot_scatter)+sq(transfer_error))*.92103404;
    out.nights=xx.size();return out;
}

static int train_transfer(Sky&sky,const int*rows,int n,double seconds){
    std::array<std::array<std::vector<double>,108>,2> examples;
    auto end=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
    int used=0;
    for(int i=0;i<n&&std::chrono::steady_clock::now()<end;i++){
        int row=rows[i];auto same=reference_light(sky,row);
        if(same.stratum>=0||same.nights<2||std::hypot(same.x,same.y)>.34||
           !(same.flux>.72&&same.flux<1.36))continue;
        auto other=reference_light(sky,row,true);
        if(other.stratum<0||other.nights<4||std::hypot(other.x-same.x,other.y-same.y)>.32)continue;
        // The target is the independent same-band reference. Both fits
        // omit the measured peak. Group entire sky cells into opposite folds.
        double shift=1.0857362*std::log(other.flux/same.flux);
        if(!(std::abs(shift)<1.3))continue;
        int fold=1-sky_side(sky,row),k=other.stratum;
        examples[fold][k].push_back(shift);examples[fold][90+k/5].push_back(shift);used++;
    }
    for(int fold=0;fold<2;fold++)for(int k=0;k<90;k++){
        auto &general=examples[fold][90+k/5],&local=examples[fold][k];
        if(general.size()<8)continue;
        auto data=general;
        double broad=middle(data),small=local.empty()?broad:middle(local);
        double reliability=double(general.size())/(general.size()+18.);
        double group=double(local.size())/(local.size()+14.);
        double center=reliability*(group*small+(1.-group)*broad);
        data=local.size()>=20?local:general;
        for(double &v:data)v=std::abs(v-center);
        double err=1.48*middle(data);
        sky.transfer[fold][k]={std::clamp(center,-.42,.42),
                        std::sqrt(reliability*std::max(0.,sq(std::min(.55,err))-.14*.14))};
    }
    return used;
}

struct Component {double x,y,mag,sn,change,scatter,fraction;bool valid;};
static double steady_background(const Sky &sky,int row,const ReferenceLight &r,double density){
    if(r.nights<2)return 0.;
    double center=std::hypot(.075,1.08/std::max(4.5f,sky.sn[row]));
    double var=sq(center)+sq(r.pos_sigma);
    double fv=sq(.065)+sq(1./sky.sn[row])+sq(r.flux*r.flux_sigma);
    double chi=(r.x*r.x+r.y*r.y)/var+sq(1.-r.flux)/fv;
    if(chi>14.)return 0.;
    double p=std::exp(-.5*chi)/(15.74961*std::sqrt(fv)*var);
    double reliable=(r.nights-1.)/(r.nights+2.);
    return std::min(4.8,std::log1p(reliable*p/std::max(.00006,1.086*density)));
}
static Component separate(const Sky &sky,int row,const ReferenceLight &r){
    Component z{};
    if(r.nights<2 || !std::isfinite(sky.mag[row]))return z;
    double sigma=std::clamp(sky.seeing[sky.ex[row]]/2.355,.27,1.1);
    double d2=r.x*r.x+r.y*r.y;
    double contribution=r.flux*std::exp(-d2/(4.*sigma*sigma));
    double f=1.-contribution;
    if(f<.16 || f>.95)return z;
    double gain=contribution/f,shift2=d2*gain*gain;
    if(shift2>sq(2.5*sigma)||shift2>1.45*1.45)return z;
    double e0=std::hypot(.072,1.05/std::max(4.5f,sky.sn[row]));
    double frac_error=std::hypot(1./sky.sn[row],contribution*r.flux_sigma);
    double center_var=(sq(e0)+sq(contribution*r.pos_sigma))/sq(f)+
                      d2*sq(contribution*std::hypot(1./sky.sn[row],r.flux_sigma))/sq(f*f);
    double fp=std::hypot(.075,1./sky.sn[row]);
    double stationary=sq((1.-r.flux)/std::hypot(fp,r.flux*r.flux_sigma))+
                      d2/(sq(e0)+sq(r.pos_sigma));
    z.x=sky.xy[2*row]-as*gain*r.x;z.y=sky.xy[2*row+1]-as*gain*r.y;
    z.mag=sky.mag[row]-2.5*std::log10(f)-1.0857362*shift2/(4.*sigma*sigma);
    z.sn=f/std::max(.0001,frac_error);z.change=stationary;z.scatter=std::sqrt(center_var);
    z.fraction=f;z.valid=std::isfinite(z.mag+z.scatter)&&z.sn>2.5;return z;
}
