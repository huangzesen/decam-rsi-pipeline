// Predict two further shutter positions conditional on one measured pair.
namespace chord_returns {
struct Pair {float pos[4],mag,quality;int a,b;};
struct Model {double center[4],transform[4][4],poly[15][4],first[3],slope[2][2],last[3];};
struct Probe {double dx,dy,scale;int e,pad;};
struct Peak {float x,y,quality;int row;};
struct RawPixel {float x,y,mag,sigma,shape;unsigned id;};

void predict(const Pair &p,const Model &m,double *pos){
    double z[4]{};
    for(int j=0;j<4;j++)for(int k=0;k<4;k++)z[k]+=(p.pos[j]-m.center[j])*m.transform[j][k];
    std::copy_n(m.poly[0],4,pos);
    for(int j=0;j<4;j++)for(int k=0;k<4;k++)pos[k]+=z[j]*m.poly[j+1][k];
    int i=5;
    for(int j=0;j<4;j++)for(int l=j;l<4;l++,i++)
        for(int k=0;k<4;k++)pos[k]+=z[j]*z[l]*m.poly[i][k];
}
void lookup(const original::Sky &sky,const Probe &p,double mx,double my,const double *v,
            double mag,double lever,int limit,std::vector<Peak> &list){
    const auto &im=sky.images[p.e];
    double x=(mx-p.dx)*as/p.scale,y=(my-p.dy)*as/p.scale;
    double xx=v[0],xy=v[1],yy=v[2],det=xx*yy-xy*xy;
    if(!im.nx||!(det>1.e-6)||xx+yy>1.1e8)return;
    double wx=3.1*std::sqrt(xx)*as,wy=3.1*std::sqrt(yy)*as;
    if(x+wx<im.xmin||x-wx>im.xmax||y+wy<im.ymin||y-wy>im.ymax)return;
    int bin=std::clamp(int((mag-10.)*4.),0,79);
    double area=std::sqrt(det),contamination=area*im.rho[bin];
    int tier=0;
    if(contamination>4.){
        if(!sky.population)return;
        tier=1;
        if(area*im.transient[0][bin]>8.)tier=2;
        if(area*im.transient[tier-1][bin]>50.)return;
    }
    double color=.24*std::abs(lever-sky.lever[p.e]);
    size_t start=list.size();
    sky.ellipse(im,x,y,xx,xy,yy,[&](int row){
        if(tier && sky.population[row]<tier)return;
        if(sky.flag[row]&1)return;
        double dm=sky.mag[row]-mag,sm=std::hypot(std::hypot(.27,color),1.09/sky.sn[row]);
        if(!(std::abs(dm)<2.95*sm))return;
        double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
        double chi=(yy*dx*dx+xx*dy*dy-2.*xy*dx*dy)/det;if(!(chi<9.7))return;
        double sh=(sky.sharp[row]-1.)/std::hypot(.25,3.6/sky.sn[row]);
        if(!(std::abs(sh)<4.1))return;
        double q=-.39*chi-.48*sq(dm/sm)-.25*sq(sh)+.67*(sky.population?sky.population[row]:0);
        list.push_back({float(sky.xy[2*row]/as*p.scale+p.dx),float(sky.xy[2*row+1]/as*p.scale+p.dy),
                        float(q),row});
    });
    if(list.size()>start+limit){
        std::nth_element(list.begin()+start,list.begin()+start+limit,list.end(),
                         [](auto &a,auto &b){return a.quality>b.quality;});
        list.resize(start+limit);
    }
}
bool fit(const original::Sky &sky,int *ids,double gamma,double &g,double &rms){
    std::sort(ids,ids+4,[&](int a,int b){return sky.images[sky.ex[a]].exposure.t<sky.images[sky.ex[b]].exposure.t;});
    quartet_fit::Shutter e[4];float x[4],y[4];
    for(int i=0;i<4;i++){
        const auto &z=sky.images[sky.ex[ids[i]]].exposure;e[i]={z.t,z.ox,z.oy,z.oz};
        x[i]=sky.xy[2*ids[i]]/as;y[i]=sky.xy[2*ids[i]+1]/as;
    }
    float a[]={x[0],y[0],x[1]-x[0],y[1]-y[0]},b[]={x[2],y[2],x[3]-x[2],y[3]-y[2]};
    quartet_fit::State s;if(!quartet_fit::solve(a,b,e,gamma,s,rms))return false;
    g=s.g;return true;
}
}
extern "C" int conditional_frame(const original::Sky *sky,int e,int *shape,chord_returns::RawPixel *points,
                                 int *starts,float *density){
    if(e<0||e>=int(sky->images.size()))return -1;
    const auto &im=sky->images[e];
    int cells=im.starts.size();
    int dims[]={im.bx,im.by,im.nx,im.ny,3*cells,int(im.rows.size())};
    std::copy_n(dims,6,shape);
    if(!points||!starts||!density)return 0;
    std::copy(im.starts.begin(),im.starts.end(),starts);
    for(int b=0;b<80;b++){
        density[b]=im.rho[b];density[80+b]=im.transient[0][b];density[160+b]=im.transient[1][b];
    }
    for(int k=0;k<int(im.rows.size());k++){
        int r=im.rows[k];double s=std::max(4.5,double(sky->sn[r]));
        double sharp=(sky->sharp[r]-1.)/std::hypot(.25,3.6/s);
        unsigned flags=(sky->flag[r]&1)||!std::isfinite(sky->mag[r])||!(std::abs(sharp)<4.1);
        unsigned tier=sky->population?sky->population[r]:0;
        points[k]={float(sky->xy[2*r]/as),float(sky->xy[2*r+1]/as),sky->mag[r],float(1.09/s),
                   float(.25*sq(sharp)),unsigned(r)|(tier<<29)|(flags<<31)};
    }
    for(int cell=0;cell+1<cells;cell++){
        auto first=points+im.starts[cell],end=points+im.starts[cell+1];
        auto clean=std::partition(first,end,[](auto &p){return ((p.id>>29)&3)>=2;});
        auto transient=std::partition(clean,end,[](auto &p){return ((p.id>>29)&3)>=1;});
        starts[cells+cell]=transient-first;starts[2*cells+cell]=clean-first;
    }
    if(cells){starts[2*cells-1]=0;starts[3*cells-1]=0;}
    return 0;
}
extern "C" int conditional_chord(const original::Sky *sky,const chord_returns::Pair *pairs,int n,
                  const chord_returns::Model *model,const chord_returns::Probe *third,int nt,
                  const chord_returns::Probe *fourth,int nf,double lever,double gamma,double seconds,
                  int cap,ArcSeed *seeds,int64_t *stats){
    using namespace chord_returns;
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        const auto &m=*model;std::vector<ArcSeed> hits;std::vector<Peak> middles,lasts;
        auto better=[](const auto &a,const auto &b){return a.score>b.score;};
        for(int j=0;j<n;j++){
            if((j&7)==0&&expired(end))break;
            stats[0]++;const Pair &p=pairs[j];
            double z[4];predict(p,m,z);middles.clear();
            for(int e=0;e<nt;e++)lookup(*sky,third[e],z[0],z[1],m.first,p.mag,lever,16,middles);
            stats[1]+=middles.size();
            for(const auto &q:middles){
                double dx=q.x-z[0],dy=q.y-z[1];
                double x=z[2]+dx*m.slope[0][0]+dy*m.slope[1][0];
                double y=z[3]+dx*m.slope[0][1]+dy*m.slope[1][1];
                lasts.clear();
                for(int e=0;e<nf;e++)lookup(*sky,fourth[e],x,y,m.last,p.mag,lever,9,lasts);
                stats[2]+=lasts.size();
                for(const auto &r:lasts){
                    int ids[]={p.a,p.b,q.row,r.row};double g,rms;
                    if(!fit(*sky,ids,gamma,g,rms))continue;
                    hits.push_back({{ids[0],ids[1],ids[2],ids[3]},float(g),
                                    float(p.quality+q.quality+r.quality-1.7*sq(rms/.22))});
                }
            }
            if(int(hits.size())>4*cap){
                std::nth_element(hits.begin(),hits.begin()+cap,hits.end(),better);hits.resize(cap);
            }
        }
        stats[3]+=hits.size();
        if(int(hits.size())>cap){
            std::nth_element(hits.begin(),hits.begin()+cap,hits.end(),better);hits.resize(cap);
        }
        std::copy(hits.begin(),hits.end(),seeds);return hits.size();
    }catch(...){return -1;}
}
