// Differential astrometry on a member night. Use comparison peaks in the
// same two actual images to measure the chance of a matching second peak.
namespace companions {
struct Query {
    int row, exposure;
    float dx, dy, dm, error, scatter;
};
struct Result {
    int row, controls, covered;
    float statistic, sky, expected, dx, dy, dm;
};
struct Match {
    int row=-1;
    double weight=0., dx=0., dy=0., dm=0.;
};
void match(const original::Sky &sky, const original::Frame &frame,
           int anchor, const Query &q, Match &best){
    const double x=sky.xy[2*anchor]+as*q.dx, y=sky.xy[2*anchor+1]+as*q.dy;
    const double mag=sky.mag[anchor]+q.dm;
    const double radius=3.6*q.error;
    sky.nearby(frame,x,y,radius*as,radius*as,[&](int j){
        if((sky.flag[j]&1)||!std::isfinite(sky.mag[j]))return;
        const double dx=(sky.xy[2*j]-x)/as,dy=(sky.xy[2*j+1]-y)/as,dm=sky.mag[j]-mag;
        const double cost=(dx*dx+dy*dy)/sq(q.error)+sq(dm/q.scatter);
        if(cost>15.)return;
        const double weight=std::exp(-.5*cost);
        if(weight>best.weight)best={j,weight,dx,dy,dm};
    });
}

// Cheap coverage at a control position. Require detections close on both
// sides in each dimension to avoid calling a CCD gap an empty image.
bool covered(const original::Sky &sky,const original::Frame &im,double x,double y){
    if(x<im.xmin||x>im.xmax||y<im.ymin||y>im.ymax||!im.nx)return false;
    bool left=false,right=false,above=false,below=false;double nearest=1.e5;
    sky.nearby(im,x,y,26.*as,26.*as,[&](int row){
        double dx=(sky.xy[2*row]-x)/as,dy=(sky.xy[2*row+1]-y)/as;
        double rr=dx*dx+dy*dy;
        if(rr>676.)return;
        nearest=std::min(nearest,rr);
        left|=dx<0.;right|=dx>0.;above|=dy>0.;below|=dy<0.;
    });
    return nearest<324.&&left&&right&&above&&below;
}

void examine(const original::Sky &sky,const Query &q,const uint8_t *discovery,Result &out){
    out={-1,0,0,0.f,0.f,0.f,0.f,0.f,0.f};
    if(q.row<0||q.row>=sky.size||q.exposure<0||q.exposure>=int(sky.images.size()))return;
    const auto &first=sky.images[sky.ex[q.row]],&second=sky.images[q.exposure];
    if(!first.nx||!second.nx||first.exposure.night!=second.exposure.night)return;
    const double x=sky.xy[2*q.row],y=sky.xy[2*q.row+1],sx=q.dx*as,sy=q.dy*as;
    Match real;
    match(sky,second,q.row,q,real);
    if(real.row<0&&!covered(sky,second,x+sx,y+sy))return;
    out.covered=1;out.row=real.row;out.statistic=real.weight;
    out.dx=real.dx;out.dy=real.dy;out.dm=real.dm;
    if(real.row<0)return;

    struct Control{uint32_t key;int row;};
    std::vector<Control> sample;
    const int wanted=88;
    const double mag=sky.mag[q.row],signal=sky.sn[q.row];
    for(double r:{155.,390.,850.}){
        sample.clear();
        sky.nearby(first,x,y,r*as,r*as,[&](int j){
            if((sky.flag[j]&1)||!std::isfinite(sky.mag[j]))return;
            if(discovery&&discovery[j]!=discovery[q.row])return;
            const double dx=(sky.xy[2*j]-x)/as,dy=(sky.xy[2*j+1]-y)/as;
            const double d2=dx*dx+dy*dy;
            if(d2<sq(5.)||d2>sq(r)||std::abs(mag-sky.mag[j])>.65)return;
            if(sky.sn[j]<.48*signal||sky.sn[j]>2.1*signal)return;
            // Deterministic uniform sampling of the local catalogue.
            uint32_t key=uint32_t(j)*2654435761u+uint32_t(q.exposure)*2246822519u;
            key^=key>>16;
            sample.push_back({key,j});
        });
        if(sample.size()>=wanted||r>800.)break;
    }
    if(sample.size()>wanted){
        std::nth_element(sample.begin(),sample.begin()+wanted,sample.end(),
                         [](const auto &a,const auto &b){return a.key<b.key;});
        sample.resize(wanted);
    }
    const int bin=std::clamp(int((mag+q.dm-10.)*4.),0,79);
    const double uniform=std::clamp(second.rho[bin]*2.*3.141593*sq(q.error)*2.50663*q.scatter,1.e-5,.09);
    double count=0.,wt=0.;
    for(const auto &p:sample){
        const int j=p.row;
        if(!covered(sky,second,sky.xy[2*j]+sx,sky.xy[2*j+1]+sy))continue;
        Match other;
        match(sky,second,j,q,other);
        const double wgt=std::exp(-.5*sq((sky.mag[j]-mag)/.41));
        count+=wgt*other.weight;wt+=wgt;
        out.controls++;
    }
    // Shrink towards a stationary component when the displacement is
    // unresolved. Use a diffuse component for random/foreground tracks.
    double prior=.13*std::exp(-.25*(sq(q.dx)+sq(q.dy))/sq(q.error))+uniform+.006;
    out.sky=(count+9.*prior+.23)/(wt+10.);
    out.expected=uniform;
}
}
extern "C" int companion_measurements(const original::Sky *sky,const companions::Query *queries,
        int count,const uint8_t *classes,int threads,double seconds,companions::Result *out){
    try{
        auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(int t=0;t<threads;t++)pool.emplace_back([&]{
            while(Clock::now()<end){
                int i=next.fetch_add(1);if(i>=count)break;
                companions::examine(*sky,queries[i],classes,out[i]);
            }
        });
        for(auto &thread:pool)thread.join();
        return std::min(int(next),count);
    }catch(...){return -1;}
}
