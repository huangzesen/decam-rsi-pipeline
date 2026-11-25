// Search single detections in a succession of co-moving frames. Each frame
// defines a chart for the positions of an orbit, over a finite time interval.
// Four measured positions in one cell solve continuously for the orbit.
namespace {
namespace orbit_tiles {
struct Trial {
    double radius, speed, tilt, radial, arcsec, span, timephase, xphase, yphase, magphase;
};
struct Cell {uint64_t key;int head,count;};
struct Proposal {int id[4];double gamma,merit;int witness=-1;};
static uint64_t identity(const Proposal &p){
    std::array<int,4> ids;std::copy_n(p.id,4,ids.begin());std::sort(ids.begin(),ids.end());
    uint64_t h=1469598103934665603ull;
    for(int id:ids)h=(h^uint32_t(id))*1099511628211ull;
    return h;
}
static void compact(std::vector<Proposal> &seeds,size_t limit){
    std::sort(seeds.begin(),seeds.end(),[](auto &a,auto &b){return a.merit>b.merit;});
    std::unordered_set<uint64_t> seen;
    size_t count=0;
    for(auto &p:seeds){
        if(!seen.insert(identity(p)).second)continue;
        seeds[count++]=p;
        if(count==limit)break;
    }
    seeds.resize(count);
}
struct Law {
    double radius,co,si;
    Law(const Trial &p,double dt){
        double r=p.radius,w=std::sqrt(mu/(r*r*r));
        double ia=std::max(.025,2.-sq(p.speed)-sq(p.radial));
        double mean=w*ia*std::sqrt(ia)*dt;
        double ecc=1.-ia,rs=p.radial*std::sqrt(ia),anomaly=mean;
        for(int j=0;j<6;j++)
            anomaly-=(anomaly-ecc*std::sin(anomaly)+rs*(1.-std::cos(anomaly))-mean)
                    /(1.-ecc*std::cos(anomaly)+rs*std::sin(anomaly));
        double f=1.-(1.-std::cos(anomaly))/ia;
        double g=w*dt+(std::sin(anomaly)-anomaly)/(ia*std::sqrt(ia));
        co=f+g*p.radial;si=g*p.speed;
        double rn=std::hypot(co,si);radius=r*rn;co/=rn;si/=rn;
    }
    Vec2 sky(double x,double y,Three ob,Three axis)const{
        Three v{x,y,1.};v=v*(1./v.length());
        double d=v.product(ob),rho=std::sqrt(radius*radius-ob.product(ob)+d*d)-d;
        Three u=(ob+v*rho)*(1./radius);
        auto back=u*co-axis.cross(u)*si+axis*((1.-co)*axis.product(u));
        return {back.x/back.z,back.y/back.z};
    }
};
struct Map {
    int group;
    Vec2 center,xy,dx,dy,xx,xy2,yy;
    Vec2 at(double x,double y)const{
        x-=center.x;y-=center.y;
        return xy+dx*x+dy*y+xx*(x*x)+xy2*(x*y)+yy*(y*y);
    }
};
struct Record {float x,y,mag,q;int e,night;};
struct Run {
    const Source *s;const int *owner;const Witness &w;
    int n,limit;const std::vector<Record> &src;
    const std::vector<Vec2> &centers;
    Three pole;double start;Clock::time_point end;
    std::vector<Cell> table;
    std::vector<int> previous,busy;
    std::vector<Proposal> seeds,triangles;
    std::array<long long,8> counts{};
    double cutoff=-1.e20,cut3=-1.e20;
    struct Prediction {int witness;float gain,gamma;};
    std::unordered_map<uint64_t,Prediction> forecasts;

    void predict(Proposal &p){
        auto key=identity(p);
        auto found=forecasts.find(key);
        Prediction answer{-1,0.f,float(p.gamma)};
        if(found==forecasts.end()){
            Unit first{p.id[0],p.id[1],0.,0.},second{p.id[2],p.id[3],0.,0.};
            Orbit fit;Domain domain{14.,9200.,7000.,true};
            if(quartet(s,first,second,p.gamma,fit,false,domain)){
                answer.gain=w.evidence(fit,s,first,second,answer.witness);
                answer.gamma=fit.g;
            }
            if(forecasts.size()>=320000)forecasts.clear();
            forecasts.emplace(key,answer);
        }else answer=found->second;
        p.merit+=.76*std::min(19.f,answer.gain);
        p.witness=answer.witness;p.gamma=answer.gamma;
    }
    void prepare(const Trial &p){
        size_t cap=1;while(cap<size_t(n)*1.45)cap*=2;
        table.assign(cap,{0,-1,0});previous.resize(n);busy.clear();
        std::vector<Map> maps(w.images.size());
        Three east=pole.cross({0.,0.,1.});east=east*(1./east.length());
        Three north={-east.y,east.x,0.};
        Three dir=east*std::cos(p.tilt)+north*std::sin(p.tilt);
        Three axis={-dir.y,dir.x,0.};
        for(size_t j=0;j<maps.size();j++){
            auto &m=maps[j];const auto &ex=w.images[j].e;
            m.group=int(std::floor((ex.t-start)/p.span+p.timephase));
            double ref=start+(m.group+.5-p.timephase)*p.span;
            Law law(p,ex.t-ref);
            m.center=centers[j];double x=m.center.x,y=m.center.y;
            auto sky=[&](double a,double b){
                return law.sky(a,b,{ex.ox,ex.oy,ex.oz},axis)*(1./(as*p.arcsec));
            };
            double d=.015;auto z=sky(x,y),right=sky(x+d,y),left=sky(x-d,y);
            auto up=sky(x,y+d),down=sky(x,y-d);
            m.xy=z;m.dx=(right-left)*(.5/d);m.dy=(up-down)*(.5/d);
            m.xx=(right+left-z*2.)*(.5/(d*d));m.yy=(up+down-z*2.)*(.5/(d*d));
            m.xy2=(sky(x+d,y+d)-right-up+z)*(1./(d*d));
        }
        for(int i=0;i<n;i++){
            auto &a=src[i];const auto &m=maps[a.e];
            Vec2 xy=m.at(a.x,a.y);
            if(!std::isfinite(xy.x+xy.y))continue;
            int x=std::floor(xy.x+p.xphase),y=std::floor(xy.y+p.yphase);
            int mb=std::floor(a.mag/.96+p.magphase);
            uint64_t key=(uint64_t(uint32_t(x)&0xfffff)<<40)|(uint64_t(uint32_t(y)&0xfffff)<<20)|
                         (uint64_t(uint32_t(m.group)&0xfff)<<8)|(unsigned(mb)&255);
            size_t at=permute(key)&(cap-1);
            while(table[at].count&&table[at].key!=key)at=(at+1)&(cap-1);
            auto &cell=table[at];previous[i]=cell.head;
            cell.head=i;cell.key=key;if(++cell.count==3)busy.push_back(at);
        }
    }
    void cell(const Trial &p,const int *d,int nrow){
        // Eliminate position and rate using the two temporally outer
        // measurements. At each interior epoch the remaining residual
        // constrains the same two quantities: inverse range and range rate.
        double g=1./p.radius,vc=std::sqrt(mu*g*g*g);
        struct Middle {Vec2 off,G,H;int row;double weight;};
        std::array<Middle,24> mids;
        for(int ia=0;ia+2<nrow;ia++)for(int ib=ia+2;ib<nrow;ib++){
            int a=d[ia],b=d[ib];auto &u=s[a],&v=s[b];
            double dt=v.t-u.t;if(dt<40.)continue;
            double dm=std::max(0.,std::abs(u.mag-v.mag)-.25*std::abs(u.color-v.color));
            if(dm>.64)continue;
            double mean=.5*(u.mag+v.mag),lever=.5*(u.color+v.color);
            Propagator nominal(u,v,g,vc*p.radial),dg(u,v,g*1.025,vc*p.radial),
                       dh(u,v,g,vc*(p.radial+.12));
            int num=0;
            for(int j=ia+1;j<ib;j++){
                int c=d[j];auto &z=s[c];
                if(z.night==u.night||z.night==v.night)continue;
                double dm3=std::max(0.,std::abs(z.mag-mean)-.23*std::abs(z.color-lever));
                if(dm3>.58)continue;
                Endpoint e{z.t,z.ox,z.oy,z.oz};
                Vec2 mid=nominal(e),off=Vec2{z.x,z.y}-mid;
                Vec2 G=(dg(e)-mid)*40.,H=(dh(e)-mid)*(1./.12);
                double gg=dot(G,G)+1.e-20,perp=G.x*off.y-G.y*off.x,hr=G.x*H.y-G.y*H.x;
                double phase=std::abs(dt)*vc;
                double sigma=as*(.21+.74/z.sn)+.015*phase*phase*phase;
                double dx=dot(off,G)/gg;
                if(std::abs(perp)>.68*std::abs(hr)+4.*sigma*std::sqrt(gg)||
                   std::abs(dx)>.23+.68*std::abs(dot(G,H))/gg+4.*sigma/std::sqrt(gg))continue;
                mids[num++]={off,G,H,c,1./sq(sigma)};
                // A compact, well-localized triple may predict another
                // season even when no fourth source is in the coarse cell.
                double cross=hr/std::sqrt(gg),gap=perp/std::sqrt(gg);
                double h=gap*cross/(sq(cross)+sq(sigma/.35));
                double frac=dx-h*dot(G,H)/gg;
                double sc=sq(gap)/(sq(sigma)+sq(.35*cross))+sq(frac/.13);
                double volume=std::hypot(sigma,.13*std::sqrt(gg))*std::hypot(sigma,.35*cross)/sq(as);
                double merit=(u.quality+v.quality+z.quality)-.50*(sc+sq(dm3/.29)+sq(dm/.36))
                             -.64*std::log1p(volume);
                if(sc<12.&&merit>cut3){
                    triangles.push_back({{a,b,c,-1},g*(1.+frac),merit});
                    if(triangles.size()>48000){compact(triangles,20000);cut3=triangles.back().merit;}
                }
            }
            for(int j=0;j<num;j++)for(int k=0;k<j;k++){
                auto &c=mids[j],&z=mids[k];
                if(s[c.row].night==s[z.row].night)continue;
                double aa=1./sq(.15),bb=1./sq(.39),ab=0.,ay=0.,by=0.,ss=0.;
                for(const auto *t:{&c,&z}){
                    double w=t->weight;
                    aa+=w*dot(t->G,t->G);bb+=w*dot(t->H,t->H);ab+=w*dot(t->G,t->H);
                    ay+=w*dot(t->off,t->G);by+=w*dot(t->off,t->H);ss+=w*dot(t->off,t->off);
                }
                double det=aa*bb-ab*ab,frac=(bb*ay-ab*by)/det,h=(aa*by-ab*ay)/det;
                double chi=ss-frac*ay-h*by;
                if(chi>22.||std::abs(frac)>.24||std::abs(h+p.radial)>.92)continue;
                double sc=u.quality+v.quality+s[c.row].quality+s[z.row].quality-.38*chi;
                if(sc+14.5<cutoff)continue;
                Proposal proposal{{a,b,c.row,z.row},g*(1.+frac),sc};
                predict(proposal);
                if(proposal.merit<cutoff)continue;
                seeds.push_back(proposal);counts[3]++;
                if(seeds.size()>68000){compact(seeds,30000);cutoff=seeds.back().merit;}
            }
        }
    }
    void scan(const Trial &p){
        prepare(p);counts[0]++;
        std::array<int,24> picked;
        std::vector<int> rows;
        for(size_t i=0;i<busy.size();i++){
            if((i&255)==0&&expired(end))break;
            auto &c=table[busy[i]];rows.clear();
            for(int j=c.head;j>=0;j=previous[j])rows.push_back(j);
            // Independent nights get first use of a crowded cell.
            if(rows.size()>12){
                std::sort(rows.begin(),rows.end(),[&](int a,int b){return src[a].q>src[b].q;});
            }
            int used=0;
            for(int j:rows){
                int copies=0;
                for(int k=0;k<used;k++)copies+=src[picked[k]].night==src[j].night;
                if(copies>=(rows.size()>12?1:2))continue;
                picked[used++]=j;
                if(used==12)break;
            }
            if(used<3)continue;
            std::sort(picked.begin(),picked.begin()+used,[&](int a,int b){return s[a].t<s[b].t;});
            if(s[picked[used-1]].t-s[picked[0]].t<40.)continue;
            counts[1]++;counts[2]+=used;
            cell(p,picked.data(),used);
        }
    }
};
}
}

extern "C" int local_orbit_charts(const Source *s,const int *owner,int n,const Witness *w,const double *pole,
        const orbit_tiles::Trial *trials,int nt,int threads,double seconds,int cap,Certified *out,long long *counts){
    using namespace orbit_tiles;
    try{
        if(n<4||nt<1)return 0;
        auto start=Clock::now();
        auto stop_at=[&](double f){return start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds*f));};
        std::vector<Record> sources(n);std::vector<Vec2> centers(w->images.size());
        std::vector<int> used(centers.size());
        std::vector<double> depth(centers.size(),0.),color(centers.size(),0.);
        double tmin=1.e20;
        for(int i=0;i<n;i++){
            int e=w->objects[owner[i]].exposure;used[e]++;
            sources[i]={float(s[i].x),float(s[i].y),float(s[i].mag),float(s[i].quality),e,s[i].night};
            centers[e]=centers[e]+Vec2{s[i].x,s[i].y};tmin=std::min(tmin,s[i].t);
            depth[e]+=s[i].mag+2.5*std::log10(std::max(1.,double(s[i].sn))/5.);
            color[e]=s[i].color;
        }
        for(size_t i=0;i<centers.size();i++){
            centers[i]=centers[i]*(1./std::max(1,used[i]));
            depth[i]=used[i]?depth[i]/used[i]:23.;
        }
        double norm=std::max(1.e-12,std::hypot(pole[0],pole[1]));
        double east[2]={pole[1]/norm,-pole[0]/norm};
        std::vector<Run> runs;
        for(int j=0;j<threads;j++)
            runs.push_back({s,owner,*w,n,cap,sources,centers,{pole[0],pole[1],pole[2]},tmin,stop_at(.64)});
        std::atomic<int> next{0};std::vector<std::thread> pool;
        for(auto &r:runs)pool.emplace_back([&,pr=&r]{
            while(!expired(pr->end)){
                int j=next.fetch_add(1);if(j>=nt)break;
                pr->scan(trials[j]);
            }
        });
        for(auto &t:pool)t.join();
        std::vector<Proposal> seeds,tri;
        for(auto &r:runs){
            seeds.insert(seeds.end(),r.seeds.begin(),r.seeds.end());
            tri.insert(tri.end(),r.triangles.begin(),r.triangles.end());
            for(int i=0;i<4;i++)counts[i]+=r.counts[i];
        }
        compact(seeds,90000);compact(tri,48000);counts[4]=seeds.size();counts[5]=tri.size();
        std::vector<std::vector<Certified>> results(threads),fours(threads);next=0;pool.clear();
        std::atomic<size_t> next_three{0},next_four{0};
        for(int j=0;j<threads;j++)pool.emplace_back([&,j]{
            auto &res=results[j];auto &four=fours[j];
            while(!expired(stop_at(1.))){
                int k=next.fetch_add(1);
                bool triple=k%7==6;
                size_t ix=triple?next_three.fetch_add(1):next_four.fetch_add(1);
                if(ix>=(triple?tri:seeds).size()){
                    triple=!triple;
                    ix=triple?next_three.fetch_add(1):next_four.fetch_add(1);
                }
                auto &queue=triple?tri:seeds;
                if(ix>=queue.size())break;
                auto &p=queue[ix];
                int ids[5];for(int i=0;i<(triple?3:4);i++)ids[i]=owner[p.id[i]];
                ids[4]=p.witness;
                size_t before=res.size();
                if(triple)follow_uncertain(*w,ids,3,p.gamma,res,true);
                else{
                    Certified c{};
                    if((p.witness>=0&&certify_one(*w,ids,5,p.gamma,c))||
                       certify_one(*w,ids,4,p.gamma,c,false,true)){
                        c.merit+=.20*p.merit;
                        if(c.nights>=5)res.push_back(c);
                        else{
                            c.merit=26.+.60*cadence::evidence(*w,c,depth.data(),color.data(),east)
                                     +.55*std::min(4,c.count-c.nights);
                            four.push_back(c);
                        }
                    }
                }
                for(size_t j=before;j<res.size();j++)
                    res[j].merit+=.75*cadence::evidence(*w,res[j],depth.data(),color.data(),east);
                if(res.size()>size_t(2*cap))compact_certified(res,cap);
                if(four.size()>size_t(2*cap))compact_certified(four,cap);
            }
        });
        for(auto &t:pool)t.join();counts[6]=next.load();
        std::vector<Certified> five,four;
        for(int j=0;j<threads;j++){
            five.insert(five.end(),results[j].begin(),results[j].end());
            four.insert(four.end(),fours[j].begin(),fours[j].end());
        }
        compact_certified(five,cap*3/4);compact_certified(four,cap-five.size());
        counts[7]=five.size();
        std::copy(five.begin(),five.end(),out);std::copy(four.begin(),four.end(),out+five.size());
        return int(five.size()+four.size());
    }catch(...){return -1;}
}
