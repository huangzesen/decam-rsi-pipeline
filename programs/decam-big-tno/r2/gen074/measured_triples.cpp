// The same measured-pair join runs in the host and CUDA entry points.
#ifndef __CUDACC__
#include "night_query.cpp"
#endif
namespace measured {
using namespace night_query;
struct Pair {int a,b; float u,v,merit;};
struct Triple {int a,b,c; float merit;};
struct Grid {float x,y; int width,height,offset;};
constexpr int partners=8, alternatives=2;
constexpr int possible=48;
struct Returns {
    int count,view[possible]; Step step[possible]; float corr[possible][possible][4];
    float logrange[features],crossrange[possible][2];
};

NINLINE void root_coordinates(const Model &m,const Point &p,float *z){
    const Step &s=m.step[0];
    z[0]=(p.x-s.coef[0][0])/s.L[0];
    z[1]=(p.y-s.coef[0][1]-s.L[1]*z[0])/s.L[2];
}
NINLINE void partners_for(const Point *p,const int *edges,const Field *fields,const Model &m,
                          int root,Pair *out){
    for(int j=0;j<partners;j++)out[j]={-1,-1,0,0,0};
    const Point &a=p[root];
    const Field &A=fields[m.view[0]],&B=fields[m.view[1]];
    float z[2],v[6];
    root_coordinates(m,a,z);
    if(z[0]*z[0]+z[1]*z[1]>24.f)return;
    Position pos=forecast(m.step[1],a.x,a.y,v,basis(z,2,v));
    float w=.047f+sq(a.sigma)+sq(.28f*(A.lever-B.lever));
    Hit h[partners];
    int n=return_hits<partners>(p,edges,fields,B,pos,a.mag,w,3.45f,h,m.family>=0);
    for(int j=0;j<n;j++)out[j]={root,h[j].index,h[j].u,h[j].v,h[j].merit};
}
NINLINE float visible(const Grid &g,const float *depth,float x,float y,float mag){
    int i=int(floorf((x-g.x)/270.f)), j=int(floorf((y-g.y)/270.f));
    if(i<0 || j<0 || i>=g.width || j>=g.height)return 0.f;
    float d=depth[g.offset+i*g.height+j]-mag;
    if(d< -2.4f)return 0.f;
    return .97f/(1.f+expf(-minimum(12.f,(d+.12f)/.29f)));
}
NINLINE float return_probability(const Model &m,const Point &a,const float *v,
                                 const Grid *grids,const float *depth,float mag,const Field *fields){
    // Integrate the live footprint along the large principal directions of
    // the orbit error. Multiple pointings of a night supply one return.
    float probability[returns];int nights[returns],n=0;
    for(int j=0;j<m.count;j++){
        int id=m.view[3+j],night=fields[id].night;
        const Grid &g=grids[id];
        Position pos=forecast(m.step[3+j],a.x,a.y,v,features);
        float w=visible(g,depth,pos.x,pos.y,mag);
        if(pos.a+fabsf(pos.b)+pos.c>36.f){
            float h=1.58f;
            w=.36f*w+.16f*(visible(g,depth,pos.x+h*pos.a,pos.y+h*pos.b,mag)
                          +visible(g,depth,pos.x-h*pos.a,pos.y-h*pos.b,mag)
                          +visible(g,depth,pos.x,pos.y+h*pos.c,mag)
                          +visible(g,depth,pos.x,pos.y-h*pos.c,mag));
        }
        int k=0;while(k<n && nights[k]!=night)++k;
        if(k==n){nights[n]=night;probability[n++]=w;}
        else probability[k]=maximum(probability[k],w);
    }
    float none=1.f,one=0.f;
    for(int j=0;j<n;j++){
        float w=probability[j];
        one=one*(1.f-w)+none*w;none*=1.f-w;
    }
    return maximum(0.f,1.f-none-one);
}
NINLINE void extend(const Point *p,const int *edges,const Field *fields,const Model &m,
                    const Pair &pair,const Grid *grid,const float *depth,Triple *out){
    for(int j=0;j<alternatives;j++)out[j]={-1,-1,-1,-1.e26f};
    const Point &a=p[pair.a],&b=p[pair.b];
    const Field &A=fields[m.view[0]],&B=fields[m.view[1]],&C=fields[m.view[2]];
    float z[6],v[features];
    root_coordinates(m,a,z);z[2]=pair.u;z[3]=pair.v;
    Position pos=forecast(m.step[2],a.x,a.y,v,basis(z,4,v));
    float mag=.5f*(a.mag+b.mag),lev=.5f*(A.lever+B.lever);
    float variance=.05f+.25f*(sq(a.sigma)+sq(b.sigma))+sq(.28f*(lev-C.lever));
    Hit h[5];int n=return_hits<5>(p,edges,fields,C,pos,mag,variance,3.6f,h,m.family>=0);
    for(int j=0;j<n;j++){
        const Point &c=p[h[j].index];z[4]=h[j].u;z[5]=h[j].v;
        basis(z,6,v);
        float live=return_probability(m,a,v,grid,depth,(2.f*mag+c.mag)/3.f,fields);
        float score=.44f*a.quality+pair.merit+h[j].merit+1.28f*logf(.009f+live)
                   -.29f*log1pf(pos.a*pos.c*C.density);
        if(score<=out[alternatives-1].merit)continue;
        int k=alternatives-1;
        while(k>0 && score>out[k-1].merit){out[k]=out[k-1];--k;}
        out[k]={pair.a,pair.b,h[j].index,score};
    }
}
NINLINE void confirm(const Point *p,const int *edges,const Field *fields,const Model &m,
                     const Returns &r,const Grid *grid,const float *depth,const Triple &seed,Track *out){
    for(int j=0;j<4;j++){
        for(int k=0;k<5;k++)out[j].row[k]=-1;
        out[j].gamma=m.gamma;out[j].score=-1.e26f;
    }
    const Point &a=p[seed.a],&b=p[seed.b],&c=p[seed.c];
    const Field &A=fields[m.view[0]],&B=fields[m.view[1]],&C=fields[m.view[2]];
    float z[6],v[features];
    root_coordinates(m,a,z);
    auto second=forecast(m.step[1],a.x,a.y,v,basis(z,2,v));
    z[2]=(b.x-second.x)/second.a;z[3]=(b.y-second.y-second.b*z[2])/second.c;
    auto third=forecast(m.step[2],a.x,a.y,v,basis(z,4,v));
    z[4]=(c.x-third.x)/third.a;z[5]=(c.y-third.y-third.b*z[4])/third.c;
    basis(z,6,v);
    float logrange=0.f;
    for(int k=0;k<features;k++)logrange+=v[k]*r.logrange[k];
    float gamma=m.gamma*expf(minimum(.90f,maximum(-.90f,logrange)));
    float mag=(a.mag+b.mag+c.mag)/3.f,lev=(A.lever+B.lever+C.lever)/3.f;
    float var=.050f+(sq(a.sigma)+sq(b.sigma)+sq(c.sigma))/9.f;
    out[3].row[0]=a.row;out[3].row[1]=b.row;out[3].row[2]=c.row;
    out[3].score=seed.merit-.41f*log1pf(third.a*third.c);
    out[3].gamma=gamma;
    Position pred[possible]; int live[possible],n=0;
    struct ReturnHit {int view;Hit hit;float score;};
    ReturnHit measured[12];int nh=0;
    for(int j=0;j<r.count;j++){
        int id=r.view[j];const Field &D=fields[id];const Grid &g=grid[id];
        Position pos=forecast(r.step[j],a.x,a.y,v,features);
        float wx=3.9f*pos.a+70.f,wy=3.9f*(fabsf(pos.b)+pos.c)+70.f;
        if(pos.x+wx<g.x||pos.y+wy<g.y||pos.x-wx>g.x+270.f*g.width||
           pos.y-wy>g.y+270.f*g.height)continue;
        float chance=visible(g,depth,pos.x,pos.y,mag);
        if(chance<.005f && wx+wy<150.f && !D.pad)continue;
        pred[j]=pos;live[n++]=j;
        Hit hits[3];int count=return_hits<3>(p,edges,fields,D,pos,mag,var+sq(.28f*(lev-D.lever)),
                                             3.9f,hits,m.family>=0);
        for(int h=0;h<count;h++){
            float score=hits[h].merit-.51f*log1pf(pos.a*pos.c)+.24f*logf(.04f+chance);
            if(nh==12 && score<=measured[11].score)continue;
            int k=nh<12?nh++:11;
            while(k>0 && score>measured[k-1].score){measured[k]=measured[k-1];--k;}
            measured[k]={j,hits[h],score};
        }
    }
    if(nh){
        out[3].row[3]=p[measured[0].hit.index].row;
        out[3].score=seed.merit+measured[0].score-.22f*log1pf(third.a*third.c);
        const auto &h=measured[0];
        float shift=r.crossrange[h.view][0]*h.hit.u+r.crossrange[h.view][1]*h.hit.v;
        out[3].gamma=gamma*expf(minimum(.47f,maximum(-.47f,shift)));
    }
    for(int h=0;h<nh;h++){
        int d=measured[h].view;const Hit &hit=measured[h].hit;
        float shift=r.crossrange[d][0]*hit.u+r.crossrange[d][1]*hit.v;
        float updated=gamma*expf(minimum(.47f,maximum(-.47f,shift)));
        const Field &D=fields[r.view[d]];float mag4=.75f*mag+.25f*p[hit.index].mag;
        float lev4=.75f*lev+.25f*D.lever;
        for(int k=0;k<n;k++){
            int e=live[k];const Field &E=fields[r.view[e]];
            if(D.night==E.night)continue;
            Position pos=conditional(pred[e],r.corr[d][e],hit);
            Hit match[2];
            int nm=return_hits<2>(p,edges,fields,E,pos,mag4,.053f+sq(.29f*(lev4-E.lever)),4.05f,match,m.family>=0);
            for(int i=0;i<nm;i++){
                int ids[5]={seed.a,seed.b,seed.c,hit.index,match[i].index};
                float merit=seed.merit+measured[h].score+match[i].merit-.52f*log1pf(pos.a*pos.c);
                record(out,p,ids,updated,merit);
            }
        }
    }
}
}
#ifdef __CUDACC__
#define MEASURE_KERNEL extern "C" __global__
#define MEASURE_LOOP int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<count)
#else
#define MEASURE_KERNEL extern "C"
#define MEASURE_LOOP for(int i=0;i<count;i++)
#endif
using namespace measured;
#ifndef __CUDACC__
extern "C" void measured_layout(int *sizes){
    sizes[0]=sizeof(Pair);sizes[1]=sizeof(Triple);
    sizes[2]=sizeof(Grid);sizes[3]=sizeof(Returns);
}
#endif
MEASURE_KERNEL void observed_pairs(const Point *p,const int *edges,const Field *f,const Model *m,
                                   const int *roots,int count,Pair *out){
    MEASURE_LOOP partners_for(p,edges,f,*m,roots[i],out+partners*i);
}
MEASURE_KERNEL void observed_triples(const Point *p,const int *edges,const Field *f,const Model *m,
                                     const Pair *pairs,const Grid *g,const float *depth,
                                     int count,Triple *out){
    MEASURE_LOOP extend(p,edges,f,*m,pairs[i],g,depth,out+alternatives*i);
}
MEASURE_KERNEL void confirm_triples(const Point *p,const int *edges,const Field *f,const Model *m,
                                    const int *jobs,const Triple *s,int count,Track *out){
    MEASURE_LOOP search_one(p,edges,f,m[jobs[i]],s[i].a,out+4*i,s[i].b,s[i].c);
}
MEASURE_KERNEL void search_returns(const Point *p,const int *edges,const Field *f,const Model *m,
                                   const int *jobs,const Triple *s,const Returns *r,const Grid *g,
                                   const float *depth,int count,Track *out){
    MEASURE_LOOP confirm(p,edges,f,m[jobs[i]],r[jobs[i]],g,depth,s[i],out+4*i);
}
