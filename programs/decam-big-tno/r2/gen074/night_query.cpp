// Search independent shutter positions, conditioning each return on the first
// three measurements and on a measured position at another return.
#ifndef __CUDACC__
#include <cmath>
#include <algorithm>
#define NINLINE inline
#else
#define NINLINE __device__ inline
#endif

namespace night_query {
constexpr int returns = 10, views = 13, features = 28;
struct Step { float coef[features][2], L[3]; };
struct Model {
    int view[views], count, family;
    float gamma;
    Step step[views];
    float corr[returns][returns][4];
};
struct Point { float x,y,mag,sigma,quality; int row; };
struct Field { float xmin,inverse,lever,density; int start,columns,night,pad; };
struct Hit {int index; float u,v,merit;};
struct Track {int row[5]; float gamma,score;};
struct Position {float x,y,a,b,c;};
NINLINE float sq(float x){return x*x;}
NINLINE float maximum(float a,float b){return a>b?a:b;}
NINLINE float minimum(float a,float b){return a<b?a:b;}
NINLINE int bin(float x,const Field &f){
    return int(maximum(0.f, minimum(f.columns-1.f, (x-f.xmin)*f.inverse)));
}
NINLINE int basis(const float *coordinates,int n,float *v){
    v[0]=1.f;
    for(int j=0;j<n;j++)v[j+1]=coordinates[j];
    int k=n+1;
    for(int i=0;i<n;i++)for(int j=i;j<n;j++)v[k++]=coordinates[i]*coordinates[j];
    return k;
}
NINLINE Position forecast(const Step &s,float x,float y,const float *v,int n){
    for(int j=0;j<n;j++){x+=s.coef[j][0]*v[j];y+=s.coef[j][1]*v[j];}
    return {x,y,s.L[0],s.L[1],s.L[2]};
}

template<int N>
NINLINE int query(const Point *points,const int *edges,const Field &f,
                  const Position &p,float mag,float photwidth,float gate,Hit *best){
    if(f.columns<1 || !(p.a>.0001f && p.c>.0001f))return 0;
    float extent=gate*p.a;
    float xmax=f.xmin+f.columns/f.inverse;
    if(p.x+extent<f.xmin || p.x-extent>xmax)return 0;
    // A very large ellipse in a rich exposure is less informative than
    // the same area on an independently predicted second return.
    if(p.a*p.c*f.density>85.f)return 0;
    int lo=bin(p.x-extent,f),hi=bin(p.x+extent,f),n=0;
    float ell=gate*gate,penalty=.48f*log1pf(5.f*p.a*p.c*f.density);
    for(int col=lo;col<=hi;col++){
        float xlo=f.xmin+col/f.inverse;
        float u0=maximum(-gate,(xlo-p.x)/p.a),u1=minimum(gate,(xlo+1.f/f.inverse-p.x)/p.a);
        float y0=p.y+minimum(u0*p.b,u1*p.b)-gate*p.c;
        float y1=p.y+maximum(u0*p.b,u1*p.b)+gate*p.c;
        int l=edges[f.start+col],r=edges[f.start+col+1],end=r;
        while(l<r){int m=(l+r)>>1;if(points[m].y<y0)l=m+1;else r=m;}
        for(int j=l;j<end && points[j].y<=y1;j++){
            const Point &s=points[j];
            float dm=s.mag-mag;
            float w=photwidth+s.sigma*s.sigma;
            if(dm*dm>9.6f*w)continue;
            float u=(s.x-p.x)/p.a,v=(s.y-p.y-u*p.b)/p.c;
            float chi=u*u+v*v;
            if(chi>ell)continue;
            float score=.38f*s.quality-.43f*chi-.44f*dm*dm/w-penalty;
            int at=n;
            if(at>=N){if(score<=best[N-1].merit)continue;at=N-1;}
            while(at>0 && score>best[at-1].merit){best[at]=best[at-1];--at;}
            best[at]={j,u,v,score};if(n<N)++n;
        }
    }
    return n;
}

NINLINE Position conditional(const Position &p,const float *corr,const Hit &hit){
    // corr is E[z_first z_second^T]. The shared covariance of the two
    // forecasts can be almost singular on long extrapolations.
    float u=corr[0]*hit.u+corr[2]*hit.v;
    float v=corr[1]*hit.u+corr[3]*hit.v;
    float aa=maximum(.003f,1.f-sq(corr[0])-sq(corr[2]));
    float bb=-(corr[0]*corr[1]+corr[2]*corr[3]);
    float cc=maximum(.003f,1.f-sq(corr[1])-sq(corr[3]));
    float s=sqrtf(aa),h=bb/s;
    return {p.x+p.a*u,p.y+p.b*u+p.c*v,p.a*s,p.b*s+p.c*h,p.c*sqrtf(maximum(.003f,cc-h*h))};
}

template<int N>
NINLINE int return_hits(const Point *s,const int *edge,const Field *field,const Field &f,
                        const Position &p,float mag,float variance,float gate,Hit *hits,bool raw_allowed){
    int n=query<N>(s,edge,f,p,mag,variance,gate,hits);
    if(!raw_allowed||!f.pad)return n;
    const Field &raw=field[f.pad-1];
    if((n && hits[0].merit>-.6f)||p.a*p.c*raw.density>(n?.065f:.48f))return n;
    Hit other[N];
    int total=query<N>(s,edge,raw,p,mag,variance,gate,other);
    for(int i=0;i<total;i++){
        int at=n;
        bool same=false;
        for(int j=0;j<n;j++)if(s[hits[j].index].row==s[other[i].index].row)same=true;
        if(same)continue;
        if(at==N){if(other[i].merit<=hits[N-1].merit)continue;at=N-1;}
        while(at>0 && other[i].merit>hits[at-1].merit){hits[at]=hits[at-1];--at;}
        hits[at]=other[i];if(n<N)++n;
    }
    return n;
}
NINLINE void record(Track *out,const Point *s,const int *ids,float g,float score){
    if(score<=out[2].score)return;
    int replace=2;
    for(int j=0;j<3;j++){
        if(out[j].row[0]<0)continue;
        if(out[j].row[1]!=s[ids[1]].row || out[j].row[2]!=s[ids[2]].row)continue;
        int a=out[j].row[3],b=out[j].row[4],c=s[ids[3]].row,d=s[ids[4]].row;
        if((a==c && b==d) || (a==d && b==c)){
            if(out[j].score>=score)return;
            replace=j;break;
        }
    }
    while(replace>0 && score>out[replace-1].score){out[replace]=out[replace-1];--replace;}
    for(int i=0;i<5;i++)out[replace].row[i]=s[ids[i]].row;
    out[replace].gamma=g;out[replace].score=score;
}

NINLINE int measured_hit(const Point *s,int index,const Position &p,float mag,float variance,
                         float density,Hit *out){
    const Point &point=s[index];
    float u=(point.x-p.x)/p.a,v=(point.y-p.y-p.b*u)/p.c;
    float w=variance+sq(point.sigma);
    float merit=.38f*point.quality-.43f*(u*u+v*v)-.44f*sq(point.mag-mag)/w
                -.48f*log1pf(5.f*p.a*p.c*density);
    out[0]={index,u,v,merit};
    return 1;
}

NINLINE void search_one(const Point *s,const int *edge,const Field *field,const Model &m,
                        int root,Track *out,int known_b=-1,int known_c=-1){
    for(int j=0;j<4;j++){
        for(int k=0;k<5;k++)out[j].row[k]=-1;
        out[j].gamma=0.f;out[j].score=-1.e26f;
    }
    const Point &a=s[root];
    float z[6],v[features];
    const Step &init=m.step[0];
    z[0]=(a.x-init.coef[0][0])/init.L[0];
    z[1]=(a.y-init.coef[0][1]-z[0]*init.L[1])/init.L[2];
    if(sq(z[0])+sq(z[1])>24.f)return;
    int f1=basis(z,2,v);
    const Field &A=field[m.view[0]],&B=field[m.view[1]],&C=field[m.view[2]];
    Hit second[10],third[6],fourth[4],fifth[2];
    auto Bpos=forecast(m.step[1],a.x,a.y,v,f1);
    float wb=.047f+sq(a.sigma)+sq(.28f*(A.lever-B.lever));
    int nb=known_b<0?query<10>(s,edge,B,Bpos,a.mag,wb,3.45f,second):
                     measured_hit(s,known_b,Bpos,a.mag,wb,B.density,second);
    for(int ib=0;ib<nb;ib++){
        const Hit &b=second[ib];const Point &bp=s[b.index];
        float mag2=.5f*(a.mag+bp.mag),lev2=.5f*(A.lever+B.lever);
        z[2]=b.u;z[3]=b.v;
        auto Cpos=forecast(m.step[2],a.x,a.y,v,basis(z,4,v));
        float wc=.050f+.25f*(sq(a.sigma)+sq(bp.sigma))+sq(.28f*(lev2-C.lever));
        int nc=known_c<0?query<6>(s,edge,C,Cpos,mag2,wc,3.6f,third):
                         measured_hit(s,known_c,Cpos,mag2,wc,C.density,third);
        for(int ic=0;ic<nc;ic++){
            const Hit &c=third[ic];const Point &cp=s[c.index];
            z[4]=c.u;z[5]=c.v;
            int nf=basis(z,6,v);
            float mean=(2.f*mag2+cp.mag)/3.f,level=(2.f*lev2+C.lever)/3.f;
            float variance=.05f+(sq(a.sigma)+sq(bp.sigma)+sq(cp.sigma))/9.f;
            float triplet=.38f*a.quality+b.merit+c.merit-.54f*log1pf(Cpos.a*Cpos.c);
            if(out[3].row[3]==-1 && triplet>out[3].score){
                out[3].row[0]=a.row;out[3].row[1]=bp.row;out[3].row[2]=cp.row;
                out[3].gamma=m.gamma;out[3].score=triplet;
            }
            Position pred[returns];
            for(int j=0;j<m.count;j++)pred[j]=forecast(m.step[3+j],a.x,a.y,v,nf);
            for(int d=0;d<m.count;d++){
                const Field &D=field[m.view[3+d]];
                int nd=return_hits<4>(s,edge,field,D,pred[d],mean,variance+sq(.28f*(level-D.lever)),3.9f,fourth,m.family>=0);
                for(int id=0;id<nd;id++){
                    const Hit &hit=fourth[id];
                    float qscore=.38f*a.quality+b.merit+c.merit+hit.merit-
                        .62f*log1pf(pred[d].a*pred[d].c)-.22f*log1pf(Cpos.a*Cpos.c);
                    if(out[3].row[3]==-1||qscore>out[3].score){
                        out[3].row[0]=a.row;out[3].row[1]=bp.row;out[3].row[2]=cp.row;
                        out[3].row[3]=s[hit.index].row;out[3].gamma=m.gamma;out[3].score=qscore;
                    }
                    float mag4=.75f*mean+.25f*s[hit.index].mag,lev4=.75f*level+.25f*D.lever;
                    for(int e=0;e<m.count;e++){
                        const Field &E=field[m.view[3+e]];
                        if(d==e || D.night==E.night)continue;
                        // Either order can rescue a broad, crowded return.
                        // For two compact forecasts the lower index suffices.
                        if(e<d && pred[e].a*pred[e].c*E.density<.08f &&
                                  pred[d].a*pred[d].c*D.density<.08f)continue;
                        Position narrow=conditional(pred[e],m.corr[d][e],hit);
                        int ne=return_hits<2>(s,edge,field,E,narrow,mag4,.053f+sq(.29f*(lev4-E.lever)),4.05f,fifth,m.family>=0);
                        for(int ie=0;ie<ne;ie++){
                            int ids[5]={root,b.index,c.index,hit.index,fifth[ie].index};
                            float score=.38f*a.quality+b.merit+c.merit+hit.merit+fifth[ie].merit;
                            score-=.30f*log1pf(pred[d].a*pred[d].c)+.52f*log1pf(narrow.a*narrow.c);
                            record(out,s,ids,m.gamma,score);
                        }
                    }
                }
            }
        }
    }
}
}

#ifdef __CUDACC__
extern "C" __global__ void search_nights(const night_query::Point *p,const int *edges,
    const night_query::Field *field,const night_query::Model *models,const int *jobs,
    const int *roots,int count,night_query::Track *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)night_query::search_one(p,edges,field,models[jobs[i]],roots[i],out+4*i);
}
#else
extern "C" void night_layout(int *v){
    v[0]=sizeof(night_query::Point);v[1]=sizeof(night_query::Field);
    v[2]=sizeof(night_query::Model);v[3]=sizeof(night_query::Track);v[4]=4;
}
extern "C" void search_nights_host(const night_query::Point *p,const int *edges,
    const night_query::Field *field,const night_query::Model *models,const int *jobs,
    const int *roots,int count,night_query::Track *out){
    for(int i=0;i<count;i++)
        night_query::search_one(p,edges,field,models[jobs[i]],roots[i],out+4*i);
}
#endif
