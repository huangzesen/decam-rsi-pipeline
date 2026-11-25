// Shared scalar and CUDA work items for the measured shutter graph.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define SHUTTER_GPU
#define FN __device__ __forceinline__
#else
#include <cmath>
#define FN inline
#endif
namespace shutters {
constexpr double arc=206264.80624709636;
constexpr int width=32;
struct Visit {double t,ox,oy,oz,depth,lev;int night,pad;};
struct Pixel {
    float x,y,mag,q,color,err,peer_t,peer_x,peer_y,peer_ox,peer_oy,peer_err;
    int row,e;
};
struct Grid {int bx,by,nx,ny,start;float cell,density,pad;};
struct Edge {int other;float score,g,sg;};
struct Triple {int row[3];float g,sg,h,sh,score;};
struct Seed {int row[5];float score,g;Triple tri;};
static_assert(sizeof(Pixel)==56 && sizeof(Visit)==56 && sizeof(Grid)==32 &&
              sizeof(Edge)==16 && sizeof(Triple)==32 && sizeof(Seed)==60,"shutter catalogue layout");
struct V {
    double x,y;
    FN V operator+(V z)const{return {x+z.x,y+z.y};}
    FN V operator-(V z)const{return {x-z.x,y-z.y};}
    FN V operator*(double z)const{return {x*z,y*z};}
    FN double dot(V z)const{return x*z.x+y*z.y;}
};
FN double sqr(double z){return z*z;}
FN double clamp(double x,double a,double b){return fmax(a,fmin(b,x));}
FN V sky(const Pixel&p){return {p.x,p.y};}
FN void keep(Edge &a,Edge &b,Edge q){
    if(q.score>a.score){b=a;a=q;}else if(q.score>b.score)b=q;
}
FN void burst(const Pixel &p,V disp,V obs,double dt,double g,double sg,double &aa,double &ay,double &chi){
    if(p.peer_err<=0.)return;
    double f=p.peer_t/dt;
    V o=V{p.peer_ox,p.peer_oy}-obs*f;
    V delta=V{p.peer_x,p.peer_y}-disp*f+o*g;
    double s=1./fmax(.12,double(p.peer_err));
    V x=o*(s*sg),y=delta*s;
    aa+=x.dot(x);ay-=x.dot(y);chi+=y.dot(y);
}
FN void edges(int root,int e,int family,V east,const Pixel*points,const Visit*vis,const Grid*grid,
              const int*lookup,Edge *out){
    Edge first{-1,-1.e18f,0,0},second=first;out[0]=first;out[1]=second;
    const Pixel&a=points[root];const Visit&A=vis[a.e],&B=vis[e];const Grid&im=grid[e];
    if(!im.nx||A.night==B.night||B.depth+.8<a.mag+a.color*B.lev)return;
    double radius=family==0?44.:family==1?39.:family==2?85.:47.;
    double sw=family==0?.11:family==1?.235:family==2?.49:.24;
    double vs=family==0?.10:family==1?.236:family==2?.54:.64,ws=family==0?.14:family==1?.39:family==2?.73:.69;
    V home=sky(a),north{-east.y,east.x};
    double dt=B.t-A.t;if(fabs(dt)<.5)return;
    double norm=sqrt(1.+home.dot(home)/(arc*arc)),g=norm/radius,sg=g*sw;
    double sp=.01720209895*arc*norm/(radius*sqrt(radius)),den=1.-g*B.oz;
    V delta=(V{B.ox-A.ox,B.oy-A.oy}*arc-home*(B.oz-A.oz));
    double prograde=family==3?.07:1.;
    V velo=east*(sp*dt*prograde),change=(velo-delta*g)*(1./den);
    V mean=home+change,center=mean,der=(velo*(1.5/g)-delta+change*B.oz)*(sg/den);
    double su=sp*dt*vs/den,sv=sp*dt*ws/den;
    double su2=su*su+.23*.23,sv2=sv*sv+.23*.23;
    double xx=sqr(der.x)+su2*sqr(east.x)+sv2*sqr(east.y);
    double yy=sqr(der.y)+su2*sqr(east.y)+sv2*sqr(east.x);
    double xy=der.x*der.y+(su2-sv2)*east.x*east.y;
    // Condition the entrance aperture on the actual displacement in the
    // anchor visit, retaining its correlation with distance and velocity.
    if(a.peer_err>0. && fabs(a.peer_t)>.003){
        double tau=a.peer_t,near_den=1.-g*A.oz;
        V obs={a.peer_ox,a.peer_oy},v=east*(sp*tau*prograde);
        V m=(v-obs*g)*(1./near_den),gd=(v*(1.5/g)-obs+m*A.oz)*(sg/near_den);
        V ue=east*(sp*tau*vs/near_den),un=north*(sp*tau*ws/near_den);
        double noise=sqr(fmax(.15,1.18*a.peer_err));
        double ax=gd.x*gd.x+ue.x*ue.x+un.x*un.x+noise;
        double ay=gd.y*gd.y+ue.y*ue.y+un.y*un.y+noise;
        double t=gd.x*gd.y+ue.x*ue.y+un.x*un.y,d=ax*ay-t*t;
        V de=east*su,dn=north*sv;
        V cx=gd*der.x+ue*de.x+un*dn.x,cy=gd*der.y+ue*de.y+un*dn.y;
        V wx={(ay*cx.x-t*cx.y)/d,(ax*cx.y-t*cx.x)/d};
        V wy={(ay*cy.x-t*cy.y)/d,(ax*cy.y-t*cy.x)/d};
        V shift=V{a.peer_x,a.peer_y}-m;
        center=center+V{wx.dot(shift),wy.dot(shift)};
        xx-=wx.dot(cx);yy-=wy.dot(cy);xy-=wx.dot(cy);
    }
    double var=fmax(.00001,xx-xy*xy/yy),det=var*yy;
    double height=3.1*sqrt(yy),half=3.1*sqrt(var);
    if(center.y+height<im.by*im.cell||center.y-height>(im.by+im.ny)*im.cell||
       center.x+3.1*sqrt(xx)<im.bx*im.cell||center.x-3.1*sqrt(xx)>(im.bx+im.nx)*im.cell)return;
    int l=(int)fmax(0.,floor((center.y-height)/im.cell)-im.by);
    int h=(int)fmin(im.ny-1.,floor((center.y+height)/im.cell)-im.by);
    double x1=der.dot(east)/sqrt(su2),x2=der.dot(north)/sqrt(sv2);
    double info=1.+x1*x1+x2*x2;
    double expected=a.mag+a.color*B.lev;
    double fg=-.54*log(.014+sqrt(det)*im.density)+.37*a.q;
    for(int y=l;y<=h;y++){
        double y0=fmax(-height,(y+im.by)*im.cell-center.y),y1=fmin(height,(y+im.by+1)*im.cell-center.y);
        double x0=center.x+fmin(y0*xy/yy,y1*xy/yy)-half;
        double xh=center.x+fmax(y0*xy/yy,y1*xy/yy)+half;
        int u=(int)fmax(0.,floor(x0/im.cell)-im.bx),v=(int)fmin(im.nx-1.,floor(xh/im.cell)-im.bx);
        if(u>v)continue;
        int base=im.start+im.nx*y;
        for(int k=lookup[base+u];k<lookup[base+v+1];k++){
            const auto &b=points[k];
            V offset=sky(b)-center;
            double geo=sqr(offset.x-offset.y*xy/yy)/var+sqr(offset.y)/yy;
            if(geo>9.61)continue;
            double m2=b.mag+b.color*B.lev;
            double sm=sqrt(sqr(a.err)+sqr(b.err)+.155*.155+
                           .045*sqr(A.lev-B.lev)*(a.peer_err<=0. && b.peer_err<=0.));
            double phot=sqr((m2-expected)/sm);
            if(phot>13.)continue;
            V motion=sky(b)-home;
            V initial=sky(b)-mean;
            double q1=initial.dot(east)/sqrt(su2),q2=initial.dot(north)/sqrt(sv2);
            double aa=info,ay=q1*x1+q2*x2,chi=q1*q1+q2*q2;
            burst(a,motion,delta,dt,g,sg,aa,ay,chi);
            burst(b,motion,delta,dt,g,sg,aa,ay,chi);
            double step=ay/aa;chi=fmax(0.,chi-step*ay);
            if(chi>22.||fabs(step)>3.25)continue;
            double rank=fg+.65*b.q-.49*chi-.46*phot-log(sm)-.5*log(aa/info);
            keep(first,second,{k,float(rank),float(g+sg*step),float(sg/sqrt(aa))});
        }
    }
    out[0]=first;out[1]=second;
}

FN void preview(int root,int slot,int family,V east,const Pixel*p,const Visit*v,const Grid*m,
                const int*lookup,const int*schedule,Edge*out){
    const auto &a=p[root];const auto &im=m[a.e];
    int tx=int(clamp(floor(3.*(a.x/im.cell-im.bx)/im.nx),0.,2.));
    int ty=int(clamp(floor(3.*(a.y/im.cell-im.by)/im.ny),0.,2.));
    int e=schedule[24*(9*a.e+3*ty+tx)+slot];
    Edge z[2]={{-1,-1.e18f,0,0},{-1,-1.e18f,0,0}};
    if(e>=0)edges(root,e,family,east,p,v,m,lookup,z);
    out[slot]=z[0];
    if(slot<8)out[24+slot]=z[1];
}
struct Orbit {
    V p,v;double g,h,k,rd,beta;
    FN void fg(double t,double &f,double &s)const{
        double w=rd*t,z=k*t*t,b=beta*t*t;
        f=1.-.5*z+.5*z*w+z*(3.*b-2.*z-15.*w*w)/24.+z*w*(30.*z-45.*b+105.*w*w)/120.;
        s=t*(1.-z/6.+z*w/4.+z*(9.*b-8.*z-45.*w*w)/120.+z*w*(150.*z-180.*b+420.*w*w)/720.);
    }
    FN void set(const Pixel &a,const Pixel &b,const Visit &A,const Visit &B,double g0,double h0){
        g=g0;h=h0;
        p=sky(a)*((1.-g*A.oz)/arc)+V{A.ox,A.oy}*g;
        double l=1.+p.dot(p),t=B.t-A.t,f=1.,s=t;
        k=.0002963071*g*g*g/(l*sqrt(l));
        for(int i=0;i<3;i++){
            v=(sky(b)*((f+s*h-g*B.oz)/arc)+V{B.ox,B.oy}*g-p*f)*(1./s);
            rd=(h+p.dot(v))/l;beta=(h*h+v.dot(v))/l;
            fg(t,f,s);
        }
    }
    FN V at(double dt,const Visit&e)const{
        double f,s;fg(dt,f,s);return (p*f+v*s-V{e.ox,e.oy}*g)*(arc/(f+s*h-g*e.oz));
    }
};
struct Band {double u,v,aa,ab,bb,ax,bx,chi,score;int row,night;};
FN void intersect(int root,int j,const Pixel *p,const Visit *vis,const Edge *ed,Seed *out){
    const auto &edge=ed[j];out->score=-1.e18f;out->g=0.;
    out->tri=Triple{{-1,-1,-1},0,0,0,0,-1.e18f};
    for(int k=0;k<5;k++)out->row[k]=-1;
    if(edge.other<0||!(edge.g>.00045&&edge.g<.071))return;
    const auto &a=p[root],&b=p[edge.other];
    const auto &A=vis[a.e],&B=vis[b.e];
    Orbit orb,range,radial;
    double k=.01720209895*edge.g*sqrt(edge.g);
    double dt=B.t-A.t,g=edge.g,sg=fmax(.0035*g,double(edge.sg)),sh=.22*k;
    orb.set(a,b,A,B,g,0.);
    double hm=-orb.p.dot(orb.v)/(1.+orb.p.dot(orb.p));
    orb.set(a,b,A,B,g,hm);range.set(a,b,A,B,g+.025*sg,hm);radial.set(a,b,A,B,g,hm+.025*sh);
    Band band[width];int count=0;
    for(int m=0;m<width;m++){
        const auto &c=ed[m];if(c.other<0||m==j)continue;
        const auto &v=p[c.other];const auto&E=vis[v.e];
        if(E.night==B.night)continue;
        double t=E.t-A.t,f=t/dt;
        V pos=orb.at(t,E),du=(range.at(t,E)-pos)*40.,dv=(radial.at(t,E)-pos)*40.;
        double nl=.008*sqrt(du.dot(du))*(fmin(2.,fabs(t)*sh*35.+t*t*orb.k*sg/g*14.));
        double s2=.26*.26+.065*(sqr(f)+sqr(1.-f))+sqr(fmin(25.,nl));
        V z=sky(v)-pos;
        if(fabs(z.x)>3.4*(fabs(du.x)+fabs(dv.x)+sqrt(s2))||
           fabs(z.y)>3.4*(fabs(du.y)+fabs(dv.y)+sqrt(s2)))continue;
        double aa=du.dot(du)/s2,ab=du.dot(dv)/s2,bb=dv.dot(dv)/s2;
        double ax=du.dot(z)/s2,bx=dv.dot(z)/s2,rr=z.dot(z)/s2;
        double det=(1.+aa)*(1.+bb)-ab*ab;
        double u=(ax*(1.+bb)-ab*bx)/det,w=(bx*(1.+aa)-ab*ax)/det;
        double chi=fmax(0.,rr-u*ax-w*bx);
        if(chi>19.||fabs(u)>3.4||fabs(w)>3.4)continue;
        band[count++]={u,w,aa,ab,bb,ax,bx,rr,c.score-.20*log(det),c.other,E.night};
        double mark=.64*edge.score+.84*c.score-.5*chi-.21*log(det);
        if(mark>out->tri.score)out->tri={{a.row,b.row,v.row},float(g),float(sg),float(hm),float(sh),float(mark)};
    }
    for(int i=0;i<count;i++)for(int l=0;l<i;l++){
        const auto &x=band[i],&y=band[l];if(x.night==y.night)continue;
        double aa=1.+x.aa+y.aa,bb=1.+x.bb+y.bb,ab=x.ab+y.ab,det=aa*bb-ab*ab;
        double ax=x.ax+y.ax,bx=x.bx+y.bx,u=(bb*ax-ab*bx)/det,v=(aa*bx-ab*ax)/det;
        double chi=fmax(0.,x.chi+y.chi-u*ax-v*bx);
        if(chi>20.||fabs(u)>3.5||fabs(v)>3.4)continue;
        double gain=0.;int witness=-1;
        for(int m=0;m<count;m++){
            const auto &z=band[m];if(z.night==x.night||z.night==y.night)continue;
            double az=aa+z.aa,bz=bb+z.bb,cz=ab+z.ab,dz=az*bz-cz*cz;
            double rx=ax+z.ax,ry=bx+z.bx,ux=(bz*rx-cz*ry)/dz,uy=(az*ry-cz*rx)/dz;
            double extra=fmax(0.,x.chi+y.chi+z.chi-ux*rx-uy*ry-chi);
            if(extra>14.)continue;
            double power=5.+.33*z.score-.46*extra-.15*log(dz/det);
            if(power>gain){gain=power;witness=z.row;}
        }
        double score=.37*edge.score+.47*(x.score+y.score)-.49*chi+gain;
        if(score>out->score){
            out->score=score;out->g=g+u*sg;
            out->row[0]=a.row;out->row[1]=b.row;out->row[2]=p[x.row].row;
            out->row[3]=p[y.row].row;out->row[4]=witness>=0?p[witness].row:-1;
        }
    }
}
}

#ifdef SHUTTER_GPU
extern "C" __global__ void shutter_preview(const shutters::Pixel*p,const shutters::Visit*v,const shutters::Grid*m,
         const int*lookup,const int*roots,const int*schedule,int n,int family,const double*east,shutters::Edge*out){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<24*n)shutters::preview(roots[t/24],t%24,family,{east[0],east[1]},p,v,m,lookup,schedule,out+32*(t/24));
}
extern "C" __global__ void shutter_edges(const shutters::Pixel*p,const shutters::Visit*v,const shutters::Grid*m,
         const int *look,const int*anchors,int n,int ne,int family,const double*east,shutters::Edge*out){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<n*ne)shutters::edges(anchors[t/ne],t%ne,family,{east[0],east[1]},p,v,m,look,out+2*t);
}
extern "C" __global__ void shutter_intersect(const shutters::Pixel*p,const shutters::Visit*v,const int*roots,
            const shutters::Edge*e,int n,shutters::Seed*out){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<n*shutters::width)shutters::intersect(roots[t/shutters::width],t%shutters::width,p,v,
                                              e+(t/shutters::width)*shutters::width,out+t);
}
#else
extern "C" void shutter_preview(const shutters::Pixel*p,const shutters::Visit*v,const shutters::Grid*m,
         const int*lookup,const int*roots,const int*schedule,int n,int family,double x,double y,shutters::Edge*out){
    for(int t=0;t<24*n;t++)shutters::preview(roots[t/24],t%24,family,{x,y},p,v,m,lookup,schedule,out+32*(t/24));
}
extern "C" void shutter_edges(const shutters::Pixel*p,const shutters::Visit*v,const shutters::Grid*m,
         const int *look,const int*anchors,int n,int ne,int family,double x,double y,shutters::Edge*out){
    for(int t=0;t<n*ne;t++)shutters::edges(anchors[t/ne],t%ne,family,{x,y},p,v,m,look,out+2*t);
}
extern "C" void shutter_intersect(const shutters::Pixel*p,const shutters::Visit*v,const int*roots,
            const shutters::Edge*e,int n,shutters::Seed*out){
    for(int t=0;t<n*shutters::width;t++)shutters::intersect(roots[t/shutters::width],t%shutters::width,p,v,
                                                           e+(t/shutters::width)*shutters::width,out+t);
}
#endif
