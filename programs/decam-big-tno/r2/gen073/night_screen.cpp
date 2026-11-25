#ifndef __CUDACC__
#include <cmath>
#endif
#include "quartet_fit.hpp"

namespace night_screen {
struct Sample {float xy[5][2];int exposure[5];float gamma;};
struct Result {float gamma,rms;};
QINLINE Result evaluate(const Sample &s,const quartet_fit::Shutter *vis){
    quartet_fit::Shutter e[5];
    double xy[2][5];
    for(int j=0;j<5;j++){
        e[j]=vis[s.exposure[j]];xy[0][j]=s.xy[j][0];xy[1][j]=s.xy[j][1];
    }
    quartet_fit::State o;
    double rms=1.e8;
    if(!quartet_fit::positions(xy,e,s.gamma,o,rms))return {0.f,1.e8f};
    return {float(o.g),float(rms)};
}
}
#ifdef __CUDACC__
extern "C" __global__ void fit_five_positions(const night_screen::Sample *s,
     const quartet_fit::Shutter *vis,int n,night_screen::Result *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=night_screen::evaluate(s[i],vis);
}
#else
extern "C" void fit_five_positions_host(const night_screen::Sample *s,
     const quartet_fit::Shutter *vis,int n,night_screen::Result *out){
    for(int i=0;i<n;i++)out[i]=night_screen::evaluate(s[i],vis);
}
#endif
