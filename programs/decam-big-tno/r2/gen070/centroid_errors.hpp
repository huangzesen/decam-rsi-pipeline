// A partially masked or blended point source has a wider centroid tail.
struct CentroidError {
    double narrow,broad,tail;
    double posterior(double r2)const{
        double a=std::log((1.-tail)/narrow)-.5*r2/narrow;
        double b=std::log(tail/broad)-.5*r2/broad;
        return 1./(1.+std::exp(std::clamp(a-b,-50.,50.)));
    }
};
CentroidError centroid(const original::Sky &sky,int r,double dm){
    int e=sky.ex[r];
    double seeing=std::clamp(sky.seeing[e],.55,2.65),sn=std::max(4.5,double(sky.sn[r]));
    double shape=std::abs(sky.sharp[r]-1.04)/std::hypot(.20,3.5/sn);
    double p=.035+.10*bool(sky.flag[r]&2)+.045*bool(sky.flag[r]&4)+
             .055*std::clamp(shape-1.,0.,2.5)+.12*std::clamp(-dm-.22,0.,1.1);
    if(sky.population && sky.population[r]==0)p+=.045;
    double narrow=sq(.10)+sq(std::max(.92,.76*seeing)/sn);
    return {narrow,narrow+sq(.34*seeing),std::min(.42,p)};
}
