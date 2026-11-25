#ifdef __CUDACC__
#define LOCAL __device__ __forceinline__
#define ENTRY extern "C" __global__
#else
#include <cmath>
#define LOCAL inline
#define ENTRY extern "C"
#endif

LOCAL int first_y(const int *ys, int lo, int hi, int value) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (ys[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Columns are contiguous slices of an exposure sorted by y cell. The
// likelihood and final distance use the original double-precision positions.
ENTRY void exposure_nearest(const long long *ex, const double *qpos,
        const double *bounds, const int *directory, const int *offset,
        const int *ys, const int *row, const double *pos, const double *origin,
        long long *result, int ne, int n, double pitch, double tolerance) {
#ifdef __CUDACC__
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#else
    int begin = 0, stride = 1;
#endif
    for (int i = begin; i < n; i += stride) {
        const long long e = ex[i];
        result[i] = -1;
        if (e < 0 || e >= ne) continue;
        const double x = qpos[2LL*i], y = qpos[2LL*i+1];
        const double *box = bounds + 4*e;
        if (!(x >= box[0]-tolerance && x <= box[2]+tolerance &&
              y >= box[1]-tolerance && y <= box[3]+tolerance)) continue;
        int x0 = int(floor((x-origin[0]-tolerance)/pitch));
        int x1 = int(floor((x-origin[0]+tolerance)/pitch));
        int y0 = int(floor((y-origin[1]-tolerance)/pitch));
        int y1 = int(floor((y-origin[1]+tolerance)/pitch));
        const int *v = directory + 3*e;
        if (x0 < v[0]) x0 = v[0];
        if (x1 >= v[1]) x1 = v[1]-1;
        double best = tolerance*tolerance;
        int selected = -1;
        for (int cx=x0; cx<=x1; ++cx) {
            int j = v[2]+cx-v[0];
            int stop = offset[j+1];
            int k = first_y(ys,offset[j],stop,y0);
            for (; k<stop && ys[k]<=y1; ++k) {
                const double dx=pos[2LL*k]-x, dy=pos[2LL*k+1]-y;
                double d2=dx*dx+dy*dy;
                if (d2 < best || (d2 == best && (selected < 0 || row[k] < selected))) {
                    selected=row[k]; best=d2;
                }
            }
        }
        result[i]=selected;
    }
}
