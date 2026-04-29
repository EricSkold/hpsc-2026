#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h> 

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  float j_arr[N]; 

  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j_arr[i] = i; 
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 jvec = _mm512_load_ps(j_arr);
  
  __m512 zero = _mm512_setzero_ps(); 

  for(int i=0; i<N; i++) {
   
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 ivec = _mm512_set1_ps(i);

    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);

    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512 r2 = _mm512_add_ps(rx2, ry2);

    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    
    __m512 inv_r2 = _mm512_mul_ps(inv_r, inv_r);
    __m512 inv_r3 = _mm512_mul_ps(inv_r2, inv_r);

    __m512 fx_upd = _mm512_mul_ps(rx, _mm512_mul_ps(mvec, inv_r3));
    __m512 fy_upd = _mm512_mul_ps(ry, _mm512_mul_ps(mvec, inv_r3));

    __mmask16 mask = _mm512_cmp_ps_mask(ivec, jvec, _CMP_NEQ_OQ);

    fx_upd = _mm512_mask_blend_ps(mask, zero, fx_upd);
    fy_upd = _mm512_mask_blend_ps(mask, zero, fy_upd);

    fx[i] -= _mm512_reduce_add_ps(fx_upd);
    fy[i] -= _mm512_reduce_add_ps(fy_upd);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
