#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    bucket[i] = 0;
  }
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void scan(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  for (int j=1; j<N; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    if (i>=j) a[i] += b[i-j];
    __syncthreads();
  }
}

__global__ void fill_array(int *key, int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int start = (i == 0) ? 0 : bucket[i - 1];
    int end = bucket[i];
    for (int j = start; j < end; j++) {
      key[j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int M = 64;

  int *key;
  int *bucket;
  int *temp;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&temp, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init_bucket<<<(range+M-1)/M, M>>>(bucket, range);
  cudaDeviceSynchronize();

  count_bucket<<<(n+M-1)/M, M>>>(key, bucket, n);
  cudaDeviceSynchronize();

  scan<<<1, range>>>(bucket, temp, range);
  cudaDeviceSynchronize();

  fill_array<<<(range+M-1)/M, M>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(temp);

  return 0;
}
