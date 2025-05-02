#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream> // library for outputting to a txt

#define N       6250
#define BLOCK   1024

int guard_cells = 2;
int ref_cells   = 5;

// CPU function
void sliding_cpu(const unsigned int* in,
                 unsigned int*       out,
                 unsigned int        n,
                 int                 guard_cells,
                 int                 ref_cells)
{
    unsigned int limit = guard_cells + ref_cells;
    for (unsigned int i = 0; i < n; ++i) {
        // if too close to edges → zero
        if (i < limit || i >= n - limit) {
            out[i] = 0;
        } else {
            unsigned int sum = 0;
            // sum left reference cells (skip guard_cells next to center)
            for (int d = guard_cells + 1; d <= limit; ++d) {
                sum += in[i - d];
            }
            // sum right reference cells
            for (int d = guard_cells + 1; d <= limit; ++d) {
                sum += in[i + d];
            }
            out[i] = sum;
        }
    }
}

// GPU kernel
__global__ void sliding_gpu(const unsigned int* in,
                            unsigned int*       out,
                            unsigned int        n,
                            int                 guard_cells,
                            int                 ref_cells)
{
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int limit = guard_cells + ref_cells;

    if (idx >= limit && idx < n - limit) {
        unsigned int sum = 0;
        for (int d = guard_cells + 1; d <= limit; ++d) {
            sum += in[idx - d];
            sum += in[idx + d];
        }
        out[idx] = sum;
    }
    // else: leave out[idx] == 0  (you can cudaMemset before launch)
}

int main() {
    unsigned int bytes = N * sizeof(unsigned int);

    // host arrays
    auto *h_in      = new unsigned int[N];
    auto *h_out_cpu = new unsigned int[N];
    auto *h_out_gpu = new unsigned int[N];

    for (unsigned int i = 0; i < N; ++i)
        h_in[i] = i;

    // CPU timing
    auto t0 = std::chrono::high_resolution_clock::now();
    sliding_cpu(h_in, h_out_cpu, N, guard_cells, ref_cells);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU setup
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    // zero edges
    cudaMemset(d_out, 0, bytes);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int threads = BLOCK;
    // only indices [limit .. N-limit-1] do work
    unsigned int limit  = guard_cells + ref_cells;
    unsigned int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start);
    // note: pass guard_cells, ref_cells as extra kernel args
    sliding_gpu<<<blocks, threads>>>(d_in, d_out, N, guard_cells, ref_cells);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Copy back & validate
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    bool match = true;
    for (unsigned int i = 0; i < N; ++i)
        if (h_out_cpu[i] != h_out_gpu[i]) { match = false; break; }

    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "Match? "    << (match ? "YES\n" : "NO\n");

    // print first few
    std::cout << "\nIndex : CPU vs GPU\n";
    for (unsigned int i = 0; i < 10; ++i)
        std::cout << i << " : " << h_out_cpu[i]
                  << " vs " << h_out_gpu[i] << "\n";
                  
    std::ofstream outFile("results.txt");
    if (!outFile) {
        std::cerr << "Error opening results.txt for writing\n";
    } else {
        // header
        outFile << "Index,CPU result,GPU result\n";
        // data rows
        for (unsigned int i = 0; i < N; ++i) {
            outFile << i
                    << ',' << h_out_cpu[i]
                    << ',' << h_out_gpu[i]
                    << '\n';
        }
        outFile.close();
        std::cout << "Results written to results.txt\n";
    }

    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    return 0;
}
