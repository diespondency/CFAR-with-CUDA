#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

#define THREADS_PER_BLOCK 481

int  guard_cells = 50;
int  ref_cells   = 100;
float bias       = -0.015f;

// CPU implementation
void sliding_cpu(const float* in,
                 float*       out,
                 unsigned int n,
                 int          guard_cells,
                 int          ref_cells,
                 float        bias)
{
    unsigned int limit = guard_cells + ref_cells;
    for (unsigned int i = 0; i < n; ++i) {
        if (i < limit || i >= n - limit) {
            out[i] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int d = guard_cells + 1; d <= limit; ++d)
                sum += in[i - d];
            for (int d = guard_cells + 1; d <= limit; ++d)
                sum += in[i + d];
            // subtract bias and normalize
            out[i] = ((sum) / (2.0f * ref_cells)) - bias;
        }
    }
}

// Original GPU kernel
__global__ void sliding_gpu(const float*  in,
                            float*        out,
                            unsigned int  n,
                            int           guard_cells,
                            int           ref_cells,
                            float         bias)
{
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int limit = guard_cells + ref_cells;

    if (idx >= limit && idx < n - limit) {
        float sum = 0.0f;
        
        for (int d = guard_cells + 1; d <= limit; ++d) {
            sum += in[idx - d];
            sum += in[idx + d];
        }
        out[idx] = ((sum) / (2.0f * ref_cells)) - bias;
    }
}

// Shared memory optimized GPU kernel
__global__ void sliding_gpu_shared(const float* in,
                                   float* out,
                                   unsigned int n,
                                   int guard_cells,
                                   int ref_cells,
                                   float bias)
{
    extern __shared__ float s_data[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int limit = guard_cells + ref_cells;
    
    // Initialize shared memory with zeros
    s_data[tid] = 0.0f;
    if (tid < 2 * limit) {
        s_data[blockDim.x + tid] = 0.0f;
    }
    __syncthreads();
    
    // Load data into shared memory with padding
    // Each thread loads its own element
    if (idx < n) {
        s_data[limit + tid] = in[idx];
        
        // Some threads also load halo elements
        if (tid < limit) {
            // Left halo
            int global_idx = idx - limit;
            if (global_idx >= 0) {
                s_data[tid] = in[global_idx];
            }
        }
        
        // Right halo - handle last threads in block
        if (tid >= (blockDim.x - limit) && tid < blockDim.x) {
            int global_idx = idx + limit;
            if (global_idx < n) {
                s_data[tid + 2*limit] = in[global_idx];
            }
        }
    }
    
    __syncthreads();
    
    // Compute CFAR
    if (idx >= limit && idx < n - limit) {
        float sum = 0.0f;
        int shared_idx = tid + limit;  // Offset in shared memory
        
        // Sum reference cells (using shared memory)
        for (int d = guard_cells + 1; d <= limit; ++d) {
            sum += s_data[shared_idx - d];
            sum += s_data[shared_idx + d];
        }
        
        out[idx] = ((sum) / (2.0f * ref_cells)) - bias;
    } else if (idx < n) {
        out[idx] = 0.0f; // Zero out the guard cell regions
    }
}

int main() {
    // Read data.txt → voltages vector (the captured data)
    std::ifstream inFile("data_clean.txt");
    if (!inFile) {
        std::cerr << "Error: could not open data_clean.txt\n";
        return 1;
    }
    std::vector<float> voltages;
    std::string        line;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        float t, v;
        char  comma;
        if (!(ss >> t >> comma >> v) || comma != ',') {
            std::cerr << "Warning: skipping malformed line: " << line << "\n";
            continue;
        }
        voltages.push_back(v);
    }
    inFile.close();

    unsigned int N = static_cast<unsigned int>(voltages.size());
    if (N == 0) {
        std::cerr << "No data loaded!\n";
        return 1;
    }
    
    std::cout << "Array size: " << N << " elements\n";

    // Create data arrays
    float *h_in      = voltages.data();
    float *h_out_cpu = new float[N];
    float *h_out_gpu = new float[N];
    float *h_out_shared = new float[N];

    // CPU timing + compute
    auto t0 = std::chrono::high_resolution_clock::now();
    sliding_cpu(h_in, h_out_cpu, N, guard_cells, ref_cells, bias);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    // GPU setup
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    unsigned int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // --------------- Original GPU kernel timing (1000 runs) ---------------
    cudaMemset(d_out, 0, bytes);
    
    // Warmup run
    sliding_gpu<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        sliding_gpu<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 1000.0f; // Average time per run
    
    // Copy results back
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // --------------- Shared Memory GPU kernel timing (1000 runs) ---------------
    cudaMemset(d_out, 0, bytes);
    
    // Calculate shared memory size
    unsigned int limit = guard_cells + ref_cells;
    size_t shared_mem_size = (THREADS_PER_BLOCK + 2 * limit) * sizeof(float);
    
    // Warmup run with shared memory
    sliding_gpu_shared<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        sliding_gpu_shared<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_gpu_ms = 0.0f;
    cudaEventElapsedTime(&shared_gpu_ms, start, stop);
    shared_gpu_ms /= 1000.0f; // Average time per run
    
    // Copy shared memory results back
    cudaMemcpy(h_out_shared, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Check results against CPU implementation
    bool standard_match = true;
    bool shared_match = true;
    
    for (unsigned int i = 0; i < N; ++i) {
        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-5f) {
            standard_match = false;
            break;
        }
        if (fabs(h_out_cpu[i] - h_out_shared[i]) > 1e-5f) {
            shared_match = false;
            break;
        }
    }

    // Print timing results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "----------------------------------------\n";
    std::cout << "Performance Results (1000 kernel runs):\n";
    std::cout << "----------------------------------------\n";
    std::cout << "CPU time:                " << cpu_ms << " ms\n";
    std::cout << "Standard GPU time:       " << gpu_ms << " ms\n";
    std::cout << "Shared Memory GPU time:  " << shared_gpu_ms << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Speedup (CPU to GPU):           " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "Speedup (Standard to Shared):   " << (gpu_ms / shared_gpu_ms) << "x\n";
    std::cout << "Speedup (CPU to Shared):        " << (cpu_ms / shared_gpu_ms) << "x\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Standard GPU results match CPU? " << (standard_match ? "YES" : "NO") << "\n";
    std::cout << "Shared GPU results match CPU?   " << (shared_match ? "YES" : "NO") << "\n";
    std::cout << "----------------------------------------\n";

    // Print first few results for verification
    // std::cout << "\nFirst 10 results (index : CPU vs Standard GPU vs Shared GPU):\n";
    // for (unsigned int i = 0; i < std::min<unsigned int>(10, N); ++i) {
    //     std::cout << i << " : "
    //               << h_out_cpu[i] << " vs "
    //               << h_out_gpu[i] << " vs "
    //               << h_out_shared[i] << "\n";
    // }

    // Write results to CSV file
    // std::ofstream resultsFile("shared_memory_comparison.csv");
    // if (resultsFile.is_open()) {
    //     resultsFile << "Implementation,AverageTime(ms),SpeedupVsCPU\n";
    //     resultsFile << "CPU," << cpu_ms << ",1.0\n";
    //     resultsFile << "StandardGPU," << gpu_ms << "," << (cpu_ms / gpu_ms) << "\n";
    //     resultsFile << "SharedMemoryGPU," << shared_gpu_ms << "," << (cpu_ms / shared_gpu_ms) << "\n";
    //     resultsFile.close();
    //     std::cout << "\nPerformance comparison saved to shared_memory_comparison.csv\n";
    // }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    delete[] h_out_shared;
    
    return 0;
}
