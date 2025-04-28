#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK   1024

int  guard_cells = 50;
int  ref_cells   = 100;
float bias       = -0.015f;      // now a float

// CPU function (now floats)
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

// GPU kernel (now floats, bias passed in)
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

int main() {
    // --- 1) Read data.txt → voltages vector ---
    std::ifstream inFile("data_clean.txt");
    if (!inFile) {
        std::cerr << "Error: could not open data.txt\n";
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

    // --- 2) Allocate host arrays ---
    float *h_in      = voltages.data();          // input points at our vector
    float *h_out_cpu = new float[N];
    float *h_out_gpu = new float[N];

    // --- 3) CPU timing + compute ---
    auto t0 = std::chrono::high_resolution_clock::now();
    sliding_cpu(h_in, h_out_cpu, N, guard_cells, ref_cells, bias);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    // --- 4) GPU setup, copy, compute ---
    size_t   bytes  = N * sizeof(float);
    float   *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int blocks  = (N + BLOCK - 1) / BLOCK;
    cudaEventRecord(start);
    sliding_gpu<<<blocks, BLOCK>>>(d_in, d_out, N,
                                   guard_cells, ref_cells,
                                   bias);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    // --- 5) Validate & report ---
    bool match = true;
    for (unsigned int i = 0; i < N; ++i) {
        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-5f) {
            match = false; break;
        }
    }

    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "Match? "    << (match ? "YES\n" : "NO\n");

    // print first few
    std::cout << "\nIndex : CPU vs GPU\n";
    for (unsigned int i = 0; i < std::min<unsigned int>(10, N); ++i) {
        std::cout << i << " : "
                  << h_out_cpu[i] << " vs "
                  << h_out_gpu[i] << "\n";
    }

    // --- 6) Write out results.txt ---
    std::ofstream outFile("results.txt");
    if (!outFile) {
        std::cerr << "Error opening results.txt for writing\n";
    } else {
        for (unsigned int i = 0; i < N; ++i) {
            outFile << i
                    << ',' << h_out_gpu[i]
                    << '\n';
        }
        outFile.close();
        std::cout << "Results written to results.txt\n";
    }

    // --- 7) Cleanup ---
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    return 0;
}
