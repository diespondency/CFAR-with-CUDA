#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

// Global parameters
int  guard_cells = 50;
int  ref_cells   = 100;
float bias       = -0.015f;

// GPU kernel
__global__ void sliding_gpu(const float*  in,
                            float*        out,
                            unsigned int  n,
                            int           guard_cells,
                            int           ref_cells,
                            float         bias)
{
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int limit = guard_cells + ref_cells;

    // If cell is under consideration (either reference or guard)
    if (idx >= limit && idx < n - limit) {
        float sum = 0.0f;  // set the sum to 0.0f (initialization)
        
        // Take the average of the reference cells
        for (int d = guard_cells + 1; d <= limit; ++d) {
            sum += in[idx - d];
            sum += in[idx + d];
        }
        out[idx] = ((sum) / (2.0f * ref_cells)) - bias;
    }
}

// Function to measure GPU execution time for a specific block size
float measure_gpu_time(float* d_in, float* d_out, unsigned int N, 
                      int threads_per_block, int num_runs) {
    unsigned int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup run
    sliding_gpu<<<blocks, threads_per_block>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        sliding_gpu<<<blocks, threads_per_block>>>(d_in, d_out, N, guard_cells, ref_cells, bias);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return total_time / num_runs; // Return average time per run
}

int main() {
    // Read data.txt → voltages vector
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

    // Create host array
    float *h_in = voltages.data();

    // GPU setup
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Test different thread block sizes (1 to 1024)
    const int num_runs = 1000;  // Number of kernel executions for each block size
    std::vector<int> block_sizes;
    
    // Add powers of 2 from 1 to 1024
    for (int i = 1; i <= 1024; i++) {
        block_sizes.push_back(i);
    }
    
    std::cout << "\nTesting GPU performance with block sizes from 1 to 1024:\n";
    std::cout << "Each configuration runs the kernel " << num_runs << " times\n";
    std::cout << "This may take a while for all block sizes...\n";
    
    // Results file
    std::ofstream resultsFile("detailed_block_size_results.csv");
    if (resultsFile.is_open()) {
        resultsFile << "ThreadsPerBlock,Blocks,AvgTimeMs\n";
    } else {
        std::cerr << "Error: Could not open results file for writing\n";
        return 1;
    }
    
    // Show progress counter
    int total_tests = block_sizes.size();
    int completed = 0;
    int last_percent = -1;
    
    for (int threads_per_block : block_sizes) {
        unsigned int blocks = (N + threads_per_block - 1) / threads_per_block;
        
        // Measure average execution time
        float avg_time = measure_gpu_time(d_in, d_out, N, threads_per_block, num_runs);
        
        // Save to CSV
        resultsFile << threads_per_block << "," << blocks << "," 
                   << std::fixed << std::setprecision(6) << avg_time << "\n";
        
        // Update progress
        completed++;
        int percent = (completed * 100) / total_tests;
        if (percent != last_percent && percent % 10 == 0) {
            std::cout << "Progress: " << percent << "% complete\n";
            last_percent = percent;
        }
    }
    
    resultsFile.close();
    std::cout << "Benchmark complete!\n";
    std::cout << "Results saved to detailed_block_size_results.csv\n";
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}