#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <chrono>
#include "cpu_service.h"
#include "gpu_service.cu"
#include "constants.h"
#include <algorithm>
#include <cmath>

GPUNodeArray* gpu_chromosome_processing(std::vector<char> content, Constants constants){
    char* cuda_data_ptr = NULL;
    int data_size = content.size();
    cudaError_t cudaStatus;
    size_t block_number = std::ceil(float(data_size) / BLOCK_SIZE);
    GPUNodeArray* thread_answers = 0;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaMallocManaged(&cuda_data_ptr, data_size * sizeof(char));
    cudaStatus = cudaMemcpy(cuda_data_ptr, content.data(), data_size * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    size_t thread_answers_size = data_size;
    cudaStatus = cudaMalloc((void**)&thread_answers, thread_answers_size * sizeof(GPUNodeArray));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    substringProcessKernel<<<block_number, BLOCK_SIZE>>>(cuda_data_ptr, data_size, thread_answers, constants);

    //Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "substringProcessKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        clear_memory(cuda_data_ptr, thread_answers);
        return NULL;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        clear_memory(cuda_data_ptr, thread_answers);
        return NULL;
    }

    GPUNodeArray* cpu_thread_answers = (GPUNodeArray*)malloc(thread_answers_size * sizeof(GPUNodeArray));
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cpu_thread_answers, thread_answers, thread_answers_size * sizeof(GPUNodeArray), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed!");
        clear_memory(cuda_data_ptr, thread_answers);
        return NULL;
    }
    clear_memory(cuda_data_ptr, thread_answers);
    return cpu_thread_answers;
}

int main(int argc, char* argv[])
{   
    if (argc != 4){
        std::cout << "invalid number of params" << std::endl;
        return -1;
    }
    auto input_file_name = argv[1];
    STEM_SIZE = std::stoi(argv[2]);
    MAX_MISMATCH_NUMBER = std::stoi(argv[3]);

    std::string file_name(input_file_name);
    file_name += "_" + std::to_string(STEM_SIZE) + "_" + std::to_string(MAX_MISMATCH_NUMBER) + "_nodes_new.txt";
    std::cout << file_name << std::endl;
    std::ofstream outfile(file_name);
    outfile.clear();
    std::vector<char> contents;
    contents.reserve(2000000);

    std::string current_line;
    std::ifstream in(input_file_name);
    bool data_exists_flag = false;

    L1_MIN = int(std::ceil(float(STEM_SIZE)/4));
    L3_MIN = int(std::ceil(float(STEM_SIZE)/4));
    L1_MAX = STEM_SIZE;
    L3_MAX = STEM_SIZE;
    L2_MIN = STEM_SIZE;
    L2_MAX = int(std::floor(3 * float(STEM_SIZE) / 2));

    PROCESS_MIN_LENGTH = STEM_SIZE * 4 + L1_MIN * 2 + L2_MIN;
    PROCESS_MAX_LENGTH = STEM_SIZE * 4 + L1_MAX * 2 + L2_MAX;

    subsequence = std::vector<char>(STEM_SIZE, 'N');

    Constants constants = Constants{PROCESS_MAX_LENGTH, PROCESS_MIN_LENGTH, L1_MIN, L2_MIN, L3_MIN, L1_MAX, L2_MAX, L3_MAX, STEM_SIZE, MAX_MISMATCH_NUMBER};

    while (!in.eof())
    {
        std::getline(in, current_line);
        if (current_line[0] == '>'){
            if (data_exists_flag){ //Processing of a single chromosome
                auto cpu_thread_answers = gpu_chromosome_processing(contents, constants);
                auto node_answers = cpu_node_processing(contents, cpu_thread_answers);
                add_in_file(node_answers, contents, file_name, current_line);
                contents.clear();
                free(cpu_thread_answers);
            }
            else {
                data_exists_flag = true;
            }
            continue;
        }
        std::transform(current_line.begin(), current_line.end(), current_line.begin(), ::toupper);
        contents.insert(contents.end(), current_line.begin(), current_line.end());
    }
    return 0;
}
