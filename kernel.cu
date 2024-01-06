#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
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

GPUNodeArray* gpu_chromosome_processing(std::vector<char> content){
    char* cuda_data_ptr = NULL;
    int data_size = content.size();
    std::cout << "data size: " <<  data_size << std::endl;
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
    cudaStatus = cudaMalloc((void**)&thread_answers, thread_answers_size * sizeof(NodeArray));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    std::cout << "block number " << block_number << std::endl;
    std::cout << "gpu work begin: " << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    substringProcessKernel<<<block_number, BLOCK_SIZE>>>(cuda_data_ptr, data_size, thread_answers);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "duration time: " << duration.count() << std::endl;

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

    GPUNodeArray* cpu_thread_answers = (GPUNodeArray*)malloc(thread_answers_size * sizeof(NodeArray));
    // Copy output vector from GPU buffer to host memory.
    std::cout << "gpu to host copy begin: " << std::endl;
    cudaStatus = cudaMemcpy(cpu_thread_answers, thread_answers, thread_answers_size * sizeof(NodeArray), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed!");
        clear_memory(cuda_data_ptr, thread_answers);
        return NULL;
    }
    std::cout << "gpu to host copy ended: " << std::endl;
    clear_memory(cuda_data_ptr, thread_answers);
    return cpu_thread_answers;
}

int main()
{   
    std::string file_name("nodes.txt");
    std::ofstream outfile(file_name);
    outfile.clear();
    char* cuda_data_ptr = 0;
    std::vector<char> contents;
    contents.reserve(2000000);

    char current_char;
    std::ifstream in("hg38.fa");
    bool data_exists_flag = false;
    std::cout << "file process begin" << std::endl;
    size_t data_length = 10000000;
    std::vector<char> chromosome_number_str;
    while (!in.eof())
    {
        in >> current_char;
        if (current_char == '>'){
            if (data_exists_flag){
                auto cpu_thread_answers = gpu_chromosome_processing(contents);
                auto node_answers = cpu_node_processing(contents, cpu_thread_answers);
                add_in_file(node_answers, contents, file_name, chromosome_number_str);
                contents.clear();
                chromosome_number_str.clear();
            }
            else {
                data_exists_flag = true;
            }
            while(current_char != 'N'){
                chromosome_number_str.push_back(current_char);
                in >> current_char;
                if (chromosome_number_str.size() > 2 && chromosome_number_str[1] == 'M'){
                    break;
                }
            }
            if (!chromosome_number_str.empty()){
                std::cout << "chromosome - " << chromosome_number_str[1] << std::endl;
            }
            continue;
        }
        contents.push_back(current_char);
        // if (num_characters > data_length + 100){
        //     break;
        // }
    }
    return 0;
}