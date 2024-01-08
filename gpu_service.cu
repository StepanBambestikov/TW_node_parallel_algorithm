#include "constants.h"

void clear_memory(char* cuda_data_ptr, GPUNodeArray* thread_answers){
    cudaFree(cuda_data_ptr);
    cudaFree(thread_answers);
    return;
}

__device__ void GPU_make_complement_subsequence(const char* block_data_ptr, size_t x1_index, char* subsequence) {
    for (size_t current_subsequence_index = 0; current_subsequence_index < 5; current_subsequence_index++) {
        if (block_data_ptr[x1_index + current_subsequence_index] == 'A') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'T';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'T') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'A';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'G') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'C';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'C') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'G';
        else subsequence[STEM_SIZE - current_subsequence_index] = 'N';
    }
    return;
}

__device__ bool GPU_check_string_equality(const char* block_data_ptr, char* subsequence, size_t current_sequence_begin) {
    bool mismatch_occured = false;
    for (size_t current_subsequence_index = 0; current_subsequence_index < 5; ++current_subsequence_index) {
        if (subsequence[current_subsequence_index] != block_data_ptr[current_sequence_begin + current_subsequence_index]) {
            if (mismatch_occured) {
                return false;
            }
            mismatch_occured = true;
            // return false;
        }
    }
    return true;
}

__device__ size_t GPU_complementary_subsequence_exists_in_begin(const char* block_data_ptr, size_t x1_index, size_t thread_sequence_length) {
    char subsequence[5];
    // return 1;
    GPU_make_complement_subsequence(block_data_ptr, x1_index, subsequence);
    for (size_t current_sequence_begin = x1_index + (STEM_SIZE * 2) + L1_MIN + L2_MIN; current_sequence_begin <= x1_index + (thread_sequence_length - (STEM_SIZE * 2)) - L3_MIN; ++current_sequence_begin) {
        bool is_complement = GPU_check_string_equality(block_data_ptr, subsequence, current_sequence_begin);
        if (is_complement) {
            return current_sequence_begin;
        }
    }
    return 0;
}

__device__ size_t GPU_complementary_subsequence_exists_in_end(const char* block_data_ptr, size_t sequence_begin, size_t x4_index, size_t thread_sequence_length) {
    char subsequence[5];
    // return 1;
    GPU_make_complement_subsequence(block_data_ptr, x4_index, subsequence);
    for (size_t current_sequence_begin = ((sequence_begin + (thread_sequence_length - (STEM_SIZE * 2)) - L3_MIN) - L2_MIN) - STEM_SIZE; current_sequence_begin >= sequence_begin + STEM_SIZE + L1_MIN; --current_sequence_begin) {
        bool is_complement = GPU_check_string_equality(block_data_ptr, subsequence, current_sequence_begin);
        if (is_complement) {
            return current_sequence_begin;
        }
    }
    return 0;
}

__device__ size_t check_length_validity(const char* data_ptr, size_t current_sequence_length, size_t thread_index){
    //check if sequence has only A T or has more than a half N
        size_t N_count = 0;
        size_t AT_count = 0;
        for (size_t current_sequence_index = thread_index; current_sequence_index < thread_index + current_sequence_length; ++current_sequence_index) {
            if (data_ptr[current_sequence_index] == 'N') ++N_count;
            if (data_ptr[current_sequence_index] == 'A' || data_ptr[current_sequence_index] == 'T') ++AT_count;
        }
        if (N_count > current_sequence_length / 2 || AT_count >= current_sequence_length - 1){
            return 0;
        }

        size_t x1_index = thread_index;
        size_t x3_index = GPU_complementary_subsequence_exists_in_begin(data_ptr, x1_index, current_sequence_length);
        if (x3_index == 0) {
            return 0;
        }
        size_t x4_index = (current_sequence_length - 5) + thread_index;
        size_t x2_index = GPU_complementary_subsequence_exists_in_end(data_ptr, thread_index, x4_index, current_sequence_length);
        if (x2_index == 0) {
            return 0;
        }
    return x1_index;
}

__global__ void substringProcessKernel(const char* data_ptr, size_t data_length, GPUNodeArray* thread_answers)
{
    size_t thread_index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (thread_index >= data_length - PROCESS_MAX_LENGTH) {
        GPUNodeArray threadNodeArray = {false, 0x00, 0x00};
        thread_answers[thread_index] = threadNodeArray;
        return;
    }
    GPUNodeArray threadNodeArray = {false, 0x00, 0x00};
    for (size_t current_sequence_length = PROCESS_MAX_LENGTH; current_sequence_length >= PROCESS_MIN_LENGTH; --current_sequence_length) {
        size_t x1_index = check_length_validity(data_ptr, current_sequence_length, thread_index);
        if (x1_index != 0){
            threadNodeArray.have_one_node = true;
            if (current_sequence_length == PROCESS_MAX_LENGTH){
                threadNodeArray.add_node_mask |= 0x01;
            } else{
                threadNodeArray.node_mask |= (1 << (current_sequence_length - PROCESS_MIN_LENGTH));
            }
        }
    }
    thread_answers[thread_index] = threadNodeArray;
    return;
}