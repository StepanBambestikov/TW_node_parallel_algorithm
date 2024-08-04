#include "constants.h"

void clear_memory(char* cuda_data_ptr, GPUNodeArray* thread_answers){
    cudaFree(cuda_data_ptr);
    cudaFree(thread_answers);
    return;
}

__device__ void GPU_make_complement_subsequence(const char* block_data_ptr, size_t x1_index, char* subsequence, size_t stem_size) {
    for (size_t current_subsequence_index = 0; current_subsequence_index < stem_size; current_subsequence_index++) {
        if (block_data_ptr[x1_index + current_subsequence_index] == 'A') subsequence[(stem_size - current_subsequence_index) - 1] = 'T';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'T') subsequence[(stem_size - current_subsequence_index) - 1] = 'A';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'G') subsequence[(stem_size - current_subsequence_index) - 1] = 'C';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'C') subsequence[(stem_size - current_subsequence_index) - 1] = 'G';
        else if (block_data_ptr[x1_index + current_subsequence_index] == 'N') subsequence[(stem_size - current_subsequence_index) - 1] = 'N';
        else subsequence[(stem_size - current_subsequence_index) - 1] = 'N';
    }
    return;
}

__device__ bool GPU_check_string_equality(const char* block_data_ptr, char* subsequence, size_t current_sequence_begin, size_t stem_size, size_t max_mismatch_number) {
    size_t mismatch_number = 0;
    for (size_t current_subsequence_index = 0; current_subsequence_index < stem_size; ++current_subsequence_index) {
        if (subsequence[current_subsequence_index] != block_data_ptr[current_sequence_begin + current_subsequence_index]) {
            if (mismatch_number == max_mismatch_number) {
                return false;
            }
            ++mismatch_number;
        }
    }
    return true;
}

__device__ int GPU_complementary_subsequence_exists_in_begin(const char* block_data_ptr, size_t x1_index, size_t thread_sequence_length, Constants constants) {
    char subsequence[MAX_STEM_SIZE];
    GPU_make_complement_subsequence(block_data_ptr, x1_index, subsequence, constants.STEM_SIZE);
    for (size_t current_sequence_begin = x1_index + (constants.STEM_SIZE * 2) + constants.L1_MIN + constants.L2_MIN; current_sequence_begin <= x1_index + (thread_sequence_length - (constants.STEM_SIZE * 2)) - constants.L3_MIN; ++current_sequence_begin) {
        bool is_complement = GPU_check_string_equality(block_data_ptr, subsequence, current_sequence_begin, constants.STEM_SIZE, constants.MAX_MISMATCH_NUMBER);
        if (is_complement) {
            return current_sequence_begin;
        }
    }
    return -1;
}

__device__ int GPU_complementary_subsequence_exists_in_end(const char* block_data_ptr, size_t sequence_begin, size_t x4_index, size_t thread_sequence_length, Constants constants) {
    char subsequence[MAX_STEM_SIZE];
    GPU_make_complement_subsequence(block_data_ptr, x4_index, subsequence, constants.STEM_SIZE);
    for (size_t current_sequence_begin = ((sequence_begin + (thread_sequence_length - (constants.STEM_SIZE * 2)) - constants.L3_MIN) - constants.L2_MIN) - constants.STEM_SIZE; current_sequence_begin >= sequence_begin + constants.STEM_SIZE + constants.L1_MIN; --current_sequence_begin) {
        bool is_complement = GPU_check_string_equality(block_data_ptr, subsequence, current_sequence_begin, constants.STEM_SIZE, constants.MAX_MISMATCH_NUMBER);
        if (is_complement) {
            return current_sequence_begin;
        }
    }
    return -1;
}

__device__ int check_length_validity(const char* data_ptr, size_t current_sequence_length, size_t thread_index, Constants constants){
    //check if sequence has only A T or has more than a half N
        size_t N_count = 0;
        size_t A_count = 0;
        size_t T_count = 0;
        size_t G_count = 0;
        size_t C_count = 0;
        for (size_t current_sequence_index = thread_index; current_sequence_index < thread_index + current_sequence_length; ++current_sequence_index) {
            auto current_character = data_ptr[current_sequence_index];
            if (current_character == 'A'){
                ++A_count;
            }
            else if (current_character == 'T'){
                ++T_count;
            }
            else if (current_character == 'C'){
                ++C_count;
            }
            else if (current_character == 'G'){
                ++G_count;
            }
            else{
                ++N_count;
            }
        }
        //Checking that all types of nucleotides are in the sequence, as well as the number of indeterminate nucleotides is less than half
        if (N_count > current_sequence_length / 2 || A_count == 0 || T_count == 0 || G_count == 0 || C_count == 0){
            return -1;
        }

        int x1_index = thread_index;
        int x3_index = GPU_complementary_subsequence_exists_in_begin(data_ptr, x1_index, current_sequence_length, constants);
        if (x3_index == -1) {
            return -1;
        }
        int x4_index = (current_sequence_length - constants.STEM_SIZE) + thread_index;
        int x2_index = GPU_complementary_subsequence_exists_in_end(data_ptr, thread_index, x4_index, current_sequence_length, constants);
        if (x2_index == -1) {
            return -1;
        }
    return x1_index;
}

__global__ void substringProcessKernel(const char* data_ptr, size_t data_length, GPUNodeArray* thread_answers, Constants constants)
{
    size_t thread_index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    GPUNodeArray threadNodeArray = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    if (thread_index > data_length - constants.PROCESS_MIN_LENGTH) { //Filtering out unnecessary streams
        thread_answers[thread_index] = threadNodeArray;
        return;
    }
    //The work of a single thread is to bypass all possible lengths of the required pseudo-nodes for a fixed beginning of the sequence
    for (int current_sequence_length = constants.PROCESS_MAX_LENGTH - constants.PROCESS_MIN_LENGTH; current_sequence_length >= 0; --current_sequence_length) {
        if (thread_index + constants.PROCESS_MIN_LENGTH + current_sequence_length > data_length){
            continue;
        }
        int x1_index = check_length_validity(data_ptr, current_sequence_length + constants.PROCESS_MIN_LENGTH, thread_index, constants);
        if (x1_index != -1){
            threadNodeArray.have_one_node |= 0xFF;
            int byte_index = current_sequence_length / 8;
            int bit_index = current_sequence_length % 8;
            if (byte_index == 0){
                threadNodeArray.node_mask[0] |= (1 << bit_index);
            }
            if (byte_index == 1){
                threadNodeArray.node_mask[1] |= (1 << bit_index);
            }
            if (byte_index == 2){
                threadNodeArray.node_mask[2] |= (1 << bit_index);
            }
            if (byte_index == 3){
                threadNodeArray.node_mask[3] |= (1 << bit_index);
            }
            if (byte_index == 4){
                threadNodeArray.node_mask[4] |= (1 << bit_index);
            }
            if (byte_index == 5){
                threadNodeArray.node_mask[5] |= (1 << bit_index);
            }
        }
    }
    thread_answers[thread_index] = threadNodeArray;
    return;
}
