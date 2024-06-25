#pragma once

#define BLOCK_SIZE 1024
#define MAX_STEM_SIZE 10
// __device__ static size_t GPU_PROCESS_MAX_LENGTH = 37;
// __device__ static size_t GPU_PROCESS_MIN_LENGTH = 29;
// __device__ static size_t GPU_L1_MIN = 2;
// __device__ static size_t GPU_L2_MIN = 5;
// __device__ static size_t GPU_L3_MIN = 2;
// __device__ static size_t GPU_L1_MAX = 5;
// __device__ static size_t GPU_L2_MAX = 7;
// __device__ static size_t GPU_L3_MAX = 5;
// __device__ static size_t GPU_STEM_SIZE = 5;
// __device__ static size_t GPU_MAX_MISMATCH_NUMBER = 0;

static size_t PROCESS_MAX_LENGTH = 37;
static size_t PROCESS_MIN_LENGTH = 29;
static size_t L1_MIN = 2;
static size_t L2_MIN = 5;
static size_t L3_MIN = 2;
static size_t L1_MAX = 5;
static size_t L2_MAX = 7;
static size_t L3_MAX = 5;
static size_t STEM_SIZE = 5;
static size_t MAX_MISMATCH_NUMBER = 0;

struct GPUNodeArray {
    bool have_one_node;
    char add_node_mask;
    char node_mask;
};

struct NodeArray {
    size_t x1_index;
    size_t x2_index;
    size_t x3_index;
    size_t x4_index;
    size_t x3_mismatches;
    size_t x4_mismatches;

    NodeArray(size_t x1_index, size_t x2_index, size_t x3_index, size_t x4_index, size_t x3_mismatches, size_t x4_mismatches){
        this->x1_index = x1_index;
        this->x2_index = x2_index;
        this->x3_index = x3_index;
        this->x4_index = x4_index;
        this->x3_mismatches = x3_mismatches;
        this->x4_mismatches = x4_mismatches;
    };
};

struct Constants{
    size_t PROCESS_MAX_LENGTH;
    size_t PROCESS_MIN_LENGTH;
    size_t L1_MIN;
    size_t L2_MIN;
    size_t L3_MIN;
    size_t L1_MAX;
    size_t L2_MAX;
    size_t L3_MAX;
    size_t STEM_SIZE;
    size_t MAX_MISMATCH_NUMBER;
};