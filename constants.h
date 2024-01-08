#pragma once

#define BLOCK_SIZE 1024
#define PROCESS_MAX_LENGTH 37
#define PROCESS_MIN_LENGTH 29
#define X1_length 5
#define STEM_SIZE 5
#define L1_MIN 2
#define L2_MIN 5
#define L3_MIN 2
#define L1_MAX 5
#define L2_MAX 7
#define L3_MAX 5

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

    NodeArray(size_t x1_index, size_t x2_index, size_t x3_index, size_t x4_index){
        this->x1_index = x1_index;
        this->x2_index = x2_index;
        this->x3_index = x3_index;
        this->x4_index = x4_index;
    };
};