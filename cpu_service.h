#pragma once
#include <vector>
#include "constants.h"

bool check_string_equality(std::vector<char>::const_iterator substring_begin, std::vector<char>::const_iterator substring_end, __gnu_cxx::__normal_iterator<const char *, std::vector<char>> content_begin) {
    bool mismatch_occured = false;
    size_t current_index = 0;
    for (auto current_subsequence_value = substring_begin; current_subsequence_value != substring_end; ++current_subsequence_value) {
        auto d = content_begin + int(current_index);
        if (*current_subsequence_value != *d) {
            // if (mismatch_occured) {
            //     return false;
            // }
            // mismatch_occured = true;
            return false;
        }
        ++current_index;
    }
    return true;
}

std::vector<char> make_complement_subsequence(const std::vector<char>& contents, size_t begin_index) {
    std::vector<char> subsequence(5, 'N');
    // if (begin_index == 371425){
    //     auto f = std::vector<char>(contents.begin() + int(begin_index), contents.begin() + int(begin_index) + 30);
    // }
    for (size_t current_subsequence_index = 0; current_subsequence_index < 5; current_subsequence_index++) {
        if (contents[begin_index + current_subsequence_index] == 'A') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'T';
        else if (contents[begin_index + current_subsequence_index] == 'T') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'A';
        else if (contents[begin_index + current_subsequence_index] == 'G') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'C';
        else if (contents[begin_index + current_subsequence_index] == 'C') subsequence[(STEM_SIZE - current_subsequence_index) - 1] = 'G';
    }
    return subsequence;
}

std::vector<size_t> complementary_subsequence_exists_in_begin(const std::vector<char>& contents, size_t x1_index, size_t current_sequence_length, const std::vector<char>& comp_subsequence){
    std::vector<size_t> indexes{};
    for (size_t current_sequence_begin = x1_index + (STEM_SIZE * 2) + L1_MIN + L2_MIN; current_sequence_begin <= x1_index + (current_sequence_length - (STEM_SIZE * 2)) - L3_MIN; ++current_sequence_begin) {
        bool is_complement = check_string_equality(comp_subsequence.begin(), comp_subsequence.end(), contents.begin() + current_sequence_begin);
        if (is_complement) {
            indexes.push_back(current_sequence_begin);
        }
    }
    return indexes;
}

std::vector<size_t> complementary_subsequence_exists_in_end(const std::vector<char>& contents, size_t sequence_begin, size_t x3_index, const std::vector<char>& comp_subsequence){
    std::vector<size_t> indexes{};
    for (size_t current_sequence_begin = (x3_index - L2_MIN) - STEM_SIZE; current_sequence_begin >= sequence_begin + STEM_SIZE + L1_MIN; --current_sequence_begin) {
        bool is_complement = check_string_equality(comp_subsequence.begin(), comp_subsequence.end(), contents.begin() + current_sequence_begin);
        if (is_complement && comp_subsequence[0] != 'N') {
            indexes.push_back(current_sequence_begin);
        }
    }
    return indexes;
}

std::vector<NodeArray> substring_process(const std::vector<char>& contents, size_t current_absolute_index, size_t max_subsequence_length){
    auto node_answers = std::vector<NodeArray>();
    auto begin_comp_subsequence = make_complement_subsequence(contents, current_absolute_index);
    for (size_t current_sequence_length = max_subsequence_length; current_sequence_length >= PROCESS_MIN_LENGTH; --current_sequence_length) {
        //check if sequence has only A T or has more than a half N
        size_t N_count = 0;
        size_t AT_count = 0;
        for (size_t current_sequence_index = current_absolute_index; current_sequence_index < current_absolute_index + current_sequence_length; ++current_sequence_index) {
            if (contents[current_sequence_index] == 'N') ++N_count;
            if (contents[current_sequence_index] == 'A' || contents[current_sequence_index] == 'T') ++AT_count;
        }
        if (N_count > current_sequence_length / 2 || AT_count >= current_sequence_length - 1){
            continue;
        }

        size_t x1_index = current_absolute_index;
        auto x3_indexes = complementary_subsequence_exists_in_begin(contents, x1_index, current_sequence_length, begin_comp_subsequence);
        if (x3_indexes.empty()) {
            continue;
        }
        size_t x4_index = (current_sequence_length - STEM_SIZE) + current_absolute_index;
        auto end_comp_subsequence = make_complement_subsequence(contents, x4_index);
        for (auto current_x3_index : x3_indexes){
            auto x2_indexes = complementary_subsequence_exists_in_end(contents, x1_index, current_x3_index, end_comp_subsequence);
            for (auto current_x2_index : x2_indexes){
                if (
                        (current_x2_index - (x1_index + STEM_SIZE)) <= L1_MAX &&
                        (current_x3_index - (current_x2_index + STEM_SIZE)) <= L2_MAX &&
                        (x4_index - (current_x3_index + STEM_SIZE)) <= L3_MAX)
                {
                    node_answers.emplace_back(x1_index, current_x2_index, current_x3_index, x4_index);
                }
            }
        }
    }
    return node_answers;
}

std::vector<NodeArray> cpu_node_processing(const std::vector<char>& content, GPUNodeArray* thread_answers){
    size_t thread_answers_size = content.size();
    auto node_answers = std::vector<NodeArray>();
    for (size_t current_thread_answer = 0; current_thread_answer < thread_answers_size; ++current_thread_answer) {
        if (current_thread_answer % 40000000 == 0){
            std::cout << (float(current_thread_answer) / thread_answers_size) * 100 << "%" << std::endl;
        }
        auto current_raw_node = thread_answers[current_thread_answer];
        if (current_raw_node.x1_index == 0){
            continue;
        }
        auto substring_nodes = substring_process(content, current_raw_node.x1_index, current_raw_node.first_length);
        node_answers.insert(node_answers.end(), substring_nodes.begin(), substring_nodes.end());
    }
    return node_answers;
}

void add_in_file(const std::vector<NodeArray>& node_answers, std::vector<char> content, const std::string& file_name, const std::vector<char>& chromosome_number){
    std::ofstream outfile;
    outfile.open(file_name, std::ios::app);
    // outfile << "<";
    for (NodeArray current_answer : node_answers) {
        for (auto current_symbol : chromosome_number){
            outfile << current_symbol;
        }
        outfile << " " 
                << current_answer.x1_index << " "
                << current_answer.x1_index + STEM_SIZE << " "
                << current_answer.x2_index << " "
                << current_answer.x2_index + STEM_SIZE << " "
                << current_answer.x3_index << " "
                << current_answer.x3_index + STEM_SIZE << " "
                << current_answer.x4_index << " "
                << current_answer.x4_index + STEM_SIZE << " "
                << current_answer.x2_index - STEM_SIZE - current_answer.x1_index << " " //l1
                << current_answer.x3_index - STEM_SIZE - current_answer.x2_index << " " //l2
                << current_answer.x4_index - STEM_SIZE - current_answer.x3_index << " " //l3
                << std::endl;
        for (size_t current_base_index = current_answer.x1_index;
             current_base_index < current_answer.x4_index + STEM_SIZE; ++current_base_index) {
            outfile << content[current_base_index];
        }
        outfile << std::endl;
    }
}