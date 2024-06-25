
# Parallel algorithm for searching for TW-type pseudo-nodes using GPU

This project is an implementation of a parallel algorithm for searching for TW-type pseudonodes in genomes using GPU.

## Description

The TW node structure consists of pairwise complementary sections (X1 - X3) and (X2 - X4) separated by linker sections L1, L2 and L3

![alt text](https://github.com/StepanBambestikov/cuda_dna/blob/main/TW-type.jpg?raw=true)

The algorithm performs filtering using the GPU, repeatedly reducing the possible search locations in the genome, after which the sequential algorithm outputs the final answer.


The answer for one pseudo node found is:
```
   >NT_187361.1 Homo sapiens chromosome 1 unlocalized genomic scaffold, GRCh38.p14 Primary Assembly HSCHR1_CTG1_UNLOCALIZED 630904 630910 630912 630918 630926 630932 630936 630942 4 10 6 x3 score 0 x4 score 0 
 CTCCCGCCGCCGGGAAAAAAGGCGGGAGAAGCCCCGGC
   ```
Where there is information about the location in the chromosome of all the constituent parts.

## Installation

### 1. Prepare package Installation

   ```bash
   sudo apt-get install nvcc=V10.1.243
   ```

### 2. Compile the program

   ```bash
   nvcc --std=c++17 -g kernel.cu -o program_name
   ```

Where:
   - `nvcc`: The path to your installed nvcc package
   - `program_name`: The preferred name of the compiled program

### 3. Run the program

   ```bash
   program_name file_name.fna 5 0
   ```
Where:
   - `file_name.fna`: Genomic file in .fna format
   - `5`: duplex area size
   - `0`: maximum number of mismatches
