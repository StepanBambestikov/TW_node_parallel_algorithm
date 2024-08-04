#!bin/bash

kernel_path=../cuda_project
for file in "../../GCF_000001405.40"
    do  
        if [[ $file == *.fna ]]
        then
            for STEM_SIZE in {5..20}
            do  
                echo $STEM_SIZE
                $kernel_path $file $STEM_SIZE 0
            done
        fi
    done
