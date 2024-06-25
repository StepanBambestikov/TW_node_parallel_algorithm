#!bin/bash

kernel_path=./program_path
for dir in ./path_to_data_folder
do
    if [ -d "$dir" ]
    then
        for file in "$dir"/*
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
    fi
done
