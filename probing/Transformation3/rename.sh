#!/bin/bash


function main(){ 
    rm -rf Mutated.c
    source_file=$1 
    file=$2
    action=$3
    cp $source_file RM/ &&
    cd ./RM/ &&
    ./mutation.sh $file $action &&
    mv Mutated.c ../mutated_code/$2 || cp $file ../not_mutated_code/$2    
    rm -rf $file 
    cd ..
    echo "[+] result..."
    # cat Mutated.c
}


# $1 file 
# $2 action
main $1 $2 $3




