#!/bin/bash

arr=("haea" "sa" "hill" "selfadaptation" "derandomize" "nsga")
 
for i in "${arr[@]}"
do
	python3 -W ignore test.py $i &
done

