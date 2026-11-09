#!/bin/bash


for subdir in theta*/; do
	cd "$subdir"
	./runCFF.sh &
	cd ../
done

wait
