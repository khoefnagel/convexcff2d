#!/bin/bash


for subdir in N*/; do
	cd "$subdir"
	./runCFF.sh
	cd ../
done

