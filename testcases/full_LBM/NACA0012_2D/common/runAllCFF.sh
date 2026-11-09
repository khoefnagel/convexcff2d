#!/bin/bash


for subdir in alpha*/; do
	cd "$subdir"
	./runCFF.sh
	cd ../
done
