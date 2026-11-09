#!/bin/bash


for subdir in c*/; do
	cd "$subdir"
	./runCFF.sh
	cd ../
done
