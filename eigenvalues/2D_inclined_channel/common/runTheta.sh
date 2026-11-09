#!/bin/bash


for subdir in N*/; do
	cd "$subdir"
	./run.sh
	cd ../
done
