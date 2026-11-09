#!/bin/bash


for subdir in *-*/; do
	cd "$subdir"
	./run.sh
	cd ../
done
