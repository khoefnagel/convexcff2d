#!/bin/bash


for subdir in theta*/; do
	cd "$subdir"
	./run.sh &
	cd ../
done

wait
