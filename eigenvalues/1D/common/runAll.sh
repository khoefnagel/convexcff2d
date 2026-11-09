#!/bin/bash


for subdir in pfluid*/; do
	cd "$subdir"
	./run.sh &
	cd ../
done

wait
