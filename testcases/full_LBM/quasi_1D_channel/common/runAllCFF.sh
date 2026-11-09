#!/bin/bash


for subdir in pfluid*/; do
	cd "$subdir"
	./runCFF.sh &
	cd ../
done

wait
