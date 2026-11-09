#!/bin/bash


for subdir in theta*/; do
	cd "$subdir"
	./runErrors.sh &
	cd ../
done

wait
