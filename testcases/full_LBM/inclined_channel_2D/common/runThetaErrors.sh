#!/bin/bash


for subdir in N*/; do
	cd "$subdir"
	./runErrors.sh &
	cd ../
done

wait

