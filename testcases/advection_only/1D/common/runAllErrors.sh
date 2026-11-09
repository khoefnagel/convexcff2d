#!/bin/bash


for subdir in pfluid*/; do
	cd "$subdir"
	./runErrors.sh &
	cd ../
done

wait
