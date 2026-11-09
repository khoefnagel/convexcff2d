#!/bin/bash

for subdir in theta*/; do
	cd "$subdir"
	./clean.sh &
	cd ../
done

rm -f orderOfConvergence.txt

wait
