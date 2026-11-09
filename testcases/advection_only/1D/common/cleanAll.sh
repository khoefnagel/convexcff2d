#!/bin/bash

for subdir in pfluid*/; do
	cd "$subdir"
	./clean.sh &
	cd ../
done

rm -f orderOfConvergence.txt

wait
