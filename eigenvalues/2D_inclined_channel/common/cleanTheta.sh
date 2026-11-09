#!/bin/bash


for subdir in N*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done
