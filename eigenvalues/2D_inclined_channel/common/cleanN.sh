#!/bin/bash


for subdir in *-*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done
