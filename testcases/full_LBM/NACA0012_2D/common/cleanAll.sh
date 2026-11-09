#!/bin/bash

for subdir in alpha*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done
