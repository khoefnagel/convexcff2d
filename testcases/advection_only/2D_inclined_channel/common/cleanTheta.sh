#!/bin/bash

for subdir in N*/; do
	cd "$subdir"
	./clean.sh &
	cd ../
done

rm -f convergencePlots/velocity/*.pdf
rm -f convergencePlots/species/*/*.pdf

wait
