#!/bin/bash

FILENAME="mi_matrix"

for ((i=1; i<129; i++))
do
    cp "/home/sirens/Public/csi-location-estimation/experiment/phase/detrended/sc_${i}/uniqueness_results/${FILENAME}.png" \
       "/home/sirens/Public/csi-location-estimation/experiment/images/${FILENAME}_${i}.png"
done