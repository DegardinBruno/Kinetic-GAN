#!/bin/bash

for i in $(seq 2.1 0.1 3)
do
    python fid-graph.py /home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy /home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_data.npy --sigma $i
done