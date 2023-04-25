#!/bin/bash

for i in {0..10}
do
  python exps/sudoku4.py --no_cuda --nEpoch 800
done