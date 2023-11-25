#!/bin/bash

export epochs=10
export this='that'
export that='the other'

# echo is printing out to the terminal
# 
./runAmanda.py
echo this $this ${that}
export epochs=50
echo "setting new epochs to $epochs"
./runAmanda.py
