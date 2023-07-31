#!/bin/bash
# CHOOSE:
ngpus=4
export TESTMODULOTOTAL=16

# GENERAL:
low=0
high=$(($TESTMODULOTOTAL-1))
for mod in $(seq $low $high)
do
  # in some cases launch gradio server, TGI server, or gradio server as inference server with +1 and +2 off base port
  # ports always increment by 3
  export GRADIO_SERVER_PORT=$((7860+$(($mod*3))))
  export TESTMODULO=$mod
  # CVD loops over number of GPUs
  export CUDA_VISIBLE_DEVICES=$((7860+$(($mod % $ngpus))))
  pytest -s -v -n 1 tests &> testsparallel"${mod}".log &
done

# to check on results while running, do:
# grep -a PASSED testsparallel*.log | sed 's/.*PASSED//g' | sort | uniq |wc -l
# grep -a FAILED testsparallel*.log | sed 's/.*FAILED//g' | sort | uniq |wc -l

# to interrupt but still get some results, do:
# ps -auxwf | grep "[v]i" | grep pytest | awk '{print $2}' |xargs kill -s SIGINT
