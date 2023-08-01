#!/bin/bash
# CHOOSE:
ngpus=4
export TESTMODULOTOTAL=8

NPHYSICAL=`lscpu -p | egrep -v '^\#' | sort -u -t, -k 2,4 | wc -l`
NPROCS=`lscpu -p | egrep -v '^\#' | wc -l`
#
n_jobs=$(($NPROCS / $TESTMODULOTOTAL))
echo "CORES: $NPHYSICAL $NPROCS $n_jobs"

# GENERAL:
lowergpuid=0
low=0
high=$(($TESTMODULOTOTAL-1))
pids=""
for mod in $(seq $low $high)
do
  # in some cases launch gradio server, TGI server, or gradio server as inference server with +1 and +2 off base port
  # ports always increment by 3
  export GRADIO_SERVER_PORT=$((7860+$(($mod*3))))
  export TESTMODULO=$mod
  # CVD loops over number of GPUs
  export CUDA_VISIBLE_DEVICES=$(($lowergpuid+$(($mod % $ngpus))))
  pytest -s -v -n 1 tests &> testsparallel"${mod}".log &
  pid=$!
  echo "MODS: $mod $GRADIO_SERVER_PORT $CUDA_VISIBLE_DEVICES"
  pids="$pids $pid"
done
trap "kill $pids; exit 1" INT

echo "to check on results while running, do:"
echo "grep -a PASSED testsparallel*.log | sed 's/.*PASSED//g' | sort | uniq |wc -l"
echo "grep -a FAILED testsparallel*.log | sed 's/.*FAILED//g' | sort | uniq |wc -l"

echo "to interrupt but still get some results, do:"
#echo "ps -auxwf | grep -v "[g]rep" | grep pytest | awk '{print $2}' |xargs kill -s SIGINT"
echo "kill -s SIGINT $pids"
wait
