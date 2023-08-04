#!/bin/bash
# CHOOSE:
ngpus=4
# below has to match GPUs for A6000s due to long context tests
export TESTMODULOTOTAL=4

pip install pytest-instafail || true
docker ps | grep text-generation-inference | awk '{print $1}' | xargs docker stop
killall -s SIGINT pytest
killall -s SIGTERM pytest
killall -s 9 pytest
pkill --signal 9 -f weaviate-embedded/weaviate

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
  export n_jobs=$n_jobs
  export OMP_NUM_THREADS=$n_jobs
  export NUMEXPR_MAX_THREADS=$n_jobs
  export OPENBLAS_NUM_THREADS=$n_jobs
  # By default, OpenBLAS will restrict the Cpus_allowed to be 0x1.
  export OPENBLAS_MAIN_FREE=$n_jobs
  export MKL_NUM_THREADS=$n_jobs
  export H2OGPT_BASE_PATH="./base_$mod"

  # huggyllama test uses alot of memory, requires TESTMODULOTOTAL=ngpus for even A6000s
  # pytest --instafail -s -v -n 1 tests -k "not test_huggyllama_transformers_pr" &> testsparallel"${mod}".log &
  pytest --instafail -s -v -n 1 tests  &> testsparallel"${mod}".log &
  pid=$!
  echo "MODS: $mod $GRADIO_SERVER_PORT $CUDA_VISIBLE_DEVICES $H2OGPT_BASE_PATH"
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
