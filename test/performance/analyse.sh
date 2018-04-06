#nvprof --print-summary-per-gpu --print-summary-per-gpu  --print-summary --kernels cudaParAddKernel --events threads_launched,warps_launched --metrics warp_execution_efficiency,warp_nonpred_execution_efficiency python runPerformanceTests.py
nvprof --print-api-trace --print-gpu-trace --events threads_launched,warps_launched --metrics inst_per_warp,inst_executed,warp_execution_efficiency,warp_nonpred_execution_efficiency  ./runFunction2047