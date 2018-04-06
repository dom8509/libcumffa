#nvprof --analysis-metrics -o metrics.nvprof python runPerformanceTests.py
#nvprof -o timeline.nvprof python runPerformanceTests.py

#rm -fR profile_results
#mkdir profile_results

#CURRENT=parAdd
#mkdir profile_results/00_$CURRENT
#nvprof --analysis-metrics -o profile_results/00_$CURRENT/metrics.nvprof ./runFunction $CURRENT 2047 100
#nvprof -o profile_results/00_$CURRENT/timeline.nvprof ./runFunction $CURRENT 2047 100

#CURRENT=parAddWithEvents
#mkdir profile_results/01_$CURRENT
#nvprof --analysis-metrics -o profile_results/01_$CURRENT/metrics.nvprof ./runFunction $CURRENT 2047 100
#nvprof -o profile_results/01_$CURRENT/timeline.nvprof ./runFunction $CURRENT 2047 100

#CURRENT=parAddWithEventsAndDummy
#mkdir profile_results/02_$CURRENT
#nvprof --analysis-metrics -o profile_results/02_$CURRENT/metrics.nvprof ./runFunction $CURRENT 2047 100
#nvprof -o profile_results/02_$CURRENT/timeline.nvprof ./runFunction $CURRENT 2047 100

CURRENT=parAddOwnStream
mkdir profile_results/04_$CURRENT
nvprof --analysis-metrics -o profile_results/04_$CURRENT/metrics.nvprof ./runFunction $CURRENT 2047 100 1
nvprof -o profile_results/04_$CURRENT/timeline.nvprof ./runFunction $CURRENT 2047 100 1
