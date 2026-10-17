# **Senior Research Project**

## GPU Acceleration of Image Processing, _By Joshua Baxter_

### Summary will go here asdhfjl;asdjfl;kajef

### Machine Specs will go here kjasdf;lkasdfdlk

### File Structure will go here askldhjfasl;jf;lsaejf

### How to Run

build everything

```
make
```

run benchmark

```
make run <number of tests>
```

writes results1.csv ... results<N>.csv, plus summary_results.csv with the average of all of them

compare gpu and cpu outputs (reads every results file, writes comparison.csv and summary_results.csv)

```
make compare
```

or, to compare only certain runs

```
./compare_test results1.csv results2.csv
```

# Next Steps:

x axis different sizes, y axis is either mean or mpix per sec
gpu memory vs unified memory for photos

(CLUSTER BAR GRAPH) IF LINE ARE SIMILAR

# Slightly inconsistent results, possible fixes:

have it create a summary_results.csv where it averages all of the results from all the results<number>.csv files, also i dont want it to console out everything, just console out that it is currently working of test set 1 or 2 or whatever and then say when it is done and say when it starts the next.
