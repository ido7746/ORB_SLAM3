#!/bin/bash

# Initialize time variables for each dataset
total_time_V103=0
total_time_V203=0

# Specify a log file in logs folder
log_file="logs/time_logs.txt"

# Create logs directory if it does not exist
mkdir -p logs

# Empty the log file if it exists, otherwise create it
echo "" > $log_file

# Run each command 5 times
for i in {1..5}
do
    echo "Running dataset V103, run $i"
    start_time=$(date +%s%3N)
    ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml ~/Datasets/EuRoc/V103 ./Examples/Monocular/EuRoC_TimeStamps/V103.txt dataset-V103_mono > /dev/null 2>&1
    end_time=$(date +%s%3N)
    elapsed_time=$(($end_time-$start_time))
    total_time_V103=$(($total_time_V103+$elapsed_time))

    # Log the time it took for this run
    echo "V103 run $i time: $elapsed_time ms" >> $log_file

    echo "Running dataset V203, run $i"
    start_time=$(date +%s%3N)
    ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml ~/Datasets/EuRoc/V203 ./Examples/Monocular/EuRoC_TimeStamps/V203.txt dataset-V203_mono > /dev/null 2>&1
    end_time=$(date +%s%3N)
    elapsed_time=$(($end_time-$start_time))
    total_time_V203=$(($total_time_V203+$elapsed_time))

    # Log the time it took for this run
    echo "V203 run $i time: $elapsed_time ms" >> $log_file
done

# Compute average times and print them out
average_time_V103=$(($total_time_V103/5))
average_time_V203=$(($total_time_V203/5))

echo "Average running time for V103: $average_time_V103 ms"
echo "Average running time for V203: $average_time_V203 ms"

# Log the average times
echo "Average running time for V103: $average_time_V103 ms" >> $log_file
echo "Average running time for V203: $average_time_V203 ms" >> $log_file
