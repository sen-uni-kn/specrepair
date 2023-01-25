#!/bin/bash
# Some experiments yield huge log files (ouroboros_rmi_repair_2).
# Running collect_experiment_results.py for such experiments may be unnecessarily
# time consuming.
# This script provides a faster way to summarise the results of an experiment.
# It prints the number of successful repairs and failing repairs and
# summarises repaired network performance and total runtimes.
# Input: The directory to summarise.

NUM_SUCCESS=$(grep -o "All properties verified! Repair successful." "$1"/**/log.txt | wc -l)
NUM_MAX_ITERATIONS_EXHAUSTED=$(grep -o "Repair failed: Maximum number of iterations exhausted." "$1"/**/log.txt | wc -l)
NUM_BACKEND_FAILURE=$(grep -o "Repair finished: failure" "$1"/**/log.txt | wc -l)
NUM_BACKEND_FAILURE=$(bc <<< "$NUM_BACKEND_FAILURE - $NUM_MAX_ITERATIONS_EXHAUSTED")
NUM_VERIFICATION_PROBLEM=$(grep -o "The following properties could not be verified due to errors" "$1"/**/log.txt | wc -l)
NUM_TIMEOUT=$(grep -o "Experiment timed out" "$1"/**/log.txt | wc -l)
TOTAL_CASES=$(ls -1q "$1"/**/log.txt | wc -l)
NUM_OTHER=$(bc <<< "$TOTAL_CASES - $NUM_SUCCESS - $NUM_BACKEND_FAILURE - $NUM_MAX_ITERATIONS_EXHAUSTED - $NUM_VERIFICATION_PROBLEM - $NUM_TIMEOUT")

cat > "$1/results.yaml" << EOF
summary:
    successful_experiments:
        count: $NUM_SUCCESS
    experiments_failing_due_to_backend_failure:
        count: $NUM_BACKEND_FAILURE
    experiments_failing_due_to_max_iterations_exhausted:
        count: $NUM_MAX_ITERATIONS_EXHAUSTED
    experiments_failing_due_to_verification_problem:
        count: $NUM_VERIFICATION_PROBLEM
    experiments_failing_due_to_timeout:
        count: $NUM_TIMEOUT
    other_experiments:
        count: $NUM_OTHER
cases:
EOF

for FILE in "$1"/**/log.txt;
do
  TOTAL_RUNTIME=$(grep -oe "Executing overall repair took: [0-9]\+\.[0-9]\+ seconds\." "$FILE" | sed -r 's/Executing overall repair took: ([0-9]+\.[0-9]+) seconds\./\1/')
  CASE_NAME=$(echo "$FILE" | sed -r "s:.*/([^/]*)/log.txt:\1:")
  cat >> "$1/results.yaml" << EOF
    $CASE_NAME:
        total_runtime: $TOTAL_RUNTIME
EOF
done

# cat "$1/results.yaml"
