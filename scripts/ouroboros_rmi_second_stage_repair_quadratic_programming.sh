#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for rmi_i in {1..10}
do
  for part in {0..9}
  do
    python ouroboros_rmi_second_stage_repair_quadratic_programming.py \
      --timestamp "$TIMESTAMP" --rmi "rmi_10_$rmi_i" --part "$part" "$@"
  done
done

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for rmi_i in {1..10}
do
  for part in {0..9}
  do
    python ouroboros_rmi_second_stage_repair_quadratic_programming.py \
      --timestamp "$TIMESTAMP" --rmi "rmi_10_$rmi_i" --part "$part" \
      --second_stage_tolerance 150
  done
done
