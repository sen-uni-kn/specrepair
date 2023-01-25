#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for rmi_i in {1..10}
do
  for part in {0..9}
  do
    timeout --foreground --signal=SIGINT --kill-after=5m 1h \
            python ouroboros_rmi_repair_2.py --timestamp "$TIMESTAMP" \
            --timeout 1 --rmi "rmi_10_$rmi_i" --part "$part" "$@"
  done
done
