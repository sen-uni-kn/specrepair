#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for rmi_i in {1..50}
do
  timeout --foreground --signal=SIGINT --kill-after=5m 1.1h \
          python ouroboros_rmi_repair_1.py --timestamp "$TIMESTAMP" \
          --timeout 1 --rmi "rmi_304_$rmi_i" "$@"
done
