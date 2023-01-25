#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"

for i in {0..990..10}
do
  timeout --foreground --signal=SIGINT --kill-after=5m 1.1h \
          python collision_detection_repair_7.py --timestamp "$TIMESTAMP" \
          --timeout 1 --radius 0.05 --first_data_point "$i" "$@"
done
