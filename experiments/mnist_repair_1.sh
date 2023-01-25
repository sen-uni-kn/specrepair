#!/bin/bash
# Repairs the network mnist_cnn_1 for robustness
# on 10, 25, 50 and 100 data points.

TIMEOUT=3

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
TIMEOUT_PYTHON="$TIMEOUT"
TIMEOUT_BASH="$(bc <<< "$TIMEOUT + 0.1")h"

# 25 images, 100 cases
for i in {0..1225..25}
do
  timeout --foreground --signal=SIGINT --kill-after=5m "$TIMEOUT_BASH" \
          python image_classification_repair_1.py mnist mnist_cnn_1 \
          --timestamp "$TIMESTAMP" --timeout "$TIMEOUT_PYTHON" --radius 03 \
          --num_data_points 25 --first_property "$i" "$@"
done
