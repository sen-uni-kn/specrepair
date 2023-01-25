#!/bin/bash
# This experiment is just for demonstrating that we can repair more using falsifiers
# than when only using verifiers. Complete verification is too expensive for the
# network used.

TIMEOUT=3

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
TIMEOUT_PYTHON="$TIMEOUT"
TIMEOUT_BASH="$(bc <<< "$TIMEOUT + 0.1")h"

# 1 image
for i in {0..5}
do
  timeout --foreground --signal=SIGINT --kill-after=5m "$TIMEOUT_BASH" \
          python image_classification_repair_1.py cifar10 eran_cifar_conv_maxpool \
          --timestamp "$TIMESTAMP" --timeout "$TIMEOUT_PYTHON" --radius 0012 \
          --num_data_points 1 --first_property "$i" "$@"
done

# # 10 images
# for i in {0..90..10}
# do
#   timeout --foreground --signal=SIGINT --kill-after=5m "$TIMEOUT_BASH" \
#           python image_classification_repair_1.py cifar10 cifar10_cnn_1 \
#           --timestamp "$TIMESTAMP" --timeout "$TIMEOUT_PYTHON" --radius 0012 \
#           --num_data_points 10 --first_property "$i" "$@"
# done
