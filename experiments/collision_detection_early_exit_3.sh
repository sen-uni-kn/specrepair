#!/bin/bash

for _ in {0..9}  # repeat all experiments 10 times
do
  ## verifier-only experiments
  bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
       --max_repair_steps 250 --l1_penalty \
       --falsifier None --verifier ERAN_plain_single --verifier_exit_mode optimal "$@"
  bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
       --max_repair_steps 250 --l1_penalty \
       --falsifier None --verifier ERAN_plain_single --verifier_exit_mode early_exit "$@"
  # switch exit modes
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier None --verifier ERAN_plain_single --verifier_exit_mode switch1 "$@"
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier None --verifier ERAN_plain_single --verifier_exit_mode switch3 "$@"
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier None --verifier ERAN_plain_single --verifier_exit_mode switch5 "$@"
  # # runtime budget modes
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier None --verifier ERAN_plain_single --verifier_exit_mode runtime_threshold "$@"
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier None --verifier ERAN_plain_single --verifier_exit_mode runtime_threshold_decrease "$@"
  ## experiments with falsifiers
  bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
       --max_repair_steps 250 --l1_penalty \
       --falsifier DeepOpt_single --verifier ERAN_plain_single --verifier_exit_mode early_exit "$@"
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier DeepOpt_single --verifier ERAN_plain_single --verifier_exit_mode optimal "$@"
  bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
       --max_repair_steps 250 --l1_penalty \
       --falsifier PGD_single[Adam,10] --verifier ERAN_plain_single --verifier_exit_mode early_exit "$@"
  # bash collision_detection_repair_7.sh --save_checkpoints --log_execution_times \
  #      --max_repair_steps 250 --l1_penalty \
  #      --falsifier PGD_single[Adam,10] --verifier ERAN_plain_single --verifier_exit_mode optimal "$@"
done
