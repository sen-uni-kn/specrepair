#!/bin/bash
# verifier-only experiments
bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
     --l1_penalty --loss_function random_sample_hcas_loss \
     --falsifier None --verifier ERAN_single --verifier_exit_mode optimal "$@"
bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
     --l1_penalty --loss_function random_sample_hcas_loss \
     --falsifier None --verifier ERAN_single --verifier_exit_mode early_exit "$@"
# switch exit modes
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier None --verifier ERAN_single --verifier_exit_mode switch1 "$@"
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier None --verifier ERAN_single --verifier_exit_mode switch3 "$@"
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier None --verifier ERAN_single --verifier_exit_mode switch5 "$@"
# runtime budget modes
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier None --verifier ERAN_single --verifier_exit_mode runtime_threshold "$@"
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier None --verifier ERAN_single --verifier_exit_mode runtime_threshold_decrease "$@"
# experiments with falsifiers
bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
     --l1_penalty --loss_function random_sample_hcas_loss \
     --falsifier DeepOpt_single --verifier ERAN_single --verifier_exit_mode early_exit "$@"
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier DeepOpt_single --verifier ERAN_single --verifier_exit_mode optimal "$@"
# PGD never finds counterexamples => this is the same as running verifier only
# bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
#      --l1_penalty --loss_function random_sample_hcas_loss \
#      --falsifier PGD_single[Adam,100] --verifier ERAN_single --verifier_exit_mode early-exit "$@"

# extra experiments with multiple counterexamples
bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
     --l1_penalty --loss_function random_sample_hcas_loss \
     --falsifier None --verifier ERAN --verifier_exit_mode optimal "$@"
bash acasxu_repair_1.sh --save_checkpoints --log_execution_times --max_repair_steps 250 \
     --l1_penalty --loss_function random_sample_hcas_loss \
     --falsifier None --verifier ERAN --verifier_exit_mode early_exit "$@"
