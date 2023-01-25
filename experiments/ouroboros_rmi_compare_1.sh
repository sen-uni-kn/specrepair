#!/bin/bash
# We have to give it 3 repair steps, as it only verifiers at the start of a repair step.
bash ouroboros_rmi_repair_2.sh --save_checkpoints --log_execution_times \
     --max_repair_steps 3 --l1_penalty \
     --falsifier None --verifier LinearRegressionVerifier_single "$@"

bash ouroboros_rmi_repair_2.sh --save_checkpoints --log_execution_times \
     --max_repair_steps 6 --augment_lin_reg \
     --falsifier None --verifier LinearRegressionVerifier_single "$@"

bash ouroboros_rmi_repair_2.sh --save_checkpoints --log_execution_times \
     --max_repair_steps 3 --second_stage_tolerance 150 --l1_penalty \
     --falsifier None --verifier LinearRegressionVerifier_single "$@"

bash ouroboros_rmi_repair_2.sh --save_checkpoints --log_execution_times \
     --max_repair_steps 6 --second_stage_tolerance 150 --augment_lin_reg \
     --falsifier None --verifier LinearRegressionVerifier_single "$@"
