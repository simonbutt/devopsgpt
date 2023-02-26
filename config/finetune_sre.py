import time

out_dir = "out_sre_long_train"
eval_interval = 5
eval_iters = 40
wandb_log = False  # feel free to turn on
wandb_project = "sre.google_char"
wandb_run_name = "ft-" + str(time.time())

dataset = "sre.google_char"
init_from = "gpt2-large"  # this is the large 774M GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# sre.google has 1,506,703 tokens, so 1 epoch ~= 11.5 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 500

# finetune at constant LR
learning_rate = 2e-5
decay_lr = False
