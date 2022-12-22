# Diffsched

This is a PyTorch script for training diffusion-based generative models with different noise schedules using [Diffusers](https://github.com/huggingface/diffusers).

## Prerequisites

Make sure to install the required dependencies:

````bash
pip install -r requirements.txt
````

If no [Accelerate](https://github.com/huggingface/accelerate) environment is configured, initialize one:

````bash
accelerate config
````

## Usage

### Arguments

Run `python train.py --help` for an overview of available arguments.

#### Noise Schedule

The `beta_schedule` argument specifies the noise schedule parameters commonly denoted $\beta_t$ in the literature.
Currently supported options:

* `constant`: All betas are set to a small constant value such that all computations remain numerically stable.
* `linear`: The betas increase linearly from $0.0001$ to $0.02$, which is a default choice in the literature (see [Ho et al.](https://arxiv.org/abs/2006.11239))
* `hyperbolic`: The betas increase such that a constant fraction of the signal is lost in each diffusion step (see [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585))
* `cos_squared`: The betas increase such that the remaining signal in noised images decreases with $\cos^2$ (see [Nichol et al.](https://arxiv.org/abs/2102.09672))
* `learned`: The betas are learnable parameters of a monotonic net that is jointly optimized (see [Kingma et al.](https://arxiv.org/abs/2107.00630))

Check out the [Jupyter notebook](schedules.ipynb) for an illustration of the non-learned schedules.

### Execution

Execute the script with `accelerate launch train.py <ARGS>`, for example:

```` bash
accelerate launch train.py \
  --dataset_name="nelorth/oxford-flowers" \
  --output_dir="runs/publish" \
  --overwrite_output_dir
  --logging_dir="/root/tb_logs/publish" \
  --resolution=64 \
  --prediction_type="epsilon" \
  --num_diffusion_steps=1000 \
  --beta_schedule="hyperbolic" \
  --train_batch_size=32 \
  --num_epochs=100 \
  --save_images_epochs=10 \
  --save_model_epochs=10 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --adam_beta1=0.95 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-8 \
  --adam_weight_decay=1e-6 \
  --gradient_accumulation_steps=1 \
  --mixed_precision=no
````