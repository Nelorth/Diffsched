"""
This file contains the main PyTorch training script.
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from monotonicity import MonotonicNet
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm


def parse_args():
    """
    Parses command line arguments.and collects them in one object.

    :return: An argparse.Namespace containing all parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training script for DDPMs with different noise schedules.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="If `output_dir` already exists, it will be overwritten if this flag is specified."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the diffusion noise (\"epsilon\") "
             "or directly the reconstructed image (\"sample\").",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="simple",
        choices=["simple", "weighted"],
        help="Whether the squared errors should just be averaged (simple)"
             "or weighted according to the theoretical loss derivation (weighted).",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=1000,
        help="Number of diffusion steps to use, i.e., the length of the DDPM Markov chain."
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["constant", "linear", "cos_squared", "hyperbolic"],
        help="Schedule type to use for the noise variance in the diffusion process."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (a the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset_name from the HuggingFace hub or a local train_data_dir.")

    return args


def extract_into_tensor(arr, indices, broadcast_shape):
    """
    Extracts values from a 1D numpy array according to a list of indices.

    :param arr: The 1D numpy array.
    :param indices: A tensor of indices into the array to extract.
    :param broadcast_shape: A larger shape of K dimensions with the first
                            dimension equal to the length of time steps.
    :return: A tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[indices].float().to(indices.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main(args):
    logger = get_logger(__name__)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    denoising_model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    learnable_params = list(denoising_model.parameters())

    if args.beta_schedule == "learned":
        cumalpha_model = MonotonicNet(min=-10., max=10.)  # negative logits for alphas_cumprod
        learnable_params += list(cumalpha_model.parameters())
    else:
        if args.beta_schedule == "linear":
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_diffusion_steps,
                prediction_type=args.prediction_type,
                beta_schedule="linear"
            )
        elif args.beta_schedule == "constant":
            betas = torch.tensor(1 - 0.01 ** (1 / args.num_diffusion_steps)).repeat(args.num_diffusion_steps)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_diffusion_steps,
                prediction_type=args.prediction_type,
                trained_betas=betas.tolist()
            )
        elif args.beta_schedule == "hyperbolic":
            betas = (1 / (args.num_diffusion_steps - torch.arange(0, args.num_diffusion_steps))).clamp(1e-4, 0.5)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_diffusion_steps,
                prediction_type=args.prediction_type,
                trained_betas=betas.tolist()
            )
        elif args.beta_schedule == "cos_squared":
            s = 0.008
            f = lambda t: torch.cos(torch.pi / 2 * (t / args.num_diffusion_steps + s) / (1 + s)) ** 2
            cumalphas = f(torch.arange(args.num_diffusion_steps))
            cumalphas_prev = cumalphas.roll(shifts=1)
            cumalphas_prev[0] = 1
            betas = torch.maximum(1 - cumalphas / cumalphas_prev, torch.tensor(1e-4))
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_diffusion_steps,
                prediction_type=args.prediction_type,
                trained_betas=betas.tolist()
            )
        else:
            raise ValueError(f"Unsupported beta schedule: {args.beta_schedule}")

        if accelerator.is_main_process:
            logger.info("beta: from {:.8f} to {:.8f}".format(noise_scheduler.betas[0], noise_scheduler.betas[-1]))
            logger.info("cumalpha: from {:.8f} to {:.8f}".format(
                noise_scheduler.alphas_cumprod[0], noise_scheduler.alphas_cumprod[-1]))

    optimizer = torch.optim.AdamW(
        learnable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
    preprocessing = Compose([Resize(args.resolution),
                             CenterCrop(args.resolution),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             Normalize([0.5], [0.5])])
    dataset.set_transform(lambda batch: {"input": [preprocessing(image.convert("RGB")) for image in batch["image"]]})
    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    logger.info(f"Dataset size: {len(dataset)}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )

    denoising_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        denoising_model, optimizer, train_dataloader, lr_scheduler
    )
    if args.beta_schedule == "learned":
        cumalpha_model = accelerator.prepare(cumalpha_model)

    # Initialize directories and trackers
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=args.overwrite_output_dir)
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    def update_beta_schedule(old_scheduler, cumalpha_model):
        with accelerator.accumulate(cumalpha_model):
            cumalpha_logits = cumalpha_model(torch.linspace(0, 1, old_scheduler.num_train_timesteps))

        cumalpha = torch.sigmoid(-cumalpha_logits)
        alpha = cumalpha / torch.roll(cumalpha, shifts=1)
        alpha[0] = cumalpha[0]
        beta = 1 - alpha

        return DDPMScheduler(
            num_train_timesteps=old_scheduler.num_train_timesteps,
            prediction_type=old_scheduler.prediction_type,
            trained_betas=beta.tolist(),
        )

    for epoch in range(1, args.num_epochs + 1):
        denoising_model.train()
        if args.beta_schedule == "learned":
            cumalpha_model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.beta_schedule == "learned":
                noise_scheduler = update_beta_schedule(noise_scheduler, cumalpha_model)

            clean_images = batch["input"]
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            if args.prediction_type == "epsilon":
                model_target = noise
            elif args.prediction_type == "sample":
                model_target = clean_images
            else:
                raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

            # Sample a random timestep for each image
            timesteps = torch.randint(1, noise_scheduler.config.num_train_timesteps,
                                      (clean_images.shape[0],), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(denoising_model):
                # Predict the noise residual
                model_output = denoising_model(noisy_images, timesteps).sample

                # Calculate the loss
                if args.loss_type == "simple":
                    loss = F.mse_loss(model_output, model_target)
                else:
                    alpha_t = extract_into_tensor(noise_scheduler.alphas, timesteps, (clean_images.shape[0], 1, 1, 1))
                    cumalpha_prev = extract_into_tensor(noise_scheduler.alphas_cumprod, timesteps - 1,
                                                        (clean_images.shape[0], 1, 1, 1))
                    weights = (1 - alpha_t) / 2.0 / alpha_t / (1 - cumalpha_prev)
                    loss = torch.mean(weights * F.mse_loss(model_output, model_target, reduction="none"))

                # Backpropagate to obtain gradients
                accelerator.backward(loss)

                # Clip large gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(denoising_model.parameters(), 1.0)
                    if args.beta_schedule == "learned":
                        accelerator.clip_grad_norm_(cumalpha_model.parameters(), 1.0)

                # Update the model parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Generate sample images for visual inspection
            if epoch == 1 or epoch % args.save_images_epochs == 0 or epoch == args.num_epochs:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(denoising_model), scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device).manual_seed(0)

                # Run the pipeline to get a batch of generated images
                images = pipeline(generator=generator, batch_size=args.eval_batch_size, output_type="numpy").images

                # Denormalize the images and save to tensorboard
                images_processed = ((images * 255).round().astype("uint8")).transpose(0, 3, 1, 2)
                accelerator.trackers[0].writer.add_images("test_samples", images_processed, epoch)

            # Save a model checkpoint
            if epoch == 1 or epoch % args.save_model_epochs == 0 or epoch == args.num_epochs:
                pipeline.save_pretrained(f"{args.output_dir}/{epoch}")

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    check_min_version("0.10.0.dev0")
    main(parse_args())
