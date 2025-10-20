# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Evaluate CDiT on PushT valid split using the same evaluation in train.py (evaluate_ours).
# Loads datasets like train.py and reports DreamSim score while saving sample images.

import os
import argparse
import yaml

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL

from distributed import init_distributed
from models_nwm import CDiT_models
from diffusion import create_diffusion

# PushT dataset utilities (same as train.py)
from datasets.img_transforms import default_transform
from datasets.pusht_dset import load_pusht_slice_train_val
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Reuse evaluation routine implemented in train.py
from train import evaluate_ours


def build_pusht_loaders(rank, seed):
    """Create distributed dataloaders for PushT train/valid, mirroring train.py."""
    transform2 = default_transform()
    datasets, _ = load_pusht_slice_train_val(
        data_path='/checkpoint/amaia/video/gzhou/datasets/pusht_noise',
        with_velocity=True,
        normalizer_type='mean_std',
        use_sin_cos=True,
        n_rollout=None,
        num_hist=1,
        num_pred=1,
        frameskip=25,
        transform=transform2,
        state_based=False,
    )

    samplers = {
        split: DistributedSampler(
            datasets[split],
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=seed,
        )
        for split in ["train", "valid"]
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=10,
            shuffle=False,
            sampler=samplers[split],
            num_workers=16,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        for split in ["train", "valid"]
    }
    return loaders


@torch.no_grad
def main(args):
    _, rank, device, _ = init_distributed()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving evaluation outputs to: {args.output_dir}")

    # Load config: base eval_config + user config override (matches train.py)
    # with open("config/eval_config.yaml", "r") as f:
    #     base_cfg = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_cfg = yaml.safe_load(f)
    # config = {**base_cfg, **user_cfg}
    config = user_cfg

    # Build loaders for PushT
    loaders = build_pusht_loaders(rank=rank, seed=int(config.get('global_seed', 0)))

    # Load model + diffusion + VAE
    ckpt_path = args.ckpt_path
    latent_size = config['image_size'] // 8
    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)
    # ckp = torch.load(f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
    ckp = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(model.load_state_dict(ckp["ema"], strict=True))
    model.eval()
    model.to(device)
    model = torch.compile(model)
    diffusion = create_diffusion(str(250))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # Load decoder weights from checkpoint if present
    if "vae_decoder" in ckp:
        print("Loading decoder weights from checkpoint...")
        vae.decoder.load_state_dict(ckp["vae_decoder"], strict=True)
    else:
        print("No decoder weights found in checkpoint; using default VAE decoder.")

    # Load post_quant_conv weights from checkpoint if present
    if "vae_post_quant_conv" in ckp:
        print("Loading post_quant_conv weights from checkpoint...")
        vae.post_quant_conv.load_state_dict(ckp["vae_post_quant_conv"], strict=True)
    else:
        print("No post_quant_conv weights found in checkpoint; using default VAE post_quant_conv.")
        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
    model_lst = (model, diffusion, vae)

    model, diffusion, vae = model_lst

    # We evaluate on the 'valid' split
    sim_score = evaluate_ours(
        model=model,
        vae=vae,
        diffusion=diffusion,
        test_dataloaders=loaders['valid'],
        rank=rank,
        batch_size=int(config['batch_size']),
        num_workers=int(config['num_workers']),
        latent_size=latent_size,
        device=device,
        save_dir=args.output_dir,
        seed=int(config.get('global_seed', 0)),
        bfloat_enable=None, # not used
        num_cond=num_cond,
    )

    if rank == 0:
        print(f"DreamSim score (lower is better): {float(sim_score):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate CDiT on PushT valid split (DreamSim metric)")

    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckpt-path", type=str, default='')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML (like config/tst.yaml)')
    # parser.add_argument('--ckpt', type=str, default='latest.pth.tar', help='Checkpoint name or absolute path')
    # parser.add_argument('--bfloat16', type=int, default=1)
    # parser.add_argument('--torch-compile', type=int, default=1)
    # parser.add_argument('--save-subdir', type=str, default='pusht_eval', help='Subdir under experiment to save outputs')
    args = parser.parse_args()
    main(args)
