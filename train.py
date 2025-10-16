# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

from isolated_nwm_infer import model_forward_wrapper, model_forward_wrapper_ours
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt 
import yaml
from tqdm import tqdm


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL
import torch.nn.functional as F

from distributed import init_distributed
from models import CDiT_models
from diffusion import create_diffusion
from datasets_nwm import TrainingDataset
from misc import transform

# import sys
# sys.path.insert(0,'/home/gaoyuezhou/dev/dino_wm_private_hierarchical')

from datasets.pusht_dset import PushTDataset, load_pusht_slice_train_val

def rotate_data(x, y, rel_t):
    """
    Rotates all images in x by 0, 90, 180, and 270 degrees.
    x: (B, T, C, H, W)
    y: (B, ...)
    rel_t: (B, ...)
    Returns:
        x_rot: (4*B, T, C, H, W)
        y_rot: (4*B, ...)
        rel_t_rot: (4*B, ...)
    """
    # Rotations: 0, 90, 180, 270 degrees
    x_list = [x]
    for k in [1, 2, 3]:
        # Rotate each image in the batch by k*90 degrees (dim -1 and -2 are H, W)
        x_rot = torch.rot90(x, k=k, dims=(-2, -1))
        x_list.append(x_rot)
    x_rot = torch.cat(x_list, dim=0)  # (4*B, T, C, H, W)
    y_rot = y.repeat(4, *[1 for _ in range(y.dim()-1)])
    rel_t_rot = rel_t.repeat(4, *[1 for _ in range(rel_t.dim()-1)])
    return x_rot, y_rot, rel_t_rot


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, device, _ = init_distributed()

    print(f"[Rank {dist.get_rank()}] Node: {os.uname().nodename}, "
      f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, on device={device}")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # -------- Independent decoder fine-tuning setup --------
    finetune_decoder = bool(config.get('finetune_decoder', False))
    # Freeze entire VAE by default
    for p in tokenizer.parameters():
        p.requires_grad = False
    decoder_optimizer = None
    if finetune_decoder:
        decoder_params = []
        if hasattr(tokenizer, "decoder"):
            for p in tokenizer.decoder.parameters():
                p.requires_grad = True
            decoder_params += list(tokenizer.decoder.parameters())
            tokenizer.decoder.train()
        if hasattr(tokenizer, "post_quant_conv"):
            for p in tokenizer.post_quant_conv.parameters():
                p.requires_grad = True
            decoder_params += list(tokenizer.post_quant_conv.parameters())
            tokenizer.post_quant_conv.train()
        decoder_lr = float(config.get('decoder_lr', 3e-4))
        decoder_optimizer = torch.optim.AdamW(decoder_params, lr=decoder_lr, weight_decay=0)
        

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # load existing checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
        if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False) 

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")
        
        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint['epoch'] + 1
        
        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]
        
        if "scaler" in latest_checkpoint:
            scaler.load_state_dict(latest_checkpoint["scaler"])
        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        # goals_per_obs = 4 # standardize testing
                        goals_per_obs = 1 
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    # loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=4,
    #         shuffle=False, # already shuffled in TrajSlicerDataset
    #         num_workers=12,
    #         collate_fn=None,
    #     )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    from datasets.img_transforms import default_transform
    transform2 = default_transform()
    datasets, traj_dsets = load_pusht_slice_train_val(
        data_path = '/checkpoint/amaia/video/gzhou/datasets/pusht_noise', 
        with_velocity=True,
        normalizer_type='mean_std', # doesn't matter
        use_sin_cos=True,
        n_rollout=None,
        num_hist=1,
        num_pred=1,
        frameskip=25,
        transform=transform2,
        state_based=False,
    )
    # dataloaders = {
    #     x: torch.utils.data.DataLoader(
    #         datasets[x],
    #         batch_size=4,
    #         shuffle=False, # already shuffled in TrajSlicerDataset
    #         num_workers=12,
    #         collate_fn=None,
    #     )
    #     for x in ["train", "valid"]
    # }
    samplers = {
        x:
            DistributedSampler(
                datasets[x],
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=False,
                seed=args.global_seed
            )
        for x in ["train", "valid"]
    }

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=config['batch_size'],
            shuffle=False,
            sampler=samplers[x],
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        for x in ["train", "valid"]
    }
    


    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    data_augmentation = bool(config.get('data_augmentation', False))

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

    # for x, y, rel_t in loader:
        # count = 0
        # for x, y, rel_t in tqdm(loader, desc=f"Epoch {epoch} [train]", disable=(rank != 0), total=len(loader)):
        for x, y, z in tqdm(dataloaders['train'], desc=f"Epoch {epoch} [train]", disable=(rank != 0), total=len(dataloaders['train'])):
            x = x['visual']
            x = x.to(device, non_blocking=True) # B, T, C, H, W
            y = y.to(device, non_blocking=True) # B, T/2, 3
            # change y to all zeros
            y = torch.zeros(y.shape[0], y.shape[1], 3).to(device)
            rel_t = torch.zeros(y.shape[0], 1).to(device)
            # rel_t = rel_t.to(device, non_blocking=True)
            if data_augmentation:
                x, y, rel_t = rotate_data(x, y, rel_t)

            x_img = x  # keep reference to original images for reconstruction
            
            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B, T = x.shape[:2]
                    x = x.flatten(0,1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215) # _, 4, 28, 28
                    x = x.unflatten(0, (B, T))

            # ----- Decoder reconstruction (independent loss/optimizer) -----
            if finetune_decoder:
                latents_flat = x.flatten(0, 1)  # (B*T, 4, H/8, W/8)
                # Unscale before decoding as SD VAE expects latents scaled by 1/0.18215
                recon_flat = tokenizer.decode(latents_flat / 0.18215).sample  # (B*T, C, H, W)
                recon = recon_flat.unflatten(0, (B, T))
                decoder_loss = F.l1_loss(recon, x_img)

                # Separate decoder optimizer step
                decoder_optimizer.zero_grad(set_to_none=True)
                decoder_loss.backward()
                decoder_optimizer.step()

            num_goals = T - num_cond
            x_start = x[:, num_cond:].flatten(0, 1)
            x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            # y = y.flatten(0, 1)
            y = y[:, num_cond:].flatten(0, 1)
            rel_t = rel_t.flatten(0, 1)
            # import pdb; pdb.set_trace()

            t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
            model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
            loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            if not bfloat_enable:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()
            
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size()*x_cond.shape[0]*steps_per_sec
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if finetune_decoder:
                    try:
                        dec_l1 = float(decoder_loss.item())
                    except Exception:
                        dec_l1 = float('nan')
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Dec L1: {dec_l1:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    # Save decoder and post_quant_conv weights if finetuning
                    if finetune_decoder:
                        if hasattr(tokenizer, "decoder") and tokenizer.decoder is not None:
                            checkpoint["vae_decoder"] = tokenizer.decoder.state_dict()
                        if hasattr(tokenizer, "post_quant_conv") and tokenizer.post_quant_conv is not None:
                            checkpoint["vae_post_quant_conv"] = tokenizer.post_quant_conv.state_dict()
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10*args.ckpt_every) == 0 and train_steps > 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                # sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                sim_score = evaluate_ours(ema, tokenizer, diffusion, dataloaders['valid'], rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()

@torch.no_grad
def evaluate_ours(
    model, vae, diffusion, 
    test_dataloaders, 
    rank, 
    batch_size, 
    num_workers, 
    latent_size, 
    device, 
    save_dir, 
    seed, 
    bfloat_enable, 
    num_cond
):  
    num_samples_eval = 10
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    # for x, y, rel_t in loader:
    for x, y, z in tqdm(test_dataloaders, desc=f"Eval [valid]", disable=(rank != 0), total=len(test_dataloaders)):
        x = x['visual']
        x = x.to(device)
        y = y.to(device)
        y = torch.zeros(y.shape[0], y.shape[1] - num_cond, 3).to(device)
        rel_t = torch.zeros(y.shape[0], 1).to(device).flatten(0, 1)
        # rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples_ddim_interp = model_forward_wrapper_ours(
                (model, diffusion, vae), 
                x, 
                y, 
                num_timesteps=None, 
                latent_size=latent_size, 
                device=device, 
                num_cond=num_cond, 
                num_goals=num_goals, 
                rel_t=rel_t,
                num_samples=num_samples_eval,
                noise_interp=True, 
                use_ddim=True, 
            )
            samples_ddim_rand = model_forward_wrapper_ours(
                (model, diffusion, vae), 
                x, 
                y, 
                num_timesteps=None, 
                latent_size=latent_size, 
                device=device, 
                num_cond=num_cond, 
                num_goals=num_goals, 
                rel_t=rel_t,
                num_samples=num_samples_eval,
                noise_interp=False,  
                use_ddim=True,  
            )
            samples_ddpm_rand = model_forward_wrapper_ours(
                (model, diffusion, vae), 
                x, 
                y, 
                num_timesteps=None, 
                latent_size=latent_size, 
                device=device, 
                num_cond=num_cond, 
                num_goals=num_goals, 
                rel_t=rel_t,
                num_samples=num_samples_eval,
                noise_interp=False, 
                use_ddim=False,  
            )
            # samples shape: (B*num_goals, num_samples, 3, H, W)
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            samples_ddim_interp = samples_ddim_interp * 0.5 + 0.5
            samples_ddim_rand = samples_ddim_rand * 0.5 + 0.5
            samples_ddpm_rand = samples_ddpm_rand * 0.5 + 0.5  # normalize all samples for plotting

            samples_for_score = samples_ddpm_rand[:, 0]  # (B*num_goals, 3, 224, 224)
            res = eval_model(x_start_pixels, samples_for_score)
            score += res.sum()
            n_samples += len(res)
        break
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples_ddim_rand.shape[0], 10)):
            # Create a 2-row plot: first row shows [context, GT], second row shows all num_samples predictions
            num_samples_plot = samples_ddim_rand.shape[1]  # number of sample predictions
            _, ax = plt.subplots(4, max(2, num_samples_plot), figsize=(3*max(2, num_samples_plot), 12), dpi=128)
            
            # First row: context (last frame), ground truth, and avg DDIM random sample
            ax[0, 0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[0, 0].set_title('Context')
            ax[0, 0].axis('off')

            ax[0, 1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[0, 1].set_title('Ground Truth')
            ax[0, 1].axis('off')

            # Third image: averaged DDIM random sample
            avg_ddim_rand = samples_ddim_rand[i].mean(dim=0)
            ax[0, 2].imshow((avg_ddim_rand.permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            ax[0, 2].set_title('Avg DDIM Rand')
            ax[0, 2].axis('off')

            # 4th image: averaged DDPM random sample
            avg_ddpm_rand = samples_ddpm_rand[i].mean(dim=0)
            ax[0, 3].imshow((avg_ddpm_rand.permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            ax[0, 3].set_title('Avg DDPM Rand')
            ax[0, 3].axis('off')

            # Hide remaining subplots in first row if num_samples > 4
            for j in range(4, max(4, num_samples_plot)):
                ax[0, j].axis('off')
            
            # Second row: all DDIM interp samples
            for j in range(num_samples_plot):
                ax[1, j].imshow((samples_ddim_interp[i, j].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
                ax[1, j].set_title(f'DDIM interp {j+1}')
                ax[1, j].axis('off')

            # Third row: all DDIM rand samples
            for j in range(num_samples_plot):
                ax[2, j].imshow((samples_ddim_rand[i, j].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
                ax[2, j].set_title(f'DDIM rand {j+1}')
                ax[2, j].axis('off')

            # Fourth row: all DDPM rand samples
            for j in range(num_samples_plot):
                ax[3, j].imshow((samples_ddpm_rand[i, j].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
                ax[3, j].set_title(f'DDPM rand {j+1}')
                ax[3, j].axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/{i}.png', bbox_inches='tight')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score


@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for x, y, rel_t in loader:
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples = model_forward_wrapper((model, diffusion, vae), x, y, num_timesteps=None, latent_size=latent_size, device=device, num_cond=num_cond, num_goals=num_goals, rel_t=rel_t)
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            _, ax = plt.subplots(1,3,dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[2].imshow((samples[i].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
