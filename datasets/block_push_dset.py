import os
import torch
import einops
import numpy as np
from pathlib import Path
from typing import Optional
from datasets.traj_dset import TrajDataset, get_train_val_sliced
from typing import Optional, Callable, Any
from datasets.normalizer import LinearNormalizer, DummyNormalizer, MeanStdNormalizer


class PushMultiviewTrajectoryDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalizer_type: str = "mean_std",  # "combined", "linear", "mean_std", "dummy"
        action_scale=0.03,
        onehot_goals=False,
        prefetch: bool = False,
        state_based: bool = True,
        multiview: bool = False,
    ):  
        self.n_rollout = n_rollout
        self.transform = transform
        self.normalizer_type = normalizer_type
        self.data_directory = Path(data_path)
        self.state_based = state_based
        self.multiview = multiview # not fully integrated yet

        self.states = np.load(self.data_directory / "multimodal_push_observations.npy")
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")

        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.masks = self.masks[:n]

        self.states = torch.from_numpy(self.states).float()
        self.states = self.states[..., :8]
        self.actions = torch.from_numpy(self.actions).float() / action_scale
        self.masks = torch.from_numpy(self.masks).bool()
        self.proprios = self.states[..., 6:].clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        self.prefetch = prefetch
        if self.prefetch:
            self.obses = []
            for i in range(n):
                vid_path = self.data_directory / "obs_multiview" / f"{i:03d}.pth"
                self.obses.append(torch.load(vid_path))
        self.onehot_goals = onehot_goals
        if self.onehot_goals:
            self.goals = torch.load(self.data_directory / "onehot_goals.pth").float()
            self.goals = self.goals[:n]

        self.seq_lengths = self.masks.sum(dim=1).long()
        
        self.initialize_normalizers()
        self.actions = self.action_normalizer.normalize(self.actions)
        self.proprios = self.proprio_normalizer.normalize(self.proprios)

        self.normalized_states = self.state_normalizer.normalize(self.states.clone())
    
    def initialize_normalizers(self):
        # initialize linear normalizers 
        self.linear_action_normalizer = LinearNormalizer()
        self.linear_state_normalizer = LinearNormalizer()
        self.linear_proprio_normalizer = LinearNormalizer()
        
        valid_actions = []
        valid_proprios = []
        valid_states = []

        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            valid_actions.append(self.actions[i, :T, :])
            valid_proprios.append(self.proprios[i, :T, :])
            valid_states.append(self.states[i, :T, :])

        all_valid_actions = torch.cat(valid_actions, dim=0)
        all_valid_proprios = torch.cat(valid_proprios, dim=0)
        all_valid_states = torch.cat(valid_states, dim=0)

        self.linear_action_normalizer.fit(all_valid_actions)
        self.linear_state_normalizer.fit(all_valid_states)
        self.linear_proprio_normalizer.fit(all_valid_proprios)

        # initialize mean_std normalizers
        action_mean, action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
        state_mean, state_std = self.get_data_mean_std(self.states, self.seq_lengths)
        proprio_mean, proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)

        self.mean_std_action_normalizer = MeanStdNormalizer(
            mean=action_mean, std=action_std
        )
        self.mean_std_state_normalizer = MeanStdNormalizer(
            mean=state_mean, std=state_std
        )
        self.mean_std_proprio_normalizer = MeanStdNormalizer(
            mean=proprio_mean, std=proprio_std
        )

        if self.normalizer_type == "dummy":
            self.action_normalizer = DummyNormalizer()
            self.state_normalizer = DummyNormalizer()
            self.proprio_normalizer = DummyNormalizer()
        elif self.normalizer_type == "mean_std":
            self.action_normalizer = self.mean_std_action_normalizer
            self.state_normalizer = self.mean_std_state_normalizer
            self.proprio_normalizer = self.mean_std_proprio_normalizer
        elif self.normalizer_type == "linear":
            self.action_normalizer = self.linear_action_normalizer
            self.state_normalizer = self.linear_state_normalizer
            self.proprio_normalizer = self.linear_proprio_normalizer
        elif self.normalizer_type == "combined":
            self.action_normalizer = self.mean_std_action_normalizer
            self.state_normalizer = self.linear_state_normalizer
            self.proprio_normalizer = self.linear_proprio_normalizer
        else:
            raise ValueError(f"Unknown normalizer type: {self.normalizer_type}")

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        mask = self.masks[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        if not self.state_based:
            if self.prefetch:
                obs_images = self.obses[idx][frames]
            else:
                obs_images = torch.load(self.data_directory / "obs_multiview" / f"{idx:03d}.pth")[
                    frames
                ]
            obs_images = obs_images / 255.0 # TVHWC
            if self.multiview: # missing transform
                obs = {
                    "visual": obs_images,
                    "proprio": proprio,
                }
            else:
                if self.transform:
                    image = self.transform(image)
                obs = {
                    "visual": obs_images[:, 0],
                    "proprio": proprio,
                }
        else:
            normalized_state = self.normalized_states[idx, frames]
            obs = {
                "visual": normalized_state,
                "proprio": proprio,
            } 
        if self.onehot_goals:
            goal = self.goals[idx, frames]
            return obs, act, state, {} # {"goal":goal, "mask":mask}
        else: 
            return obs, act, state, {} # {"mask":mask}

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return self.get_frames(idx, range(T))

    def __len__(self):
        return len(self.states)

def load_block_push_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/pusht_dataset',
    normalizer_type="mean_std",
    split_ratio=0.8,
    num_hist=1,
    num_pred=1,
    frameskip=1,
    state_based=True,
    multiview=False,
):
    dset = PushMultiviewTrajectoryDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalizer_type=normalizer_type,
        state_based=state_based,
        multiview=multiview,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
