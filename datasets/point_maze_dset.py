import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
from datasets.normalizer import LinearNormalizer, DummyNormalizer, MeanStdNormalizer
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalizer_type: str = "mean_std",  # "combined", "linear", "mean_std", "dummy"
        action_scale=1.0,
        state_based: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalizer_type = normalizer_type
        self.state_based = state_based
        states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path /'seq_lengths.pth')

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
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
        self.linear_state_normalizer.fit(all_valid_proprios)
        self.linear_proprio_normalizer.fit(all_valid_states)

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
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / "obses"
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        if not self.state_based:
            image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
            image = image[frames]  # THWC
            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")
            if self.transform:
                image = self.transform(image)
            obs = {
                "visual": image,
                "proprio": proprio
            }
        else:
            normalized_state = self.normalized_states[idx, frames]
            obs = {
                "visual": normalized_state,
                "proprio": proprio
            } # normalized state as obs, but returned state is not normalized
        return obs, act, state, {} # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
def load_point_maze_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/pusht_dataset',
    normalizer_type="mean_std",
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False,
):
    dset = PointMazeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalizer_type=normalizer_type,
        state_based=state_based,
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
