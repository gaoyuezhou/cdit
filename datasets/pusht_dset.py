import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from datasets.normalizer import LinearNormalizer, DummyNormalizer, MeanStdNormalizer
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])

# don't normalize sin/cos states
STATE_MEAN_SIN_COS = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 0, 0, -2.93032027,  2.54307914])
STATE_STD_SIN_COS = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1, 1, 74.84556075, 74.14009094])
# STATE_MEAN_SIN_COS = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 0.1650,   0.6914, -2.93032027,  2.54307914])
# STATE_STD_SIN_COS = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 0.4698,   0.5234, 74.84556075, 74.14009094])

PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalizer_type: str = "mean_std",  # "combined", "linear", "mean_std", "dummy"
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True, # agent's velocity
        state_based: bool = False,
        use_sin_cos: bool = False,
        action_normalizer = None,
        state_normalizer = None,
        proprio_normalizer = None,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalizer_type = normalizer_type
        self.state_based = state_based
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        self.use_sin_cos = use_sin_cos
        if use_sin_cos:
            # replace the last two dimensions with sin/cos of the last two dimensions
            sin_cos = torch.cat(
                [torch.sin(self.states[..., -1]).unsqueeze(-1), 
                 torch.cos(self.states[..., -1]).unsqueeze(-1)], dim=-1
            )
            self.states = torch.cat([self.states[..., :-1], sin_cos], dim=-1)

        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.proprio_normalizer = proprio_normalizer
        if None in (self.action_normalizer, self.state_normalizer, self.proprio_normalizer):
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
        self.linear_proprio_normalizer.fit(all_valid_proprios)
        self.linear_state_normalizer.fit(all_valid_states)

        # initialize mean_std normalizers
        self.mean_std_action_normalizer = MeanStdNormalizer(
            mean=ACTION_MEAN, std=ACTION_STD
        )
        if self.use_sin_cos:
            self.mean_std_state_normalizer = MeanStdNormalizer(
                mean=STATE_MEAN_SIN_COS[:self.state_dim], std=STATE_STD_SIN_COS[:self.state_dim]
            )
        else:
            self.mean_std_state_normalizer = MeanStdNormalizer(
                mean=STATE_MEAN[:self.state_dim], std=STATE_STD[:self.state_dim]
            )
        self.mean_std_proprio_normalizer = MeanStdNormalizer(
            mean=PROPRIO_MEAN[:self.proprio_dim], std=PROPRIO_STD[:self.proprio_dim]
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

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]
        if not self.state_based:
            image = reader.get_batch(frames)  # THWC
            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")
            if self.transform:
                image = self.transform(image)
            obs = {"visual": image, "proprio": proprio}
        else:
            # normalized_state = self.state_normalizer.normalize(state)
            normalized_state = self.normalized_states[idx, frames]
            obs = {
                "visual": normalized_state, 
                "proprio": proprio
            } # normalized state as obs, but returned state is not normalized
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalizer_type="mean_std",
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    state_based=False,
    use_sin_cos=False,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalizer_type=normalizer_type,
        with_velocity=with_velocity,
        state_based=state_based,
        use_sin_cos=use_sin_cos,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalizer_type=normalizer_type,
        with_velocity=with_velocity,
        state_based=state_based,
        use_sin_cos=use_sin_cos,
        action_normalizer=train_dset.action_normalizer,
        state_normalizer=train_dset.state_normalizer,
        proprio_normalizer=train_dset.proprio_normalizer,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset