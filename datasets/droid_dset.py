# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from einops import rearrange
from scipy.spatial.transform import Rotation

from .traj_dset import TrajDataset
# from traj_dset import TrajDataset

_GLOBAL_SEED = 0
logger = getLogger()


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(TrajDataset):
    """DROID trajectory dataset returning full trajectories or random slices."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=1,
        transform=None,
        camera_frame=False,
        n_rollout=None,
        mode="full",
        num_frames=None,
        state_based=False,
    ):
        assert not state_based, "state_based mode is not supported for DROID"
        self.data_path = data_path
        self.frameskip = frameskip
        self.transform = transform
        self.camera_frame = camera_frame
        self.state_based = state_based
        self.mode = mode
        self.num_frames = num_frames
        if mode == "slice":
            assert num_frames is not None, "num_frames must be set when mode='slice'"
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        if n_rollout is not None:
            samples = samples[:n_rollout]
        self.samples = samples

        # DROID: 7-dim state (6 cartesian + 1 gripper), 7-dim action (diffs)
        self.action_dim = 7 * frameskip
        self.state_dim = 7
        self.proprio_dim = 7

        self.state_normalizer = None
        self.action_normalizer = None
        self.proprio_normalizer = None

        # For compatibility with code that accesses .dataset (e.g. TrajSlicerDataset pattern)
        self.dataset = self

    def _resolve_idx(self, idx):
        """Resolve idx to a valid trajectory, skipping corrupted ones deterministically."""
        rng = np.random.RandomState(idx)
        for _ in range(len(self.samples)):
            try:
                path = self.samples[idx]
                metadata = get_json(path)
                if metadata is None:
                    raise Exception(f"No metadata for video {path=}")
                camera_view = self.camera_views[0]
                mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
                vpath = os.path.join(path, "recordings/MP4", mp4_name)
                vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
                T = len(vr)
                min_len = (self.num_frames * self.frameskip + 1) if self.mode == "slice" else 2
                if T < min_len:
                    raise Exception(f"Video too short {vpath=}, {T=}, need {min_len}")
                return idx
            except Exception as e:
                logger.info(f"Skipping corrupted trajectory {idx}: {e}")
                idx = rng.randint(len(self.samples))
        raise RuntimeError("No valid trajectories found in dataset")

    def get_seq_length(self, idx):
        idx = self._resolve_idx(idx)
        path = self.samples[idx]
        metadata = get_json(path)
        camera_view = self.camera_views[0]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        return len(vr) - 1  # last frame dropped so obs aligns with actions

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack(angle_diff, axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]
            theta = pose[3:6]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose.append(transform_to_pose(new_pose_transform))
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path, frame_indices=None):
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath, "r")

        # -- randomly sample a camera view
        camera_view = self.camera_views[int(torch.randint(0, len(self.camera_views), (1,)))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = np.array(trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"])
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        trajectory.close()

        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        T = len(vr)

        states = states[:T]
        extrinsics = extrinsics[:T]

        if frame_indices is not None:
            # Slice mode: only load the requested frames
            # Need one extra frame for action diffs
            extended_indices = np.append(frame_indices, frame_indices[-1] + 1)
            extended_indices = np.clip(extended_indices, 0, T - 1)
            slice_states = states[extended_indices]
            slice_extrinsics = extrinsics[extended_indices]
            if self.camera_frame:
                slice_states = self.transform_frame(slice_states, slice_extrinsics)
            actions = self.poses_to_diffs(slice_states)  # len(frame_indices) actions
            states = slice_states[:-1]
            extrinsics = slice_extrinsics[:-1]
            vr.seek(0)
            buffer = vr.get_batch(frame_indices).numpy()
        else:
            # Full mode: load all frames
            if self.camera_frame:
                states = self.transform_frame(states, extrinsics)
            actions = self.poses_to_diffs(states)  # [T-1, 7]
            # Drop last obs/state so everything is length T-1
            T = T - 1
            states = states[:T]
            extrinsics = extrinsics[:T]
            indices = np.arange(T)
            vr.seek(0)
            buffer = vr.get_batch(indices).numpy()

        return buffer, actions, states, extrinsics

    def _to_tensors(self, buffer, actions, states):
        images = torch.from_numpy(buffer).float() / 255.0
        images = rearrange(images, "T H W C -> T C H W")
        if self.transform is not None:
            images = self.transform(images)
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        return images, actions, states

    def __getitem__(self, idx):
        while True:
            idx = self._resolve_idx(idx)
            path = self.samples[idx]
            try:
                if self.mode == "slice":
                    return self._getitem_slice(path)
                else:
                    return self._getitem_full(path)
            except Exception as e:
                logger.info(f"Encountered exception when loading video {path=} {e=}")
                idx = np.random.randint(len(self))

    def _getitem_full(self, path):
        buffer, actions, states, extrinsics = self.loadvideo_decord(path)
        images, actions, states = self._to_tensors(buffer, actions, states)
        obs = {"visual": images, "proprio": states}
        return obs, actions, states, {}

    def _getitem_slice(self, path):
        # Randomly sample a contiguous slice of num_frames * frameskip frames
        metadata = get_json(path)
        camera_view = self.camera_views[0]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        vlen = len(vr)

        nframes = self.num_frames * self.frameskip
        if vlen < nframes + 1:
            raise Exception(f"Video too short for slice: {vpath=}, {vlen=}, need {nframes + 1}")

        sf = np.random.randint(0, vlen - nframes)
        frame_indices = np.arange(sf, sf + nframes).astype(np.int64)

        buffer, actions, states, extrinsics = self.loadvideo_decord(path, frame_indices=frame_indices)
        images, actions, states = self._to_tensors(buffer, actions, states)

        # Subsample obs/states by frameskip, concat actions (matching TrajSlicerDataset format)
        obs = {"visual": images[::self.frameskip], "proprio": states[::self.frameskip]}
        state = states[::self.frameskip]
        act = rearrange(actions, "(n f) d -> n (f d)", n=self.num_frames)
        return obs, act, state

    def __len__(self):
        return len(self.samples)


def load_droid_slice_train_val(
    transform,
    train_data_path,
    val_data_path,
    camera_views=["left_mp4_path", "right_mp4_path"],
    num_hist=0,
    num_pred=0,
    frameskip=1,
    camera_frame=False,
    n_rollout=None,
    state_based=False,
):
    num_frames = num_hist + num_pred

    train_dset = DROIDVideoDataset(
        data_path=train_data_path,
        camera_views=camera_views,
        transform=transform,
        camera_frame=camera_frame,
        n_rollout=n_rollout,
        mode="full",
        state_based=state_based,
    )
    val_dset = DROIDVideoDataset(
        data_path=val_data_path,
        camera_views=camera_views,
        transform=transform,
        camera_frame=camera_frame,
        n_rollout=n_rollout,
        mode="full",
        state_based=state_based,
    )

    train_slices = DROIDVideoDataset(
        data_path=train_data_path,
        camera_views=camera_views,
        frameskip=frameskip,
        transform=transform,
        camera_frame=camera_frame,
        n_rollout=n_rollout,
        mode="slice",
        num_frames=num_frames,
        state_based=state_based,
    )
    val_slices = DROIDVideoDataset(
        data_path=val_data_path,
        camera_views=camera_views,
        frameskip=frameskip,
        transform=transform,
        camera_frame=camera_frame,
        n_rollout=n_rollout,
        mode="slice",
        num_frames=num_frames,
        state_based=state_based,
    )

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset


if __name__ == "__main__":
    dataset = DROIDVideoDataset(
        # data_path="/checkpoint/amaia/video/basileterv/data/droid/droid_val_paths_cw.csv",
        data_path="/checkpoint/amaia/video/basileterv/data/droid/droid_train_paths_cw.csv",
        camera_views=["left_mp4_path"],
        frameskip=1,
        camera_frame=True,
    )
    print(len(dataset))
    obs, actions, states, info = dataset[0]
    print(f"visual: {obs['visual'].shape}, proprio: {obs['proprio'].shape}")
    print(f"actions: {actions.shape}, states: {states.shape}")
    import pdb; pdb.set_trace()
