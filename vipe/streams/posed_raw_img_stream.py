# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vipe.ext.lietorch import SE3
from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils.geometry import so3_matrix_to_quat


class PosedRawImgStream(VideoStream):
    """
    A video stream from a directory containing raw jpg or png files, using PIL.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, cameras: Path, seek_range: range | None = None, name: str | None = None) -> None:
        super().__init__()
        
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.stem

        # List all files in the directory
        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")

        # check if contains .jpg or .png files
        frames_paths = sorted([f for f in path.iterdir() if f.suffix.lower() in [".jpg", ".png"]])
        if len(frames_paths) == 0:
            raise ValueError(f"No image files found in directory {path}")
        # print("frames_paths:", frames_paths)

        # get files extension from first frame path
        _frame_ext = frames_paths[0].suffix.lower()
        # 
        base_frames_path = frames_paths[0].parent

        # list all files in cameras
        camera_paths = sorted([f for f in cameras.iterdir() if f.suffix.lower() in [".json"]])
        if len(camera_paths) == 0:
            raise ValueError(f"No camera files found in directory {cameras}")
        # print("camera_paths:", camera_paths)

        # PROBLEM: some scenes have less camera files than rgb files
        # use camera files names to load rgb files

        # Read all images
        self.frames_rgb = []
        # for fpath in frames_paths:
        for fpath in camera_paths:
            # fpath_ = fpath
            fpath_ = base_frames_path / Path(str(fpath.stem) + _frame_ext)
            img = Image.open(fpath_).convert("RGB")
            img = np.array(img)
            self.frames_rgb.append(img)

        # Read metadata from the first image
        self._height, self._width = self.frames_rgb[0].shape[:2]
        _fps = 30.0  # Assume a default fps of 30.0
        _n_frames = len(self.frames_rgb)

        self.poses = []
        self.intrinsics = []

        if True:
            for fpath in camera_paths:
                # load json
                with open(fpath, "r") as f:
                    camera_data = json.load(f)
                # process camera_data
                # print(f"Loaded camera data from {fpath}: {camera_data}")
                orientation = np.array(camera_data.get("orientation"))
                position = np.array(camera_data.get("position"))
                pose = np.zeros((4, 4))
                pose[:3, :3] = orientation
                pose[:3, 3] = position
                focal_length = camera_data.get("focal_length")
                principal_point = camera_data.get("principal_point")
                image_size = camera_data.get("image_size")
                intrinsics = np.zeros((3, 3))
                intrinsics[0, 0] = focal_length
                intrinsics[1, 1] = focal_length
                intrinsics[0, 2] = principal_point[0]
                intrinsics[1, 2] = principal_point[1]
                # print("orientation", orientation)
                # print("position", position)
                # print("image_size", image_size)
                # print("intrinsics", intrinsics)
                
                s_width, s_height = self._width / image_size[0], self._height / image_size[1]
                # print("og width, height", self._width, self._height)
                # print("K width, height", image_size[0], image_size[1])
                # print("scales", s_width, s_height)
                intrinsics[0, :] *= s_width
                intrinsics[1, :] *= s_height
                # exit(0)
                
                # print("intrinsics", intrinsics)
                # exit(0)
                self.poses.append(pose)
                self.intrinsics.append(intrinsics)
        else:
            # Handle cases where camera data is not available
            self.poses = [None] * len(self.frames_rgb)
            self.intrinsics = [None] * len(self.frames_rgb)

        assert len(self.poses) == len(self.frames_rgb), f"Number of poses {len(self.poses)} does not match number of frames {len(self.frames_rgb)}"

        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step
        self._fps = _fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        # self.vcap = cv2.VideoCapture(self.path)
        self.current_frame_idx = -1
        return self

    def __next__(self) -> VideoFrame:
        while True: 
            self.current_frame_idx += 1

            # if not ret:
            #     self.vcap.release()
            #     raise StopIteration

            if self.current_frame_idx >= self.end:
                # self.vcap.release()
                raise StopIteration

            if self.current_frame_idx < self.start:
                continue

            if (self.current_frame_idx - self.start) % self.step == 0:
                break
        
        frame_rgb = self.frames_rgb[self.current_frame_idx]
        frame_rgb = torch.as_tensor(frame_rgb).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        c2w = self.poses[self.current_frame_idx]
        if c2w is not None:
            c2w = torch.as_tensor(c2w).float().cuda()
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            quat = so3_matrix_to_quat(R)
            # print("t:", t)
            # print("quat:", quat)
            # exit(0)

            # TODO: convert c2w (4, 4) to SE3 (translation, quaternion xyzw)
            quaternion = quat # torch.as_tensor([0.0, 0.0, 0.0, 1.0])
            translation = t # torch.as_tensor([0.0, 0.0, 0.0])
            data = torch.cat([translation, quaternion], -1)
            pose: SE3 = SE3(data)
        else:
            pose = None

        intrinsics = self.intrinsics[self.current_frame_idx]
        if intrinsics is not None:
            intrinsics = torch.as_tensor(intrinsics).float().cuda()

        return VideoFrame(
            raw_frame_idx=self.current_frame_idx,
            rgb=frame_rgb,
            pose=pose,
            intrinsics=intrinsics,
        )



