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

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vipe.streams.base import VideoFrame, VideoStream


class RawImgStream(VideoStream):
    """
    A video stream from a directory containing raw jpg or png files, using PIL.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None) -> None:
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

        # Read all images
        self.frames_rgb = []
        for fpath in frames_paths:
            img = Image.open(fpath).convert("RGB")
            img = np.array(img)
            self.frames_rgb.append(img)

        # Read metadata from the first image
        self._height, self._width = self.frames_rgb[0].shape[:2]
        _fps = 30.0  # Assume a default fps of 30.0
        _n_frames = len(self.frames_rgb)

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

        return VideoFrame(raw_frame_idx=self.current_frame_idx, rgb=frame_rgb)



