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

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.posed_raw_img_stream import PosedRawImgStream
from vipe.streams.raw_img_stream import RawImgStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--cameras", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to camera files directory (default: None)",
    default=None
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
def infer(video: Path, cameras: Path, output: Path, pipeline: str, visualize: bool):
    """Run inference on a video file."""

    print(f"video path: {video}")
    print(f"camera path: {cameras}")
    print(f"output path: {output}")
    print(f"pipeline: {pipeline}")
    print(f"visualize: {visualize}")

    logger = configure_logging()

    overrides = [
        f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"
    ]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {video}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
    print(f"video path: {video}")

    # check if ends with .mp4
    if video.suffix.lower() == ".mp4":
        raw_stream = RawMp4Stream(video)
    # else check if it is a directory
    elif video.is_dir():
        if cameras is not None and cameras.is_dir():
            raw_stream = PosedRawImgStream(video, cameras)
        else:
            raw_stream = RawImgStream(video)
    else:
        raise ValueError(f"No supported video format found for {video}")

    video_stream = ProcessedVideoStream(raw_stream, []).cache(desc="Reading video stream")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
