"""
Dataset module for loading and pre-processing video data for classification tasks.
"""
import gc
import logging
import os
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from typing import Any, Callable, Optional, Type

import av
import numpy as np
import torch
import torch.nn.functional as F
from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset, RandomSampler
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

NUM_LABELS = 32

logger = logging.getLogger(__name__)


class SequentialRepeatedVideoSampler(torch.utils.data.Sampler):
    """
    Sequential-style sampler that repeats each video index consecutively.
    For each video the number of repeats = ceil(video_duration / clip_duration).
    Falls back to 1 repeat when duration cannot be determined.

    Assumes labeled_videos is a list of tuples (video_path, info_dict) where info_dict contains 'duration' key.

    Usage:
        sampler = SequentialVideoSampler(labeled_video_paths, clip_duration)
        val_dataset = VideoDataset(..., video_sampler=sampler, ...)
    """
    def __init__(self, labeled_videos: list[tuple[str, dict[str, Any]]], clip_duration: float = 2.0):
        super().__init__()
        self._labeled_videos = labeled_videos
        self._clip_duration = float(clip_duration)
        self._indices = self._build_index_list()

    def _build_index_list(self):
        indices = []
        for idx, (_, info_dict) in enumerate(self._labeled_videos):
            duration = info_dict.get("duration", 0.0) if isinstance(info_dict, dict) else 0.0
            if duration <= 0 or self._clip_duration <= 0:
                repeats = 1
            else:
                repeats = int(ceil(duration / self._clip_duration))
                repeats = max(1, repeats)
            indices.extend([idx] * repeats)
        return indices

    def __iter__(self):
        # Return indices sequentially (no shuffling)
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


# Modified from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py
class VideoDataset(IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)

    Note that the label is assigned aftet the clip is sampled
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: list[tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        fn2labels: dict,
        split: str = "train",
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, decode audio from video.

            decode_video (bool): If True, decode video frames from a video container.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        # if video_sampler == torch.utils.data.RandomSampler:
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos, clip_duration=self._clip_sampler._clip_duration)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._last_clip_end_time = Fraction(0, 1)
        self.video_path_handler = VideoPathHandler()

        # the procedure to load labels is different to LabeledVideoDataset from pytorchvideo
        self.fn2labels = fn2labels

        assert split in ["train", "val"], "split must be one of 'train', 'val'"
        self._split = split

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def _get_labels(self, video_info: dict[str, Any]):
        start, end, rate, name = video_info["clip_start"], video_info["clip_end"], video_info["rate"], video_info["video_name"]
        start_frame, end_frame = floor(start * rate), floor(end * rate)
        frame_labels = torch.from_numpy(self.fn2labels[Path(name).stem][start_frame:end_frame])
        label = F.one_hot(frame_labels, NUM_LABELS).sum(dim=0).argmax()
        return label

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                # NOTE: During training, StopIteration would never be raised with this dataset
                # need to manually handle epoch end in training and testing loops.
                if self._split == "val":
                    video_index = next(self._video_sampler_iter)
                else:
                    try:
                        video_index = next(self._video_sampler_iter)
                    except StopIteration:
                        self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))
                        video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.warning(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    logger.error("Video load exception")
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, video.duration, info_dict)

            if isinstance(clip_start, list):  # multi-clip in each sample
                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  # single clip case
                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._last_clip_end_time = clip_end

            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._last_clip_end_time = None
                self._clip_sampler.reset()

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.error(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            info_dict["label"] = self._get_labels({
                "clip_start": clip_start,
                "clip_end": clip_end,
                "video_name": video.name,
                # We assume info_dict contains 'rate' as Fraction
                **info_dict
            })

            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **{k: float(v) if isinstance(v, Fraction) else v for k, v in info_dict.items()},
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    ALPHA = 4

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def create_dataset(batch_size: int = 4, seed: int = 42, val_size: float = 0.2):
    # Pre-defined parameters
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30

    # This is in seconds
    clip_duration = (num_frames * sampling_rate) / frames_per_second

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )

    # Iterate through all videos and collect info dicts
    video_dir = Path("dataset") / "videos" / "mp4"
    labeled_video_paths = []
    for video_fn in os.listdir(video_dir):
        container = av.open(video_dir / video_fn)
        video_stream = container.streams.video[0]

        # Duration is in time_base units, convert to seconds
        video_duration_seconds = float(video_stream.duration * video_stream.time_base)

        # Labels to be assigned on the fly during dataset loading
        labeled_video_paths.append((str(video_dir / video_fn), {'label': None, 'duration': video_duration_seconds, "rate": video_stream.base_rate}))
        container.close()

        logger.info(f"Read video {video_fn}, duration: {video_duration_seconds} seconds, rate: {video_stream.base_rate}")

    train_video_paths, val_video_paths = train_test_split(labeled_video_paths, test_size=val_size, random_state=seed)

    # Load pre-processed labels
    filename_to_labels = dict()
    with np.load(Path("dataset") / "annotations_processed.npz") as npz_file:
        for key in npz_file.files:
            filename_to_labels[key] = npz_file[key]

    # Create datasets and dataloaders
    train_dataset = VideoDataset(
        labeled_video_paths=train_video_paths,
        clip_sampler=make_clip_sampler("random", clip_duration),
        fn2labels=filename_to_labels,
        video_sampler=RandomSampler,
        transform=transform
    )

    val_dataset = VideoDataset(
        labeled_video_paths=val_video_paths,
        clip_sampler=make_clip_sampler("uniform", clip_duration),
        fn2labels=filename_to_labels,
        video_sampler=SequentialRepeatedVideoSampler,
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=8)

    logger.info("Created train and val datasets")

    return train_dataloader, val_dataloader
