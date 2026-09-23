"""
Deepfake Detection Dataset Loader and Video Processing Utilities

This module provides utilities for loading and processing video datasets
for deepfake detection tasks.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import Tuple, List, Optional, Dict


class DeepfakeVideoDataset(Dataset):
    """
    Custom dataset for loading videos for deepfake detection.
    
    Args:
        video_paths: List of tuples (video_path, label)
        transform: Transforms to apply to frames
        frames_per_video: Number of frames to extract per video
        img_size: Target image size (height, width)
    """
    
    def __init__(
        self,
        video_paths: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        frames_per_video: int = 16,
        img_size: Tuple[int, int] = (224, 224)
    ):
        self.video_paths = video_paths
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        
    def __len__(self):
        return len(self.video_paths)
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return frames
        
        # Calculate frame indices to extract
        if total_frames >= self.frames_per_video:
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        else:
            # Repeat frames to match frames_per_video
            indices = []
            for i in range(self.frames_per_video):
                indices.append(i % total_frames)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.img_size)
                frames.append(frame)
        
        cap.release()
        
        # Ensure we always return exactly frames_per_video frames
        while len(frames) < self.frames_per_video:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((*self.img_size, 3), dtype=np.uint8))
        
        # Limit to frames_per_video if we have more
        frames = frames[:self.frames_per_video]
        
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.video_paths[idx]
        frames = self._extract_frames(video_path)
        
        if len(frames) == 0:
            # Return blank frames if video cannot be read
            frames = [np.zeros((*self.img_size, 3), dtype=np.uint8) 
                      for _ in range(self.frames_per_video)]
        
        # Apply transforms
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
            frames = torch.stack(frames)
        else:
            frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        return frames, label


def load_dataset_from_directory(
    dataset_dir: str,
    split: str = 'train',
    train_ratio: float = 0.8
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Load dataset from directory structure.
    
    Directory structure:
        dataset_dir/
            Celeb-real/      (label=1, real)
            Celeb-synthesis/ (label=0, fake)
            YouTube-real/   (label=1, real)
    
    Args:
        dataset_dir: Root directory containing video folders
        split: Not used, kept for compatibility
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_videos, val_videos)
    """
    video_paths = []
    
    # Map folders to labels
    folder_label_map = {
        'Celeb-real': 1,       # Real videos
        'YouTube-real': 1,     # Real videos  
        'Celeb-synthesis': 0,  # Fake/Deepfake videos
    }
    
    for folder_name, label in folder_label_map.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append((os.path.join(folder_path, filename), label))
    
    # Shuffle and split
    random.shuffle(video_paths)
    
    split_idx = int(len(video_paths) * train_ratio)
    train_videos = video_paths[:split_idx]
    val_videos = video_paths[split_idx:]
    
    print(f"Loaded {len(train_videos)} training videos, {len(val_videos)} validation videos")
    print(f"Real videos: {sum(1 for _, l in video_paths if l == 1)}, Fake videos: {sum(1 for _, l in video_paths if l == 0)}")
    
    return train_videos, val_videos


def load_testing_videos(dataset_dir: str, list_file: str) -> List[Tuple[str, int]]:
    """
    Load testing videos from a list file.
    
    Args:
        dataset_dir: Root directory
        list_file: Path to list file with format "label path"
        
    Returns:
        List of tuples (video_path, label)
    """
    video_paths = []
    
    if os.path.exists(list_file):
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = int(parts[0])
                    video_path = os.path.join(dataset_dir, parts[1])
                    video_paths.append((video_path, label))
    else:
        print(f"Testing list file not found: {list_file}")
    
    return video_paths


def get_transforms(img_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or inference.
    
    Args:
        img_size: Target image size
        is_training: Whether this is for training
        
    Returns:
        Compose of transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 8,
    img_size: int = 224,
    frames_per_video: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        dataset_dir: Root dataset directory
        batch_size: Batch size
        img_size: Image size
        frames_per_video: Frames to extract per video
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_videos, val_videos = load_dataset_from_directory(dataset_dir)
    
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    train_dataset = DeepfakeVideoDataset(
        train_videos, 
        transform=train_transform,
        frames_per_video=frames_per_video,
        img_size=(img_size, img_size)
    )
    
    val_dataset = DeepfakeVideoDataset(
        val_videos,
        transform=val_transform,
        frames_per_video=frames_per_video,
        img_size=(img_size, img_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


class VideoFrameExtractor:
    """Utility class for extracting frames from videos."""
    
    def __init__(self, frames_to_extract: int = 10):
        self.frames_to_extract = frames_to_extract
        
    def extract_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extract evenly spaced frames from a video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, self.frames_to_extract, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames
    
    def extract_and_preprocess(self, video_path: str, img_size: int = 224) -> torch.Tensor:
        """Extract frames and preprocess for model input."""
        frames = self.extract_from_video(video_path)
        
        if len(frames) == 0:
            return torch.zeros(1, 3, img_size, img_size)
        
        # Resize and normalize
        processed = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)
            processed.append(frame)
        
        return torch.stack(processed)
