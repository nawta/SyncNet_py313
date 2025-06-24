"""SyncNet inference module with Python 3.13 optimizations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.io import wavfile
import python_speech_features

from syncnet.core.types import SyncResult, AudioData, Frame, EmbeddingBatch
from syncnet.core.compat import load_legacy_model


class InferenceConfig(NamedTuple):
    """Configuration for inference."""
    tmp_dir: Path
    batch_size: int = 20
    vshift: int = 15


class SyncNetInstance(nn.Module):
    """SyncNet inference wrapper with optimizations.
    
    Args:
        model: The SyncNet model to use for inference
        device: Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(
        self, 
        model: nn.Module,
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def calc_pdist(
        self, 
        feat1: torch.Tensor, 
        feat2: torch.Tensor, 
        vshift: int = 10
    ) -> list[torch.Tensor]:
        """Calculate pairwise distances between audio and visual features.
        
        Args:
            feat1: Visual features [num_frames, embedding_dim]
            feat2: Audio features [num_frames, embedding_dim]
            vshift: Maximum shift to consider
            
        Returns:
            List of distance tensors for each frame
        """
        win_size = vshift * 2 + 1
        feat2p = F.pad(feat2, (0, 0, vshift, vshift))
        
        dists = []
        for i in range(len(feat1)):
            dists.append(
                F.pairwise_distance(
                    feat1[[i], :].repeat(win_size, 1),
                    feat2p[i : i + win_size, :]
                )
            )
        
        return dists
    
    def load_video_frames(self, video_path: Path) -> np.ndarray:
        """Load video frames from file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of frames [time, height, width, channels]
        """
        frames = []
        frame_files = sorted(video_path.glob("*.jpg"))
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        return np.stack(frames, axis=0) if frames else np.array([])
    
    def load_audio(self, audio_path: Path) -> tuple[int, AudioData]:
        """Load audio from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (sample_rate, audio_data)
        """
        return wavfile.read(audio_path)
    
    def prepare_video_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Prepare video frames for model input.
        
        Args:
            frames: Video frames [time, height, width, channels]
            
        Returns:
            Video tensor [batch, channels, time, height, width]
        """
        # Reshape to [batch, time, height, width, channels]
        frames = np.expand_dims(frames, axis=0)
        # Transpose to [batch, channels, time, height, width]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))
        return torch.from_numpy(frames.astype(np.float32))
    
    def prepare_audio_tensor(
        self, 
        audio: AudioData, 
        sample_rate: int
    ) -> torch.Tensor:
        """Prepare audio for model input.
        
        Args:
            audio: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            MFCC tensor [batch, 1, features, time]
        """
        mfcc = python_speech_features.mfcc(audio, sample_rate)
        mfcc = np.transpose(mfcc)
        mfcc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        return torch.from_numpy(mfcc.astype(np.float32))
    
    @torch.no_grad()
    def evaluate(self, config: InferenceConfig) -> SyncResult:
        """Evaluate synchronization on audio-video pair.
        
        Args:
            config: Inference configuration
            
        Returns:
            Synchronization result with offset and confidence
        """
        self.model.eval()
        
        # Load inputs
        video_frames = self.load_video_frames(config.tmp_dir)
        audio_data, sample_rate = self.load_audio(config.tmp_dir / "audio.wav")
        
        # Prepare tensors
        video_tensor = self.prepare_video_tensor(video_frames)
        audio_tensor = self.prepare_audio_tensor(audio_data, sample_rate)
        
        # Check length consistency
        video_duration = len(video_frames) / 25.0  # Assuming 25 FPS
        audio_duration = len(audio_data) / sample_rate
        
        if abs(video_duration - audio_duration) > 0.1:
            print(f"WARNING: Audio ({audio_duration:.4f}s) and video "
                  f"({video_duration:.4f}s) lengths differ.")
        
        min_length = min(len(video_frames), len(audio_data) // 640)
        lastframe = min_length - 5
        
        # Extract features in batches
        visual_features = []
        audio_features = []
        
        start_time = time.time()
        
        for i in range(0, lastframe, config.batch_size):
            batch_end = min(lastframe, i + config.batch_size)
            
            # Visual batch
            visual_batch = [
                video_tensor[:, :, j:j+5, :, :]
                for j in range(i, batch_end)
            ]
            visual_input = torch.cat(visual_batch, 0).to(self.device)
            visual_output = self.model.forward_visual(visual_input)
            visual_features.append(visual_output.cpu())
            
            # Audio batch
            audio_batch = [
                audio_tensor[:, :, :, j*4:j*4+20]
                for j in range(i, batch_end)
            ]
            audio_input = torch.cat(audio_batch, 0).to(self.device)
            audio_output = self.model.forward_audio(audio_input)
            audio_features.append(audio_output.cpu())
        
        visual_features = torch.cat(visual_features, 0)
        audio_features = torch.cat(audio_features, 0)
        
        print(f"Feature extraction time: {time.time() - start_time:.3f} sec")
        
        # Compute synchronization
        dists = self.calc_pdist(visual_features, audio_features, vshift=config.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        
        minval, minidx = torch.min(mdist, 0)
        offset = config.vshift - minidx
        conf = torch.median(mdist) - minval
        
        # Frame-wise confidence
        fdist = np.stack([dist[minidx].numpy() for dist in dists])
        fconf = torch.median(mdist).numpy() - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)
        
        print(f"AV offset: {offset:d}")
        print(f"Min dist: {minval:.3f}")
        print(f"Confidence: {conf:.3f}")
        
        return SyncResult(
            offset=int(offset),
            confidence=float(conf),
            dists=fconfm.tolist()
        )
    
    @torch.no_grad()
    def extract_features(
        self, 
        video_path: Path,
        batch_size: int = 20
    ) -> torch.Tensor:
        """Extract visual features from video.
        
        Args:
            video_path: Path to video file
            batch_size: Batch size for processing
            
        Returns:
            Visual features tensor
        """
        self.model.eval()
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return torch.empty(0)
        
        # Prepare tensor
        video_tensor = self.prepare_video_tensor(np.stack(frames))
        
        # Extract features
        lastframe = len(frames) - 4
        features = []
        
        start_time = time.time()
        
        for i in range(0, lastframe, batch_size):
            batch_end = min(lastframe, i + batch_size)
            batch = [
                video_tensor[:, :, j:j+5, :, :]
                for j in range(i, batch_end)
            ]
            batch_input = torch.cat(batch, 0).to(self.device)
            batch_output = self.model.forward_visual_features(batch_input)
            features.append(batch_output.cpu())
        
        features = torch.cat(features, 0)
        
        print(f"Feature extraction time: {time.time() - start_time:.3f} sec")
        
        return features
    
    def load_weights(self, path: Path | str) -> None:
        """Load model weights from file.
        
        Args:
            path: Path to weights file
        """
        try:
            # Try to load as legacy model first
            self.model = load_legacy_model(path, str(self.device))
            self.model.to(self.device)
            print(f"Loaded legacy model from {path}")
        except Exception as e:
            print(f"Failed to load as legacy model: {e}")
            # Try loading as new format
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {path}")