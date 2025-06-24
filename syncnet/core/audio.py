"""Audio processing components for SyncNet."""

from pathlib import Path
from typing import Optional
import numpy as np
import subprocess
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
import torch

from .base import AudioProcessor
from .config import AudioConfig
from .exceptions import AudioProcessingError, DependencyError
from .logging import get_logger
from .utils import ensure_tensor, Timer


logger = get_logger(__name__)


class MFCCAudioProcessor(AudioProcessor):
    """MFCC-based audio processor for SyncNet."""
    
    def __init__(self, config: AudioConfig):
        """Initialize audio processor.
        
        Args:
            config: Audio processing configuration
        """
        self.config = config
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                raise DependencyError("ffmpeg", "sudo apt-get install ffmpeg")
        except FileNotFoundError:
            raise DependencyError("ffmpeg", "sudo apt-get install ffmpeg")
    
    def extract_features(self, audio_path: Path | str) -> np.ndarray:
        """Extract MFCC features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC features array of shape [time, n_mfcc]
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioProcessingError(audio_path, "Audio file not found")
        
        with Timer(f"Extracting audio from {audio_path.name}", logger):
            # Extract audio to temporary WAV file
            wav_path = audio_path.with_suffix('.wav')
            
            try:
                # Convert to WAV with specified sample rate
                cmd = [
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ar", str(self.config.sample_rate),
                    "-ac", "1",  # Mono
                    "-f", "wav",
                    str(wav_path)
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode != 0:
                    raise AudioProcessingError(
                        audio_path, 
                        f"FFmpeg conversion failed: {result.stderr}"
                    )
                
                # Read WAV file
                sample_rate, audio_data = wavfile.read(wav_path)
                
                if sample_rate != self.config.sample_rate:
                    logger.warning(
                        f"Sample rate mismatch: expected {self.config.sample_rate}, "
                        f"got {sample_rate}"
                    )
                
                # Extract MFCC features
                features = self.preprocess(audio_data, sample_rate)
                
                return features
                
            finally:
                # Clean up temporary file
                if wav_path.exists() and wav_path != audio_path:
                    wav_path.unlink()
    
    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess raw audio data to extract MFCC features.
        
        Args:
            audio: Raw audio samples
            sample_rate: Audio sample rate
            
        Returns:
            MFCC features of shape [time, n_mfcc]
        """
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1]
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Extract MFCC features
        with Timer("MFCC extraction", logger):
            features = mfcc(
                audio,
                samplerate=sample_rate,
                numcep=self.config.n_mfcc,
                winlen=self.config.window_size,
                winstep=self.config.hop_size,
                nfft=self.config.n_fft
            )
        
        logger.debug(f"Extracted MFCC features: shape={features.shape}")
        
        return features
    
    def prepare_batch(
        self, 
        features: np.ndarray, 
        window_size: int,
        hop_size: int
    ) -> torch.Tensor:
        """Prepare audio features for batch processing.
        
        Args:
            features: MFCC features [time, n_mfcc]
            window_size: Window size in frames
            hop_size: Hop size in frames
            
        Returns:
            Batched tensor [n_windows, 1, n_mfcc, window_size]
        """
        n_frames, n_mfcc = features.shape
        
        if n_frames < window_size:
            raise AudioProcessingError(
                "features", 
                f"Not enough frames ({n_frames}) for window size ({window_size})"
            )
        
        # Create sliding windows
        windows = []
        for i in range(0, n_frames - window_size + 1, hop_size):
            window = features[i:i + window_size, :].T  # [n_mfcc, window_size]
            windows.append(window)
        
        # Stack into batch
        batch = np.stack(windows)  # [n_windows, n_mfcc, window_size]
        batch = batch[:, np.newaxis, :, :]  # [n_windows, 1, n_mfcc, window_size]
        
        return ensure_tensor(batch)


class StreamingAudioProcessor(MFCCAudioProcessor):
    """Streaming audio processor for real-time processing."""
    
    def __init__(self, config: AudioConfig, buffer_size: int = 10):
        """Initialize streaming processor.
        
        Args:
            config: Audio configuration
            buffer_size: Buffer size in seconds
        """
        super().__init__(config)
        self.buffer_size = buffer_size
        self.buffer_samples = int(buffer_size * config.sample_rate)
        self.buffer: Optional[np.ndarray] = None
        self.buffer_pos = 0
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a chunk of audio data.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            MFCC features if enough data is buffered, None otherwise
        """
        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        
        # Add chunk to buffer
        chunk_size = len(audio_chunk)
        if self.buffer_pos + chunk_size > self.buffer_samples:
            # Buffer is full, process it
            remaining = self.buffer_samples - self.buffer_pos
            self.buffer[self.buffer_pos:] = audio_chunk[:remaining]
            
            # Process buffer
            features = self.preprocess(self.buffer, self.config.sample_rate)
            
            # Reset buffer with remaining samples
            self.buffer[:chunk_size - remaining] = audio_chunk[remaining:]
            self.buffer_pos = chunk_size - remaining
            
            return features
        else:
            # Add to buffer
            self.buffer[self.buffer_pos:self.buffer_pos + chunk_size] = audio_chunk
            self.buffer_pos += chunk_size
            return None
    
    def flush(self) -> Optional[np.ndarray]:
        """Process any remaining buffered audio.
        
        Returns:
            MFCC features if buffer has data, None otherwise
        """
        if self.buffer is not None and self.buffer_pos > 0:
            features = self.preprocess(
                self.buffer[:self.buffer_pos], 
                self.config.sample_rate
            )
            self.buffer_pos = 0
            return features
        return None


def create_audio_processor(
    config: Optional[AudioConfig] = None,
    streaming: bool = False
) -> AudioProcessor:
    """Factory function to create audio processor.
    
    Args:
        config: Audio configuration (uses defaults if None)
        streaming: Whether to create streaming processor
        
    Returns:
        Audio processor instance
    """
    if config is None:
        config = AudioConfig()
    
    if streaming:
        return StreamingAudioProcessor(config)
    else:
        return MFCCAudioProcessor(config)