"""Synchronization analysis components for SyncNet."""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from .base import SyncAnalyzer, AVSyncModel
from .config import SyncConfig
from .exceptions import DimensionError, ProcessingError
from .logging import get_logger
from .types import SyncResult
from .utils import ensure_tensor, Timer, compute_confidence


logger = get_logger(__name__)


class SlidingWindowAnalyzer(SyncAnalyzer):
    """Sliding window synchronization analyzer."""
    
    def __init__(
        self, 
        model: AVSyncModel, 
        config: SyncConfig,
        device: str = "cuda"
    ):
        """Initialize analyzer.
        
        Args:
            model: Audio-visual synchronization model
            config: Synchronization configuration
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def analyze(
        self, 
        audio_features: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> SyncResult:
        """Analyze audio-visual synchronization.
        
        Args:
            audio_features: Audio feature tensor [N_audio, 1, H, W]
            visual_features: Visual feature tensor [N_visual, C, T, H, W]
            
        Returns:
            Synchronization analysis result
        """
        # Validate inputs
        if audio_features.dim() != 4:
            raise DimensionError(
                "audio_features", 
                audio_features.shape, 
                "[N, 1, H, W]"
            )
        
        if visual_features.dim() != 5:
            raise DimensionError(
                "visual_features", 
                visual_features.shape, 
                "[N, C, T, H, W]"
            )
        
        n_audio = audio_features.shape[0]
        n_visual = visual_features.shape[0]
        
        logger.debug(
            f"Analyzing sync: audio={n_audio} windows, "
            f"visual={n_visual} sequences"
        )
        
        # Move to device
        audio_features = audio_features.to(self.device)
        visual_features = visual_features.to(self.device)
        
        # Compute embeddings with sliding window
        dists = []
        
        with torch.no_grad():
            with Timer("Computing sync distances", logger):
                for i in range(-self.config.vshift, self.config.vshift + 1):
                    # Determine valid range
                    if i < 0:
                        # Audio leads video
                        audio_start = -i
                        audio_end = min(n_audio, n_visual - i)
                        visual_start = 0
                        visual_end = audio_end - audio_start
                    else:
                        # Video leads audio
                        audio_start = 0
                        audio_end = min(n_audio, n_visual - i)
                        visual_start = i
                        visual_end = visual_start + (audio_end - audio_start)
                    
                    if audio_end <= audio_start:
                        continue
                    
                    # Process in batches
                    batch_dists = []
                    
                    for batch_start in range(
                        0, 
                        audio_end - audio_start, 
                        self.config.batch_size
                    ):
                        batch_end = min(
                            batch_start + self.config.batch_size,
                            audio_end - audio_start
                        )
                        
                        # Get batch slices
                        audio_batch = audio_features[
                            audio_start + batch_start:audio_start + batch_end
                        ]
                        visual_batch = visual_features[
                            visual_start + batch_start:visual_start + batch_end
                        ]
                        
                        # Compute embeddings
                        audio_emb = self.model.forward_audio(audio_batch)
                        visual_emb = self.model.forward_visual(visual_batch)
                        
                        # Compute distances
                        dist = F.pairwise_distance(audio_emb, visual_emb)
                        batch_dists.extend(dist.cpu().numpy())
                    
                    # Average distance for this offset
                    if batch_dists:
                        avg_dist = np.mean(batch_dists)
                        dists.append(avg_dist)
                    else:
                        dists.append(float('inf'))
        
        # Find best offset
        if not dists:
            raise ProcessingError("No valid sync windows found")
        
        min_idx = np.argmin(dists)
        offset = min_idx - self.config.vshift
        confidence = compute_confidence(dists, method="min_distance")
        
        logger.info(
            f"Sync analysis complete: offset={offset}, "
            f"confidence={confidence:.3f}"
        )
        
        return SyncResult(
            offset=int(offset),
            confidence=float(confidence),
            dists=dists
        )


class OptimizedSyncAnalyzer(SlidingWindowAnalyzer):
    """Optimized synchronization analyzer with parallel processing."""
    
    def __init__(
        self,
        model: AVSyncModel,
        config: SyncConfig,
        device: str = "cuda",
        num_workers: int = 2
    ):
        """Initialize optimized analyzer.
        
        Args:
            model: Audio-visual synchronization model
            config: Synchronization configuration
            device: Device for computation
            num_workers: Number of parallel workers
        """
        super().__init__(model, config, device)
        self.num_workers = num_workers
    
    def analyze(
        self, 
        audio_features: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> SyncResult:
        """Analyze with optimized parallel processing.
        
        Args:
            audio_features: Audio feature tensor
            visual_features: Visual feature tensor
            
        Returns:
            Synchronization analysis result
        """
        # Pre-compute all embeddings
        n_audio = audio_features.shape[0]
        n_visual = visual_features.shape[0]
        
        audio_features = audio_features.to(self.device)
        visual_features = visual_features.to(self.device)
        
        with torch.no_grad():
            # Compute all embeddings at once
            with Timer("Computing all embeddings", logger):
                audio_embeddings = []
                visual_embeddings = []
                
                # Process audio in batches
                for i in range(0, n_audio, self.config.batch_size):
                    batch = audio_features[i:i + self.config.batch_size]
                    emb = self.model.forward_audio(batch)
                    audio_embeddings.append(emb)
                
                audio_embeddings = torch.cat(audio_embeddings, dim=0)
                
                # Process visual in batches
                for i in range(0, n_visual, self.config.batch_size):
                    batch = visual_features[i:i + self.config.batch_size]
                    emb = self.model.forward_visual(batch)
                    visual_embeddings.append(emb)
                
                visual_embeddings = torch.cat(visual_embeddings, dim=0)
            
            # Compute distances for all offsets in parallel
            def compute_offset_distance(offset: int) -> float:
                if offset < 0:
                    # Audio leads video
                    audio_start = -offset
                    audio_end = min(n_audio, n_visual - offset)
                    visual_start = 0
                else:
                    # Video leads audio
                    audio_start = 0
                    audio_end = min(n_audio, n_visual - offset)
                    visual_start = offset
                
                if audio_end <= audio_start:
                    return float('inf')
                
                # Slice embeddings
                audio_slice = audio_embeddings[audio_start:audio_end]
                visual_slice = visual_embeddings[
                    visual_start:visual_start + (audio_end - audio_start)
                ]
                
                # Compute pairwise distances
                dists = F.pairwise_distance(audio_slice, visual_slice)
                return float(dists.mean().cpu())
            
            # Parallel computation
            offsets = list(range(-self.config.vshift, self.config.vshift + 1))
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                dists = list(executor.map(compute_offset_distance, offsets))
        
        # Find best offset
        min_idx = np.argmin(dists)
        offset = offsets[min_idx]
        confidence = compute_confidence(dists, method="min_distance")
        
        return SyncResult(
            offset=int(offset),
            confidence=float(confidence),
            dists=dists
        )


class CachedSyncAnalyzer(OptimizedSyncAnalyzer):
    """Sync analyzer with embedding caching for repeated analysis."""
    
    def __init__(
        self,
        model: AVSyncModel,
        config: SyncConfig,
        device: str = "cuda",
        num_workers: int = 2,
        cache_size: int = 100
    ):
        """Initialize cached analyzer.
        
        Args:
            model: Audio-visual synchronization model
            config: Synchronization configuration
            device: Device for computation
            num_workers: Number of parallel workers
            cache_size: Maximum number of cached embeddings
        """
        super().__init__(model, config, device, num_workers)
        self.cache_size = cache_size
        self.audio_cache: dict[int, torch.Tensor] = {}
        self.visual_cache: dict[int, torch.Tensor] = {}
    
    def _get_audio_embeddings(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Get audio embeddings with caching."""
        cache_key = hash(features.cpu().numpy().tobytes())
        
        if cache_key in self.audio_cache:
            logger.debug("Using cached audio embeddings")
            return self.audio_cache[cache_key]
        
        # Compute embeddings
        embeddings = []
        for i in range(0, features.shape[0], self.config.batch_size):
            batch = features[i:i + self.config.batch_size]
            emb = self.model.forward_audio(batch)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # Update cache
        if len(self.audio_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        
        self.audio_cache[cache_key] = embeddings
        
        return embeddings
    
    def _get_visual_embeddings(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Get visual embeddings with caching."""
        cache_key = hash(features.cpu().numpy().tobytes())
        
        if cache_key in self.visual_cache:
            logger.debug("Using cached visual embeddings")
            return self.visual_cache[cache_key]
        
        # Compute embeddings
        embeddings = []
        for i in range(0, features.shape[0], self.config.batch_size):
            batch = features[i:i + self.config.batch_size]
            emb = self.model.forward_visual(batch)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # Update cache
        if len(self.visual_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.visual_cache))
            del self.visual_cache[oldest_key]
        
        self.visual_cache[cache_key] = embeddings
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear embedding caches."""
        self.audio_cache.clear()
        self.visual_cache.clear()
        logger.debug("Cleared embedding caches")


def create_sync_analyzer(
    model: AVSyncModel,
    config: Optional[SyncConfig] = None,
    device: str = "cuda",
    optimized: bool = True,
    cached: bool = False,
    num_workers: int = 2
) -> SyncAnalyzer:
    """Factory function to create sync analyzer.
    
    Args:
        model: Audio-visual synchronization model
        config: Sync configuration (uses defaults if None)
        device: Device for computation
        optimized: Whether to use optimized analyzer
        cached: Whether to use caching
        num_workers: Number of parallel workers
        
    Returns:
        Sync analyzer instance
    """
    if config is None:
        config = SyncConfig()
    
    if cached:
        return CachedSyncAnalyzer(model, config, device, num_workers)
    elif optimized:
        return OptimizedSyncAnalyzer(model, config, device, num_workers)
    else:
        return SlidingWindowAnalyzer(model, config, device)