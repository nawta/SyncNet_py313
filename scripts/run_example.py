#!/usr/bin/env python3
"""Run SyncNet on the example video."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from script directory
sys.path.insert(0, str(Path(__file__).parent.parent / "script"))

from syncnet_pipeline import SyncNetPipeline

def main():
    """Run SyncNet on example video."""
    
    # Initialize pipeline
    pipe = SyncNetPipeline(
        {
            "s3fd_weights": "weights/sfd_face.pth",
            "syncnet_weights": "weights/syncnet_v2.model",
        },
        device="cuda",  # Change to "cpu" if no GPU
    )
    
    # Process video
    results = pipe.inference(
        video_path="example/video.avi",
        audio_path="example/speech.wav",
        cache_dir="example/cache",
    )
    
    # Show results
    offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
    
    print("\n=== SyncNet Results ===")
    print(f"Best confidence:  {max_conf:.3f}")
    print(f"Lowest distance:  {min_dist:.3f}")
    print(f"AV offset:        {offsets[0] if offsets else 'N/A'} frames")
    print(f"Has face:         {has_face}")

if __name__ == "__main__":
    main()