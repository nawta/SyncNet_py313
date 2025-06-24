#!/usr/bin/env python3
"""Run SyncNet on multiple videos."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from script directory
sys.path.insert(0, str(Path(__file__).parent.parent / "script"))

from syncnet_pipeline import SyncNetPipeline

def main():
    """Run SyncNet on multiple videos."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <video1> [video2] ...")
        sys.exit(1)
    
    # Initialize pipeline
    pipe = SyncNetPipeline(
        {
            "s3fd_weights": "weights/sfd_face.pth",
            "syncnet_weights": "weights/syncnet_v2.model",
        },
        device="cuda",
    )
    
    results_all = []
    
    for video_path in sys.argv[1:]:
        print(f"\nProcessing: {video_path}")
        
        try:
            # Assume audio is embedded in video
            results = pipe.inference(
                video_path=video_path,
                audio_path=None,  # Extract from video
            )
            
            offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
            
            result_dict = {
                "video": video_path,
                "status": "success",
                "max_confidence": float(max_conf),
                "min_distance": float(min_dist),
                "av_offset": int(offsets[0]) if offsets else None,
                "has_face": has_face
            }
            
            print(f"  Confidence: {max_conf:.3f}, Offset: {offsets[0] if offsets else 'N/A'}")
            
        except Exception as e:
            result_dict = {
                "video": video_path,
                "status": "error",
                "error": str(e)
            }
            print(f"  Error: {e}")
        
        results_all.append(result_dict)
    
    # Save results
    output_file = "batch_results.json"
    with open(output_file, "w") as f:
        json.dump(results_all, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()