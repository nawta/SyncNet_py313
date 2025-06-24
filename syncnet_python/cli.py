#!/usr/bin/env python3
"""Command-line interface for SyncNet Python."""

import argparse
import json
import sys
from pathlib import Path

from .syncnet_pipeline import SyncNetPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SyncNet: Audio-visual synchronization detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  syncnet-python video.mp4
  
  # Process multiple videos
  syncnet-python video1.mp4 video2.mp4 --output results.json
  
  # Use CPU instead of GPU
  syncnet-python video.mp4 --device cpu
        """
    )
    
    parser.add_argument(
        'videos',
        nargs='+',
        help='Video file(s) to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file for results'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run models on (default: cuda)'
    )
    
    parser.add_argument(
        '--s3fd-weights',
        default='weights/sfd_face.pth',
        help='Path to S3FD weights'
    )
    
    parser.add_argument(
        '--syncnet-weights',
        default='weights/syncnet_v2.model',
        help='Path to SyncNet weights'
    )
    
    parser.add_argument(
        '--cache-dir',
        help='Directory for temporary files'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = SyncNetPipeline(
            {
                "s3fd_weights": args.s3fd_weights,
                "syncnet_weights": args.syncnet_weights,
            },
            device=args.device,
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        return 1
    
    results_all = []
    
    # Process videos
    for video_path in args.videos:
        print(f"Processing: {video_path}")
        
        try:
            results = pipeline.inference(
                video_path=video_path,
                audio_path=None,
                cache_dir=args.cache_dir
            )
            
            offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
            
            result_dict = {
                "video": video_path,
                "status": "success",
                "offset": int(offsets[0]) if offsets else None,
                "confidence": float(max_conf),
                "min_distance": float(min_dist),
                "has_face": has_face
            }
            
            print(f"  Offset: {offsets[0] if offsets else 'N/A'} frames")
            print(f"  Confidence: {max_conf:.3f}")
            
        except Exception as e:
            result_dict = {
                "video": video_path,
                "status": "error",
                "error": str(e)
            }
            print(f"  Error: {e}")
        
        results_all.append(result_dict)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results_all, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())