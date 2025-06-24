"""Command-line interface for SyncNet with Python 3.13 features."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from syncnet.pipeline.pipeline import SyncNetPipeline
from syncnet.pipeline.config import PipelineConfig
from syncnet.utils.exceptions import SyncNetError, PipelineError, ModelLoadError


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SyncNet: Audio-Visual Synchronization Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python -m syncnet.cli video.mp4
  
  # Process with custom weights
  python -m syncnet.cli video.mp4 --s3fd-weights path/to/s3fd.pth --syncnet-weights path/to/syncnet.pth
  
  # Batch process videos
  python -m syncnet.cli videos/*.mp4 --output results.json
  
  # Use CPU instead of GPU
  python -m syncnet.cli video.mp4 --device cpu
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'videos',
        nargs='+',
        type=Path,
        help='Video file(s) to process'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Directory for temporary files (uses temp dir if not specified)'
    )
    
    # Model options
    parser.add_argument(
        '--s3fd-weights',
        type=Path,
        default=Path('weights/sfd_face.pth'),
        help='Path to S3FD face detector weights'
    )
    parser.add_argument(
        '--syncnet-weights',
        type=Path,
        default=Path('weights/syncnet_v2.model'),
        help='Path to SyncNet model weights'
    )
    
    # Processing options
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run models on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--no-scene-detection',
        action='store_true',
        help='Disable scene detection'
    )
    
    # Face detection options
    parser.add_argument(
        '--facedet-scale',
        type=float,
        default=0.25,
        help='Scale factor for face detection'
    )
    parser.add_argument(
        '--min-track',
        type=int,
        default=50,
        help='Minimum track length in frames'
    )
    parser.add_argument(
        '--min-face-size',
        type=int,
        default=100,
        help='Minimum face size in pixels'
    )
    
    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser.parse_args()


async def process_video(
    pipeline: SyncNetPipeline,
    video_path: Path,
    cache_dir: Optional[Path],
    scene_detection: bool
) -> dict:
    """Process a single video."""
    try:
        logging.info(f"Processing: {video_path}")
        
        result = await pipeline.process_video_async(
            video_path,
            cache_dir=cache_dir,
            scene_detection=scene_detection
        )
        
        # Convert to serializable format
        return {
            'video': str(video_path),
            'status': 'success',
            'num_tracks': result['num_tracks'],
            'sync_results': [
                {
                    'offset': r['offset'],
                    'confidence': r['confidence'],
                    'num_frames': len(r['dists'])
                }
                for r in result['sync_results']
            ],
            'processing_time': result['processing_time']
        }
        
    except SyncNetError as e:
        logging.error(f"Error processing {video_path}: {e}")
        return {
            'video': str(video_path),
            'status': 'error',
            'error': str(e),
            'details': e.details
        }
    except Exception as e:
        logging.exception(f"Unexpected error processing {video_path}")
        return {
            'video': str(video_path),
            'status': 'error',
            'error': str(e)
        }


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    # Create configuration
    config = PipelineConfig(
        s3fd_weights=args.s3fd_weights,
        syncnet_weights=args.syncnet_weights,
        batch_size=args.batch_size,
        facedet_scale=args.facedet_scale,
        min_track=args.min_track,
        min_face_size=args.min_face_size
    )
    
    # Initialize pipeline
    try:
        pipeline = SyncNetPipeline(config=config, device=args.device)
    except ModelLoadError as e:
        logging.error(f"Failed to load models: {e}")
        return 1
    
    # Process videos
    results = []
    errors = []
    
    # Create tasks for all videos
    tasks = [
        process_video(
            pipeline,
            video_path,
            args.cache_dir,
            not args.no_scene_detection
        )
        for video_path in args.videos
    ]
    
    # Process concurrently
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        
        if result['status'] == 'error':
            errors.append(Exception(result['error']))
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to: {args.output}")
    else:
        # Print summary to console
        for result in results:
            print(f"\n{result['video']}:")
            if result['status'] == 'success':
                print(f"  Tracks found: {result['num_tracks']}")
                for i, sync in enumerate(result['sync_results']):
                    print(f"  Track {i+1}:")
                    print(f"    Offset: {sync['offset']} frames")
                    print(f"    Confidence: {sync['confidence']:.3f}")
            else:
                print(f"  Error: {result['error']}")
    
    # Handle errors using exception groups
    if errors:
        error = PipelineError(
            f"Failed to process {len(errors)} video(s)",
            errors=errors
        )
        try:
            error.raise_group()
        except* (VideoProcessingError, AudioProcessingError) as eg:
            for e in eg.exceptions:
                logging.error(f"Processing error: {e}")
        except* Exception as eg:
            for e in eg.exceptions:
                logging.error(f"Unexpected error: {e}")
    
    return 1 if errors else 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Check if video files exist
    missing_files = [v for v in args.videos if not v.exists()]
    if missing_files:
        logging.error(f"Files not found: {', '.join(str(f) for f in missing_files)}")
        return 1
    
    # Run async main
    return asyncio.run(main_async(args))


if __name__ == '__main__':
    sys.exit(main())