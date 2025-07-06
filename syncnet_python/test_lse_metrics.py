import logging
import sys
from pathlib import Path
from syncnet_pipeline import SyncNetPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def calculate_lse_metrics(video_path):
    """
    Calculate LSE-C and LSE-D metrics for a video
    
    LSE-C (Lip Sync Error - Confidence): 
    - Higher confidence values indicate better lip sync
    - Threshold typically around 3.5-4.0 for good sync
    
    LSE-D (Lip Sync Error - Distance):
    - Lower distance values indicate better lip sync  
    - Threshold typically around 6.5-7.0 for good sync
    """
    # Initialize pipeline
    pipe = SyncNetPipeline(
        {
            "s3fd_weights": "../weights/sfd_face.pth",
            "syncnet_weights": "../weights/syncnet_v2.model",
        },
        device="cuda",  # or "cpu"
    )
    
    # Run inference
    print(f"\n=== Processing {video_path} ===")
    
    # For testing, we'll extract audio from the video itself
    results = pipe.inference(
        video_path=video_path,
        audio_path=video_path,  # Extract audio from same video
        cache_dir="../example/cache_lse",
    )
    
    offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
    
    if not has_face:
        print(f"No face detected in {video_path}")
        return None, None
    
    # LSE-C is the maximum confidence across all tracks
    lse_c = max_conf
    
    # LSE-D is the minimum distance across all tracks
    lse_d = min_dist
    
    print(f"Number of face tracks: {len(offsets)}")
    print(f"Track offsets: {offsets}")
    print(f"Track confidences: {[f'{c:.3f}' for c in confs]}")
    print(f"Track distances: {[f'{d:.3f}' for d in dists]}")
    print(f"LSE-C (max confidence): {lse_c:.3f}")
    print(f"LSE-D (min distance): {lse_d:.3f}")
    
    # Interpretation
    sync_quality = "GOOD" if lse_c > 3.5 and lse_d < 7.0 else "POOR"
    print(f"Sync Quality: {sync_quality}")
    
    return lse_c, lse_d

def main():
    # Test files
    test_files = [
        "../example/pair_0000_lipsynced.mp4",
        "../example/pair_0001_lipsynced.mp4"
    ]
    
    results = {}
    
    for video_path in test_files:
        if Path(video_path).exists():
            lse_c, lse_d = calculate_lse_metrics(video_path)
            results[video_path] = (lse_c, lse_d)
        else:
            print(f"File not found: {video_path}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Video':<40} {'LSE-C':<10} {'LSE-D':<10} {'Quality':<10}")
    print("-" * 70)
    
    for video_path, (lse_c, lse_d) in results.items():
        if lse_c is not None:
            video_name = Path(video_path).name
            quality = "GOOD" if lse_c > 3.5 and lse_d < 7.0 else "POOR"
            print(f"{video_name:<40} {lse_c:<10.3f} {lse_d:<10.3f} {quality:<10}")

if __name__ == "__main__":
    main()