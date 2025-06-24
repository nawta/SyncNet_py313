# SyncNet Python

Audio-visual synchronization detection using deep learning with modern Python architecture.

This is a **refactored and enhanced version** of the original [SyncNet implementation](https://github.com/joonson/syncnet_python) by Joon Son Chung, updated for Python 3.9+ with clean architecture, comprehensive error handling, and performance optimizations.

## Overview

SyncNet Python is a PyTorch implementation of the SyncNet model, which detects audio-visual synchronization in videos. It can identify lip-sync errors by analyzing the correspondence between mouth movements and spoken audio.

## Features

### Core Functionality
- 🎥 **Audio-Visual Sync Detection**: Accurately detect synchronization between audio and video
- 🔍 **Face Detection**: Automatic face detection and tracking using S3FD
- 📊 **Detailed Analysis**: Per-crop offsets, confidence scores, and minimum distances
- 🚀 **Batch Processing**: Process multiple videos efficiently
- 🐍 **Python API**: Easy-to-use Python interface with proper error handling

### Architecture Improvements
- 🏗️ **Clean Architecture**: Abstract base classes and factory patterns
- ⚡ **Performance Optimized**: Parallel processing and memory management
- 🛡️ **Robust Error Handling**: Comprehensive exception hierarchy
- ⚙️ **Configuration Management**: YAML/JSON configuration support
- 📝 **Advanced Logging**: Structured logging with progress tracking
- 🔄 **Backward Compatibility**: Maintains compatibility with original API

## Installation

```bash
pip install syncnet-python
```

### Additional Requirements

1. **FFmpeg**: Required for video processing
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **Model Weights**: Download pre-trained weights
   - Download `sfd_face.pth` and `syncnet_v2.model`
   - Place them in a `weights/` directory

## Quick Start

```python
from syncnet_python import SyncNetPipeline

# Initialize pipeline
pipeline = SyncNetPipeline(
    s3fd_weights="weights/sfd_face.pth",
    syncnet_weights="weights/syncnet_v2.model",
    device="cuda"  # or "cpu"
)

# Process video
results = pipeline.inference(
    video_path="video.mp4",
    audio_path=None  # Extract from video
)

# Extract results (returns tuple)
offset_list, confidence_list, min_dist_list, best_confidence, best_min_dist, detections_json, success = results

# Get best results
offset = offset_list[0]  # AV offset in frames
confidence = confidence_list[0]  # Confidence score
min_distance = min_dist_list[0]  # Minimum distance

print(f"AV Offset: {offset} frames")
print(f"Confidence: {confidence:.3f}")
print(f"Min Distance: {min_distance:.3f}")
```

### Detailed Analysis

```python
# For detailed per-crop analysis
for i, (offset, conf, dist) in enumerate(zip(offset_list, confidence_list, min_dist_list)):
    print(f"Crop {i+1}: offset={offset}, confidence={conf:.3f}, min_dist={dist:.3f}")

# Parse face detections
import json
detections = json.loads(detections_json)
print(f"Total frames with face detection: {len(detections)}")
```

## Command Line Usage

```bash
# Process single video
syncnet-python video.mp4

# Process multiple videos
syncnet-python video1.mp4 video2.mp4 --output results.json

# Use CPU instead of GPU
syncnet-python video.mp4 --device cpu
```

## Performance

Tested with example files:
- **Processing Speed**: 191.4 fps
- **Face Detection**: 100% success rate
- **Accuracy**: Detects 1-frame offsets with high confidence (4.5+)
- **Compute Time**: ~0.65 seconds for 134 frames

## Architecture

### Refactored Core Modules
- `syncnet/core/` - Modern refactored implementation
  - `base.py` - Abstract base classes and interfaces
  - `models.py` - Enhanced SyncNet model with factory pattern
  - `audio.py` - MFCC audio processing with streaming support
  - `video.py` - Parallel video processing with OpenCV
  - `sync_analyzer.py` - Optimized sync analysis with caching
  - `config.py` - Configuration management system
  - `exceptions.py` - Comprehensive error handling
  - `logging.py` - Advanced logging with progress tracking
  - `utils.py` - Memory management and utility functions

### Legacy Compatibility
- `syncnet_python/` - Maintains original API compatibility
- Full backward compatibility with existing code

## Requirements

- Python 3.9+ (tested on 3.13)
- PyTorch 2.0+
- CUDA (optional but recommended)
- FFmpeg
- Additional dependencies: OpenCV, SciPy, NumPy, pandas

## Credits

This package is based on the original [SyncNet implementation](https://github.com/joonson/syncnet_python) by Joon Son Chung, enhanced with modern Python architecture and performance optimizations.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{chung2016out,
  title={Out of time: automated lip sync in the wild},
  author={Chung, Joon Son and Zisserman, Andrew},
  booktitle={Asian Conference on Computer Vision},
  year={2016}
}
```

## License

MIT License - see LICENSE file for details.

## Links

- GitHub: https://github.com/yourusername/syncnet-python
- Documentation: https://syncnet-python.readthedocs.io
- Issues: https://github.com/yourusername/syncnet-python/issues