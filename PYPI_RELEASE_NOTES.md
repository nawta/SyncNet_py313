# SyncNet Python - PyPI Release Notes

## Version 0.1.1

### Changes
- Added proper attribution to the original [SyncNet implementation](https://github.com/joonson/syncnet_python) by Joon Son Chung
- Added Python 3.9 compatibility fixes for type hints
- Updated project description to clarify this is an updated version for modern Python

### Package Information
- **PyPI URL**: https://pypi.org/project/syncnet-python/0.1.1/
- **Install**: `pip install syncnet-python`
- **Python Support**: 3.9, 3.10, 3.11, 3.12, 3.13

### Notes on Python Version Support

While the package is marked as supporting Python 3.9+, the main development and testing was done on Python 3.13. The code includes:
- Basic compatibility fixes for Python 3.9 type hints
- Standard library features that should work across Python 3.9-3.13
- PyTorch and other dependencies that support these Python versions

Users on Python 3.9-3.12 may encounter minor issues. Please report any compatibility problems on the GitHub repository.

### Original Implementation

This package is based on the original SyncNet implementation:
- GitHub: https://github.com/joonson/syncnet_python
- Author: Joon Son Chung
- Paper: "Out of time: automated lip sync in the wild" (ACCV 2016)

### Model Weights

Due to PyPI size limitations, model weights must be downloaded separately:
1. `sfd_face.pth` - S3FD face detector
2. `syncnet_v2.model` - SyncNet model

These can be obtained from the original repository or other sources.