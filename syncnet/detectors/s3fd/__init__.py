"""S3FD face detector."""

from syncnet.detectors.s3fd.detector import S3FDNet, L2Norm
from syncnet.detectors.s3fd.utils import Detect, PriorBox

__all__ = ["S3FDNet", "L2Norm", "Detect", "PriorBox"]