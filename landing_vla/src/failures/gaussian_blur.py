"""Non-destructive Gaussian blur, applied before AeroVLA preprocessing."""
import cv2
import numpy as np

SEVERITIES = {'low': (7, 1.5), 'medium': (15, 3.0), 'high': (31, 6.0)}


class GaussianBlurFailure:
    def __init__(self, enabled=False, severity='medium'):
        if type(enabled) is not bool:
            raise ValueError('enabled must be bool')
        self.enabled = enabled
        self.set_severity(severity)

    def set_severity(self, severity):
        if severity not in SEVERITIES:
            raise ValueError(f'Unknown severity: {severity}')
        self.severity = severity

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def apply(self, image):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError('Expected a nonempty uint8 HxWx3 camera frame')
        if not self.enabled:
            return image.copy()
        kernel, sigma = SEVERITIES[self.severity]
        return cv2.GaussianBlur(image, (kernel, kernel), sigma)
