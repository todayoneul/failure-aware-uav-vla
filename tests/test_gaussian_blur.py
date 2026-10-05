import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2


class GaussianBlurTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.image[::2, ::2] = [30, 100, 250]
        self.image[1::2, 1::2] = [250, 20, 70]

    def test_disabled_is_exact_and_non_destructive(self):
        from src.failures.gaussian_blur import GaussianBlurFailure
        original = self.image.copy()
        output = GaussianBlurFailure().apply(self.image)
        np.testing.assert_array_equal(output, original)
        output[:] = 0
        np.testing.assert_array_equal(self.image, original)

    def test_each_severity_changes_both_camera_inputs_and_preserves_contract(self):
        from src.failures.gaussian_blur import GaussianBlurFailure
        original = self.image.copy()
        for severity in ('low', 'medium', 'high'):
            with self.subTest(severity=severity):
                failure = GaussianBlurFailure(enabled=True, severity=severity)
                for image in (self.image, np.rot90(self.image).copy()):
                    output = failure.apply(image)
                    self.assertEqual(output.shape, image.shape)
                    self.assertEqual(output.dtype, image.dtype)
                    self.assertFalse(np.array_equal(output, image))
                    self.assertLess(cv2.Laplacian(output, cv2.CV_64F).var(), cv2.Laplacian(image, cv2.CV_64F).var())
        np.testing.assert_array_equal(self.image, original)

    def test_toggle_and_severity_validation(self):
        from src.failures.gaussian_blur import GaussianBlurFailure
        failure = GaussianBlurFailure()
        self.assertFalse(failure.enabled)
        self.assertEqual(failure.severity, 'medium')
        self.assertTrue(failure.toggle())
        self.assertFalse(failure.toggle())
        failure.enable(); self.assertTrue(failure.enabled)
        failure.disable(); self.assertFalse(failure.enabled)
        with self.assertRaises(ValueError):
            GaussianBlurFailure(severity='unknown')


class FailureControlTests(unittest.TestCase):
    def test_keys_toggle_severity_and_quit_without_mutating_previous_state(self):
        from src.failures.control import default_control, apply_key
        initial = default_control()
        state = apply_key(initial, ord('b'))
        self.assertTrue(state['enabled']); self.assertFalse(initial['enabled'])
        for key, severity in ((ord('1'), 'low'), (ord('2'), 'medium'), (ord('3'), 'high')):
            state = apply_key(state, key)
            self.assertEqual(state['severity'], severity)
        self.assertFalse(apply_key(state, ord('B'))['enabled'])
        self.assertTrue(apply_key(state, 27)['quit'])
        self.assertTrue(apply_key(state, ord('q'))['quit'])

    def test_atomic_control_round_trip_and_malformed_state_rejected(self):
        from src.failures.control import default_control, read_control, write_control
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.json'
            write_control(path, default_control())
            self.assertEqual(read_control(path), default_control())
            path.write_text(json.dumps({'enabled': 'false', 'severity': 'medium', 'quit': False, 'revision': 0}))
            with self.assertRaises(ValueError):
                read_control(path)


if __name__ == '__main__':
    unittest.main()
