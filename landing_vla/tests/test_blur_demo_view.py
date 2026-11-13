import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np


class BlurDemoViewTests(unittest.TestCase):
    def test_click_buttons_use_the_same_toggle_and_severity_keys(self):
        from scripts.blur_demo_viewer import control_key_for_click
        self.assertEqual(control_key_for_click(800, 30), ord('b'))
        for x, key in ((800, '1'), (880, '2'), (980, '3')):
            self.assertEqual(control_key_for_click(x, 90), ord(key))
        self.assertEqual(control_key_for_click(400, 300), -1)

    def test_new_observation_does_not_overwrite_the_displayed_pair(self):
        from src.integration.blur_demo_support import BlurDemoSession
        from src.failures.control import default_control, write_control, apply_key
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            session = BlurDemoSession(Path(__file__).resolve().parents[1], output, 6)
            raw = np.random.default_rng(2).integers(0, 256, (256,256,3), dtype=np.uint8)
            state = {'position': [0,0,-2], 'orientation': [0,0,0,1]}
            session.inject([raw,raw], 1, state, 'instruction')
            first = json.loads((output/'telemetry.json').read_text())
            initial = {name: (output/file).read_bytes() for name,file in first['input_files'].items()}
            write_control(output/'control.json', apply_key(default_control(), ord('b')))
            session.inject([raw,raw], 2, state, 'instruction')
            second = json.loads((output/'telemetry.json').read_text())
            self.assertNotEqual(first['input_files'], second['input_files'])
            for name, file in first['input_files'].items():
                self.assertEqual((output/file).read_bytes(), initial[name])
            for name, file in second['input_files'].items():
                image = cv2.imread(str(output/file))
                self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), second['failure']['used_frame_sha256'][name])

    def test_uniform_observation_can_be_blurred_without_fabricating_a_tensor_difference(self):
        from src.integration.blur_demo_support import BlurDemoSession
        with tempfile.TemporaryDirectory() as directory:
            session = BlurDemoSession(Path(__file__).resolve().parents[1], Path(directory), 6)
            hashes = {'front': 'same', 'down': 'same'}
            failure = {'failure_enabled': True, 'raw_frame_sha256': hashes, 'used_frame_sha256': hashes}
            inference = {'input_evidence': {'used_frame_sha256': hashes, 'tensor_differs_from_reference': False}}
            session.verify_input(inference, failure)
            self.assertTrue(session.telemetry['input_verified'])

    def test_incomplete_display_pair_is_not_accepted_under_new_telemetry(self):
        from scripts.blur_demo_viewer import read_input_pair
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image = np.full((256,256,3), 20, dtype=np.uint8)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            cv2.imwrite(str(output/'front.png'), image)
            (output/'down.png').write_bytes(b'incomplete image')
            telemetry = {'input_files': {'front':'front.png','down':'down.png'},
                         'failure': {'used_frame_sha256': {'front':digest,'down':digest}}}
            self.assertIsNone(read_input_pair(output, telemetry))
            cv2.imwrite(str(output/'down.png'), image)
            pair = read_input_pair(output, telemetry)
            np.testing.assert_array_equal(pair['front'], image)
            np.testing.assert_array_equal(pair['down'], image)

    def test_new_run_archives_old_evidence_without_mixing_logs_or_removing_notes(self):
        from src.failures.control import initialize_run_output
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('closed-loop-log.jsonl','control-events.jsonl','blur_front.png','comparison.png'):
                (output/name).write_text('previous run')
            (output/'my_notes.txt').write_text('keep in place')
            archive = initialize_run_output(output)
            for name in ('closed-loop-log.jsonl','control-events.jsonl','blur_front.png','comparison.png'):
                self.assertFalse((output/name).exists())
                self.assertEqual((archive/name).read_text(), 'previous run')
            self.assertEqual((output/'my_notes.txt').read_text(), 'keep in place')
            self.assertTrue((output/'run-info.json').is_file())


if __name__ == '__main__':
    unittest.main()
