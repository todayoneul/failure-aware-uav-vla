"""GIF timing must follow the recorded flight, rather than speed it up."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DemoMediaTests(unittest.TestCase):
    def test_irregular_frame_timing_is_preserved(self):
        from scripts.export_demo_media import frame_durations
        samples = [{"elapsed_s": t} for t in (10, 10.12, 10.5)]
        self.assertEqual(frame_durations(samples, 10.6), [120, 380, 100])

    def test_out_of_order_frames_are_rejected(self):
        from scripts.export_demo_media import frame_durations
        with self.assertRaises(ValueError):
            frame_durations([{"elapsed_s": 1}, {"elapsed_s": .9}], 2)


if __name__ == "__main__":
    unittest.main()
