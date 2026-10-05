"""Trace the tensor that generate actually receives, without loading model weights."""
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image


class Tokenizer:
    eos_token_id = 0
    def __call__(self, *args, **kwargs):
        return {'input_ids': torch.tensor([[1, 2]])}
    def decode(self, *args, **kwargs):
        return 'Action: 98 49 49</s>'


class Processor:
    def __call__(self, images, return_tensors):
        values = np.asarray(images.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        three = torch.from_numpy(values).permute(2, 0, 1)
        return {'pixel_values': torch.cat((three, three), dim=0).unsqueeze(0)}


class ModelSpy:
    def generate(self, **inputs):
        self.received = inputs['pixel_values'].clone()
        return torch.tensor([[1, 2, 3, 4, 5]])


class ActualInputTraceTests(unittest.TestCase):
    def test_off_matches_reference_and_on_reaches_generated_tensor(self):
        from src.integration.aerovla_int4_loader import AeroVLAInt4
        from src.failures.gaussian_blur import GaussianBlurFailure
        raw = np.random.default_rng(4).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        state = {'position': [0, 0, -2], 'orientation': [0, 0, 0, 1]}
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                loader = AeroVLAInt4.__new__(AeroVLAInt4)
                loader.device = torch.device('cpu')
                loader.tokenizer = Tokenizer(); loader.image_processor = Processor(); loader.model = ModelSpy()
                used = GaussianBlurFailure(enabled=enabled).apply(raw)
                with patch('torch.cuda.synchronize'):
                    result = loader.infer(used, used, state, [3, 0, -2],
                                          'The target is 3 meters away and 0 degrees from you. Find the blocks. Please control the drone.',
                                          trace_inputs=True, reference_frames=(raw, raw))
                evidence = result['input_evidence']
                digest = hashlib.sha256(loader.model.received.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
                self.assertEqual(evidence['pixel_values_sha256'], digest)
                self.assertEqual(evidence['used_frame_sha256']['front'], hashlib.sha256(used.tobytes()).hexdigest())
                self.assertEqual(evidence['tensor_differs_from_reference'], enabled)
                self.assertEqual(result['parsed_action']['bins'], [98, 49, 49])


if __name__ == '__main__':
    unittest.main()
