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
                loader.decoder = 'free'
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


class GrammarTokenizer:
    """Just enough of the Llama vocabulary: digits, the word break, LAND, and one base-OpenVLA action token."""
    vocabulary = ['<unk>', '<s>', '</s>', '▁'] + [str(d) for d in range(10)] + ['▁L', 'AND', '식']
    eos_token_id = 2; unk_token_id = 0
    def convert_tokens_to_ids(self, token):
        return self.vocabulary.index(token) if token in self.vocabulary else self.unk_token_id
    def convert_ids_to_tokens(self, ids):
        return [self.vocabulary[i] for i in ids] if isinstance(ids, list) else self.vocabulary[ids]
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [self.vocabulary.index(t) for t in ('4', '9', '▁L', 'AND')]}


class ActionGrammarTests(unittest.TestCase):
    def test_out_of_vocabulary_token_is_replaced_by_the_best_digit_and_reported(self):
        from src.integration.aerovla_int4_loader import ActionGrammar
        tokenizer = GrammarTokenizer(); index = tokenizer.vocabulary.index
        grammar = ActionGrammar(tokenizer, prompt_length=2)
        prompt = [1, 1]
        # Position 0: a clean '9'. Position 1: the stray token leads, '7' is the best digit (the recorded `9식 49 49`).
        first = torch.full((1, len(tokenizer.vocabulary)), -9.); first[0, index('9')] = 5.
        second = torch.full((1, len(tokenizer.vocabulary)), -9.)
        second[0, index('식')] = 2.; second[0, index('7')] = 1.25; second[0, index('6')] = 0.
        kept = grammar(torch.tensor([prompt]), first)
        self.assertEqual(int(kept.argmax()), index('9'))
        masked = grammar(torch.tensor([prompt + [index('9')]]), second)
        self.assertEqual(int(masked.argmax()), index('7'))
        self.assertTrue(torch.isinf(masked[0, index('식')]))
        # After a 9 the grammar also forbids a second 9 (bins stop at 98).
        self.assertTrue(torch.isinf(masked[0, index('9')]))
        report = grammar.report()
        self.assertTrue(report['intervened']); self.assertEqual(len(report['interventions']), 1)
        change = report['interventions'][0]
        self.assertEqual((change['position'], change['rejected'], change['chosen']), (1, '식', '7'))
        self.assertEqual(change['rejected_id'], index('식'))
        self.assertGreater(change['rejected_probability'], change['chosen_probability'])
        self.assertAlmostEqual(report['minimum_valid_mass'], change['valid_mass'])
        self.assertLess(report['minimum_valid_mass'], .5)

    def test_clean_output_is_untouched(self):
        from src.integration.aerovla_int4_loader import ActionGrammar
        tokenizer = GrammarTokenizer(); index = tokenizer.vocabulary.index
        grammar = ActionGrammar(tokenizer, prompt_length=1)
        sequence = [1]
        for token in ('1', '6', '▁', '4', '9', '▁', '4', '9', '▁L', 'AND', '</s>'):
            scores = torch.full((1, len(tokenizer.vocabulary)), -9.); scores[0, index(token)] = 5.
            self.assertEqual(int(grammar(torch.tensor([sequence]), scores).argmax()), index(token))
            sequence.append(index(token))
        self.assertFalse(grammar.report()['intervened']); self.assertGreater(grammar.report()['minimum_valid_mass'], .99)

    def test_loader_rejects_an_unknown_decoder(self):
        from src.integration.aerovla_int4_loader import AeroVLAInt4
        with self.assertRaises(ValueError): AeroVLAInt4('unused', decoder='sampling')


if __name__ == '__main__':
    unittest.main()
