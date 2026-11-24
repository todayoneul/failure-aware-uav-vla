"""Demo 1 step 6: who a training scene can name, the sentences, and the ambiguity metrics. No GPU, no simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

from src.car_follow import trials, vlm_data, vlm_metrics


def car(i, lane, along, pixels=1000, same=True, lanes=3):
    return {"id": i, "pixels": pixels, "same_road": same, "index_from_left": lane if same else -1,
            "lanes": lanes if same else -1, "along_m": along}


class NameableTests(unittest.TestCase):
    def test_each_member_is_named_by_what_is_true_of_it(self):
        found = {(c["id"], e) for c, e in vlm_data.nameable([car(1, 0, 25.0), car(2, 1, 12.0), car(3, 2, 0.0)])}
        self.assertEqual(found, {(1, "lane"), (1, "front"), (2, "lane"), (3, "lane"), (3, "back")})

    def test_a_shared_lane_or_a_close_order_does_not_name_a_car(self):
        found = {(c["id"], e) for c, e in vlm_data.nameable([car(1, 0, 10.0), car(2, 0, 7.0)])}
        self.assertEqual(found, set())

    def test_small_or_off_road_cars_are_not_targets_but_still_count(self):
        found = {(c["id"], e) for c, e in vlm_data.nameable([car(1, 0, 20.0, pixels=100), car(2, 0, 0.0)])}
        self.assertEqual(found, {(2, "back")})
        self.assertEqual(vlm_data.nameable([car(1, 0, 20.0, same=False), car(2, 1, 0.0, pixels=10)]), [])

    def test_a_car_alone_is_not_an_example(self):
        self.assertEqual(vlm_data.nameable([car(1, 0, 0.0)]), [])


class OtherTargetTests(unittest.TestCase):
    @staticmethod
    def member(i, lane, along, box=0, pixels=1000):
        return {"id": i, "pixels": pixels, "index_from_left": lane, "lanes": 3, "along_m": along, "box": box}

    def test_names_a_different_car_and_keeps_the_expression_when_it_can(self):
        cars = [self.member(1, 0, 25.0), self.member(2, 1, 12.0), self.member(3, 2, 0.0)]
        car, expression = vlm_data.other_target(cars, 1, prefer="back")
        self.assertEqual((car["id"], expression), (3, "back"))

    def test_a_car_the_detector_missed_or_barely_seen_is_not_named(self):
        cars = [self.member(1, 0, 20.0), self.member(2, 1, 0.0, box=None), self.member(3, 2, 10.0, pixels=100)]
        self.assertIsNone(vlm_data.other_target(cars, 1))


class SentenceTests(unittest.TestCase):
    def test_the_first_template_is_the_trial_wording(self):
        entry = {"type": "SUV", "ko_obj": "SUV를"}
        for expression in ("lane", "front", "back"):
            want = trials.sentence(expression, entry, "255,255,255", 0, 3)
            for lang in ("en", "ko"):
                got = vlm_data.render(vlm_data.TEMPLATES[lang][expression][0], lang, "white", entry, 0, 3)
                self.assertEqual(got, want[lang])

    def test_held_out_wordings_are_never_trained_on(self):
        for lang in ("en", "ko"):
            for expression, template in vlm_data.HELD_OUT[lang].items():
                self.assertNotIn(template, vlm_data.TEMPLATES[lang][expression])


class ReliabilityTests(unittest.TestCase):
    @staticmethod
    def row(right, top, margin, agreement=1.0):
        other = max(top - margin, 0.0)
        probs = [top, other, max(1.0 - top - other, 0.0)]
        return {"right": right, "probs": probs, "margin": margin, "agreement": agreement, "target": 0 if right else 1}

    def test_an_honest_margin_separates_right_from_wrong(self):
        rows = [self.row(True, 0.95, 0.9)] * 8 + [self.row(False, 0.5, 0.1, 2 / 3)] * 2
        r = vlm_metrics.reliability(rows)
        self.assertEqual(r["margin_auroc"], 1.0)
        self.assertEqual(r["wrong_with_margin"]["0.8"], 0.0)
        self.assertEqual(r["keep"]["0.8"], {"kept": 0.8, "accuracy": 1.0})
        self.assertEqual(r["keep_agreement"], {"kept": 0.8, "accuracy": 1.0})

    def test_a_confidently_wrong_model_is_flagged(self):
        rows = [self.row(True, 0.99, 0.98)] * 5 + [self.row(False, 0.99, 0.98)] * 5
        r = vlm_metrics.reliability(rows)
        self.assertEqual(r["margin_auroc"], 0.5)
        self.assertEqual(r["wrong_with_margin"]["0.9"], 1.0)
        self.assertGreater(r["ece"], 0.45)

    def test_rows_without_candidates_are_skipped(self):
        self.assertIsNone(vlm_metrics.reliability([{"right": False, "margin": 0.0, "agreement": 0.0}]))


if __name__ == "__main__":
    unittest.main()
