"""Demo 1 trials: colour names, lane names, how the target is named among look-alikes, the road split and the sealed
main list. No simulator (the carla package must be importable: run with `uv run`).

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.car_follow import trials


class ColorTests(unittest.TestCase):
    def test_families_of_carla_colours(self):
        cases = {"217,217,217": "white", "255,255,255": "white", "0,0,0": "black", "33,33,33": "black",
                 "109,109,109": "gray", "167,166,175": "gray", "168,0,27": "red", "0,39,105": "blue",
                 "215,89,0": "orange", "255,208,0": "yellow", "16,44,21": "black"}
        for color, family in cases.items():
            self.assertEqual(trials.color_family(color), family, color)

    def test_only_clear_colours_may_be_a_target(self):
        for color in ("217,217,217", "0,0,0", "109,109,109", "168,0,27", "0,39,105", "215,89,0"):
            self.assertTrue(trials.clear_color(color), color)
        # dark, near-black green, pale, olive, grays that render white or black, and next to a hue boundary
        for color in ("16,44,21", "0,33,5", "73,0,0", "215,255,143", "105,103,0", "206,206,206", "160,160,160",
                      "167,166,175", "83,83,86", "68,68,77", "211,142,0", "158,40,82"):
            self.assertFalse(trials.clear_color(color), color)

    def test_every_target_blueprint_has_a_korean_object_form(self):
        for blueprint, entry in trials.load_types().items():
            self.assertTrue(entry["ko_obj"].startswith(entry["ko"]), blueprint)
            self.assertIn(entry["ko_obj"][-1], "을를", blueprint)


class SentenceTests(unittest.TestCase):
    def test_lane_names(self):
        self.assertEqual(trials.lane_name(0, 2, "en"), "left lane")
        self.assertEqual(trials.lane_name(1, 3, "en"), "middle lane")
        self.assertEqual(trials.lane_name(1, 4, "en"), "second lane from the left")
        self.assertEqual(trials.lane_name(2, 4, "en"), "second lane from the right")
        self.assertEqual(trials.lane_name(2, 3, "ko"), "오른쪽 차선")
        self.assertEqual(trials.lane_name(3, 4, "ko"), "맨 오른쪽 차선")

    def test_the_lane_names_the_target_only_if_no_visible_match_shares_it(self):
        me = {"index_from_left": 0, "lanes": 2, "along_m": 0.0}
        other_lane = {"index_from_left": 1, "lanes": 2, "along_m": 2.0}
        same_lane_ahead = {"index_from_left": 0, "lanes": 2, "along_m": 15.0}
        self.assertEqual(trials.expressions(me, [other_lane]), ["lane"])
        self.assertEqual(trials.expressions(me, [same_lane_ahead]), ["back"])
        self.assertEqual(trials.expressions(me, [other_lane, same_lane_ahead]), [])

    def test_order_needs_a_margin_and_other_roads_never_allow_the_lane(self):
        me = {"index_from_left": 1, "lanes": 3, "along_m": 0.0}
        self.assertEqual(trials.expressions(me, [{"index_from_left": 0, "lanes": 3, "along_m": -4.0}]), ["lane"])
        self.assertEqual(trials.expressions(me, [{"index_from_left": 0, "lanes": 3, "along_m": -7.0}]),
                         ["lane", "front"])
        self.assertEqual(trials.expressions(me, [{"index_from_left": -1, "lanes": -1, "along_m": 3.0}]), [])

    def test_sentences(self):
        entry = trials.load_types()["vehicle.nissan.patrol_2021"]
        s = trials.sentence("lane", entry, "217,217,217", 0, 2)
        self.assertEqual(s["en"], "follow the white SUV in the left lane")
        self.assertEqual(s["ko"], "왼쪽 차선의 흰색 SUV를 따라가")
        s = trials.sentence("back", trials.load_types()["vehicle.chevrolet.impala"], "0,0,0", 0, 2)
        self.assertEqual(s["en"], "follow the black sedan furthest back")
        self.assertEqual(s["ko"], "가장 뒤에 있는 검은색 세단을 따라가")


class SplitTests(unittest.TestCase):
    def test_cell_split_is_fixed_and_shares_the_map(self):
        cells = [(x, y) for x in range(-10, 10) for y in range(-10, 10)]
        splits = [trials.cell_split(c) for c in cells]
        self.assertEqual(splits, [trials.cell_split(c) for c in cells])
        counts = {s: splits.count(s) for s in ("main", "gate", "dev")}
        self.assertTrue(70 <= counts["main"] <= 130 and 70 <= counts["gate"] <= 130, counts)
        self.assertEqual(trials.cell_of(59.9, -0.1), (0, -1))

    def test_look_alike_slots_keep_the_intended_expression_true(self):
        self.assertTrue(trials.slot_fits("lane", 1, 0.0) and not trials.slot_fits("lane", 0, 12.0))
        self.assertTrue(trials.slot_fits("front", 0, -12.0) and not trials.slot_fits("front", 1, -5.0))
        self.assertTrue(trials.slot_fits("back", -1, 6.0) and not trials.slot_fits("back", -1, 0.0))

    def test_main_is_sealed_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, main = Path(tmp) / "trials.json", Path(tmp) / "main.json"
            trials.write_trials({"dev": [{"id": "dev-001"}]}, [{"id": "main-001"}], index_path=index, main_path=main)
            self.assertEqual(trials.load_trials("dev", index_path=index), [{"id": "dev-001"}])
            with self.assertRaises(PermissionError):
                trials.load_trials("main", index_path=index)
            self.assertEqual(trials.load_trials("main", open_sealed=True, index_path=index), [{"id": "main-001"}])
            main.write_text(main.read_text(encoding="utf-8").replace("main-001", "main-002"), encoding="utf-8")
            with self.assertRaises(ValueError):
                trials.load_trials("main", open_sealed=True, index_path=index)
            self.assertNotIn("main-001", index.read_text(encoding="utf-8"))
            self.assertTrue(json.loads(index.read_text(encoding="utf-8"))["main"]["sealed"])


if __name__ == "__main__":
    unittest.main()
