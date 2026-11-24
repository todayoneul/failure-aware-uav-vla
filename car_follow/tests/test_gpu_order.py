"""Demo 1 step 6: the simulator-first load order (src/car_follow/gpu_order.py). No GPU, no simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest
from unittest import mock

from src.car_follow import gpu_order as order
from src.car_follow import vlm


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class OrderTests(unittest.TestCase):
    def setUp(self):
        order.reset()
        self.addCleanup(order.reset)

    def test_a_live_vlm_needs_the_simulator_up(self):
        with self.assertRaises(order.LoadOrderError):
            order.before_vlm_load("m", settle=lambda: None)
        order.simulator_ready()
        settled = []
        order.before_vlm_load("m", settle=lambda: settled.append(True))
        self.assertEqual(settled, [True])

    def test_the_simulator_may_not_start_over_a_loaded_vlm(self):
        order.before_map_load()
        order.vlm_loaded("m")
        with self.assertRaises(order.LoadOrderError):
            order.before_map_load()
        order.vlm_unloaded("m")
        order.before_map_load()

    def test_an_offline_vlm_needs_the_simulator_off(self):
        order.before_vlm_load("m", offline=True, listening=lambda: False)
        with self.assertRaises(order.LoadOrderError):
            order.before_vlm_load("m", offline=True, listening=lambda: True)

    def test_wait_settled_waits_out_the_map_load_peak(self):
        clock = FakeClock()
        read = lambda: (10900.0 if clock.t < 3.0 else 5400.0, 12227.0)
        used = order.wait_settled(read, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(used, 5400.0)
        self.assertGreaterEqual(clock.t, 5.0)

    def test_wait_settled_gives_up_on_memory_that_keeps_moving(self):
        clock = FakeClock()
        read = lambda: (5000.0 + 1000.0 * (int(clock.t * 4) % 2), 12227.0)
        with self.assertRaises(order.LoadOrderError):
            order.wait_settled(read, timeout_s=5.0, clock=clock.now, sleep=clock.sleep)

    def test_make_applies_the_rule_and_unload_clears_it(self):
        class Fake:
            name = "fake"

            def unload(self):
                order.vlm_unloaded(self.name)

        vlm.MODELS["fake"] = lambda: Fake()
        self.addCleanup(vlm.MODELS.pop, "fake")
        with self.assertRaises(order.LoadOrderError):
            vlm.make("fake")
        order.simulator_ready()
        with mock.patch.object(order, "wait_settled", lambda: 5400.0):
            model = vlm.make("fake")
        with self.assertRaises(order.LoadOrderError):
            order.before_map_load()
        model.unload()
        order.before_map_load()


if __name__ == "__main__":
    unittest.main()
