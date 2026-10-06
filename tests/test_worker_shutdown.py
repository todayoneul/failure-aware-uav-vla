import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path


@unittest.skipUnless(os.name == 'posix' and Path('/proc').exists(), 'WSL/Linux process ownership test')
class WorkerShutdownTests(unittest.TestCase):
    def start_worker(self,script,marker,token):
        # Popen may return before exec replaces the child's inherited argv.
        # Wait for an explicit child acknowledgement, especially under GPU-load startup.
        script.write_text("from pathlib import Path\nimport time\nPath(__file__).with_suffix('.ready').write_text('ready')\ntime.sleep(60)")
        process=subprocess.Popen([sys.executable,str(script),marker,'--run-token',token])
        deadline=time.monotonic()+5
        while not script.with_suffix('.ready').exists():
            if process.poll() is not None or time.monotonic()>deadline:
                if process.poll() is None:process.terminate();process.wait(timeout=5)
                self.fail('Test worker did not acknowledge startup')
            time.sleep(.01)
        return process

    def test_mission_marker_is_required_for_owned_mission_worker(self):
        from scripts.stop_blur_worker import stop_owned_worker
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'test_mission_worker.py'
            process=self.start_worker(script,'--mission-demo','owned')
            try:
                with self.assertRaises(PermissionError):stop_owned_worker(process.pid,script,'owned')
                self.assertIsNone(process.poll())
                result=stop_owned_worker(process.pid,script,'owned',marker='--mission-demo')
                process.wait(timeout=5);self.assertTrue(result['terminated'])
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=5)

    def test_exit_between_state_and_commandline_check_is_not_a_cleanup_error(self):
        from scripts.stop_blur_worker import stop_owned_worker
        with patch('scripts.stop_blur_worker.alive', side_effect=[True, False]), \
             patch('scripts.stop_blur_worker.verify_owner', side_effect=FileNotFoundError()), \
             patch('scripts.stop_blur_worker.os.kill') as kill:
            result = stop_owned_worker(999, Path('worker.py'), 'owned')
            self.assertTrue(result['terminated'])
            kill.assert_not_called()

    def test_owned_tagged_worker_is_terminated(self):
        from scripts.stop_blur_worker import stop_owned_worker
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'test_worker.py'
            process = self.start_worker(script,'--blur-demo','owned')
            try:
                result = stop_owned_worker(process.pid, script, 'owned')
                process.wait(timeout=5)
                self.assertTrue(result['terminated'])
            finally:
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=5)

    def test_different_run_token_is_never_signalled(self):
        from scripts.stop_blur_worker import stop_owned_worker
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'test_worker.py'
            process = self.start_worker(script,'--blur-demo','other')
            try:
                with self.assertRaises(PermissionError):
                    stop_owned_worker(process.pid, script, 'owned')
                self.assertIsNone(process.poll())
            finally:
                process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
