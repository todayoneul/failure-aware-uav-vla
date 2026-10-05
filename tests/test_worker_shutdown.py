import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


@unittest.skipUnless(os.name == 'posix' and Path('/proc').exists(), 'WSL/Linux process ownership test')
class WorkerShutdownTests(unittest.TestCase):
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
            script.write_text('import time; time.sleep(60)')
            process = subprocess.Popen([sys.executable, str(script), '--blur-demo', '--run-token', 'owned'])
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
            script.write_text('import time; time.sleep(60)')
            process = subprocess.Popen([sys.executable, str(script), '--blur-demo', '--run-token', 'other'])
            try:
                with self.assertRaises(PermissionError):
                    stop_owned_worker(process.pid, script, 'owned')
                self.assertIsNone(process.poll())
            finally:
                process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
