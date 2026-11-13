"""Windows-visible debug panel; reads atomically published integration frames."""
import time
import argparse
from pathlib import Path
import cv2
parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/integration')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/args.output
window='AeroVLA x Project AirSim | Minimal integration'
cv2.namedWindow(window,cv2.WINDOW_NORMAL)
cv2.resizeWindow(window,640,520)
cv2.moveWindow(window,650,40)
last=None
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    path=root/'debug_latest.png'
    if path.exists() and path.stat().st_mtime_ns!=last:
        frame=cv2.imread(str(path))
        if frame is not None:
            cv2.imshow(window,frame)
            last=path.stat().st_mtime_ns
    if cv2.waitKey(50)&0xff==27: break
    time.sleep(.05)
cv2.destroyAllWindows()
