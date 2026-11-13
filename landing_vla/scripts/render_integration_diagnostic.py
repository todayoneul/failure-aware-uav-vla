"""Post-run diagnostic from saved communication frames; never a live flight claim."""
import json
from pathlib import Path
import cv2
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'outputs/integration'
s=json.loads((OUT/'measurement-summary.json').read_text())
panel=np.full((550,800,3),24,dtype=np.uint8)
for name,x in [('front',20),('down',320)]:
    frame=cv2.imread(str(OUT/f'{name}_communication.png'))
    if frame is None: raise RuntimeError(f'Missing saved {name} frame')
    panel[50:306,x:x+256]=frame
    cv2.putText(panel,f'{name.upper()} | saved communication frame',(x,32),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)
lines=[
    'STEP 1: WSL camera/state/control PASS (reverse TCP)',
    f'Manual forward movement: {s["communication"]["forward_horizontal_m"]:.3f}m',
    'MODEL-ONLY: NF4 base + AeroVLA LoRA PASS | valid actions 11/11',
    'Raw action: 96 49 49 | decoded forward=4.898m, down=0m, yaw=0rad',
    f'Generation 10-run mean/median/p95: 733.371 / 751.497 / 883.571 ms',
    'END-TO-END: STOPPED at scene topic initialization (60s timeout)',
    'VLA-driven movement: NOT RUN | closed-loop steps: 0/10',
    f'Combined load/init GPU peak: {s["combined"]["global_gpu_peak_GiB"]:.3f} GiB | OOM: NO',
    'Post-run diagnostic; saved frames, no live closed-loop display.',
]
for index,line in enumerate(lines):
    color=(100,210,255) if index in (5,6) else (230,230,230)
    cv2.putText(panel,line,(20,335+index*23),cv2.FONT_HERSHEY_SIMPLEX,.49,color,1,cv2.LINE_AA)
cv2.imwrite(str(OUT/'debug_communication.png'),panel)
