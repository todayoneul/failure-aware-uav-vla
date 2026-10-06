"""Interactive overview and mission requests; file-only Windows observer."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.blur_demo_viewer import draw_view,read_input_pair,control_key_for_click
from src.failures.control import read_control,write_control
from src.mission.control import mission_key,request_selection
from src.mission.geometry import world_to_pixel

OUT=ROOT/'outputs/mission_demo';WINDOW='MISSION CONTROL | Map target + AeroVLA'
MAP_RECT=(20,214,720,405)
MISSION_BUTTONS=[(20,140,135,40,ord('g'),'G Go To'),(170,140,135,40,ord('h'),'H Hover'),
                 (320,140,135,40,ord('l'),'L Land (trial)'),(470,140,135,40,ord('r'),'R Reset')]


def map_pixel(x,y,meta):
    left,top,width,height=MAP_RECT
    if not left<=x<left+width or not top<=y<top+height:return None
    return [int((x-left)*meta['width']/width),int((y-top)*meta['height']/height)]


def render_map(image,packet,telemetry):
    shown=image.copy();meta=packet['camera'];mission=telemetry.get('mission') or {}
    def project(point):
        return tuple(int(round(v)) for v in world_to_pixel(point,meta))
    path=telemetry.get('trajectory') or []
    if len(path)>1:
        pts=np.array([project(point) for point in path],dtype=np.int32)
        cv2.polylines(shown,[pts],False,(220,120,20),2,cv2.LINE_AA)
    target=mission.get('target')
    if target:
        pixel=project(target['surface_position'])
        cv2.drawMarker(shown,pixel,(20,60,230),cv2.MARKER_TILTED_CROSS,14,2)
        cv2.putText(shown,'Target',(pixel[0]+10,pixel[1]-5),0,.45,(20,60,230),1,cv2.LINE_AA)
    state=telemetry.get('state')
    if state:
        p=state['position'];pixel=project(p)
        q=state['orientation'];yaw=math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
        end=project([p[0]+.7*math.cos(yaw),p[1]+.7*math.sin(yaw),p[2]])
        cv2.circle(shown,pixel,5,(20,160,30),-1);cv2.arrowedLine(shown,pixel,end,(20,160,30),2,tipLength=.5)
        cv2.putText(shown,'Drone',(pixel[0]+8,pixel[1]+15),0,.4,(20,140,30),1,cv2.LINE_AA)
    return shown


def render_canvas(telemetry,control,images):
    canvas=draw_view(telemetry,control,images)
    canvas[:190,:760]=(248,245,241)
    mission=telemetry.get('mission') or {};status=mission.get('state','IDLE')
    cv2.putText(canvas,'MISSION CONTROL / AeroVLA',(20,35),0,.8,(55,43,31),2,cv2.LINE_AA)
    cv2.putText(canvas,status,(20,75),0,.85,(30,130,30) if status=='SUCCESS' else (190,100,30),2,cv2.LINE_AA)
    errors=mission.get('errors');distance=f'{errors["horizontal_m"]:.2f} m' if errors else '--'
    distance_label='Distance '+distance
    if status in ('SUCCESS','FAILED','ABORTED') and mission.get('target') and telemetry.get('state'):
        target=mission['target']['surface_position'];position=telemetry['state']['position']
        live=math.hypot(target[0]-position[0],target[1]-position[1])
        distance_label=f'Final {distance} | live {live:.2f} m'
    cv2.putText(canvas,f'{mission.get("type") or "Select target"} | {distance_label}',(20,110),0,.53,(55,43,31),1,cv2.LINE_AA)
    target=mission.get("target")
    if target:
        xyz=target["surface_position"]
        cv2.putText(canvas,f"Target NED: x={xyz[0]:.2f} y={xyz[1]:.2f} z={xyz[2]:.2f} m",(20,130),0,.43,(55,43,31),1,cv2.LINE_AA)
    caps=telemetry.get('capabilities') or {}
    for left,top,width,height,key,label in MISSION_BUTTONS:
        enabled=key==ord('r') or caps.get({ord('g'):'GO_TO',ord('h'):'GO_TO_AND_HOVER',ord('l'):'GO_TO_AND_LAND'}[key],False)
        cv2.rectangle(canvas,(left,top),(left+width,top+height),(185,115,40) if enabled else (210,210,210),-1)
        cv2.putText(canvas,label,(left+8,top+25),0,.48,(255,255,255) if enabled else (80,80,80),1,cv2.LINE_AA)
    canvas[186:211,:760]=(248,245,241)
    cv2.putText(canvas,'Overview / surface click | V: view | +/-: zoom',(20,204),0,.49,(55,43,31),1,cv2.LINE_AA)
    failure=telemetry.get('failure') or {}
    text='Failure: '+('BLUR' if failure.get('failure_enabled',False) else 'NORMAL')
    cv2.putText(canvas,text,(780,142),0,.52,(20,110,210) if failure.get('failure_enabled',False) else (30,130,30),1,cv2.LINE_AA)
    note=telemetry.get('control_message') or mission.get('reason') or telemetry.get('phase','Waiting for map')
    canvas[620:640,:760]=(248,245,241)
    cv2.putText(canvas,str(note)[:84],(20,634),0,.42,(55,43,31),1,cv2.LINE_AA)
    return canvas


def main():
    control=read_control(OUT/'control.json');telemetry={};packet=None;map_image=None;images={};saved=set();last_export=0
    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1064,912);cv2.moveWindow(WINDOW,20,20)
    def send(key):
        nonlocal control
        control=mission_key(read_control(OUT/'control.json'),key);write_control(OUT/'control.json',control)
    def mouse(event,x,y,*_):
        nonlocal control
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if packet is not None:
            pixel=map_pixel(x,y,packet['camera'])
            if pixel is not None:
                control=request_selection(read_control(OUT/'control.json'),packet['frame_id'],pixel)
                write_control(OUT/'control.json',control);return
        for left,top,width,height,key,_ in MISSION_BUTTONS:
            if left<=x<left+width and top<=y<top+height:send(key);return
        key=control_key_for_click(x,y)
        if key!=-1:send(key)
    cv2.setMouseCallback(WINDOW,mouse)
    try:
        while not control['quit']:
            try:
                control=read_control(OUT/'control.json');telemetry=json.loads((OUT/'telemetry.json').read_text())
                latest=json.loads((OUT/'overview.json').read_text())
                image=cv2.imread(str(OUT/latest['image_file']))
                if image is not None and hashlib.sha256(image.tobytes()).hexdigest()==latest['image_sha256']:
                    map_image,packet=image,latest
            except (OSError,ValueError):pass
            pair=read_input_pair(OUT,telemetry)
            images.update(pair or {'front':None,'down':None})
            if map_image is not None:images['chase']=render_map(map_image,packet,telemetry)
            canvas=render_canvas(telemetry,control,images)
            cv2.imshow(WINDOW,canvas)
            if map_image is not None and time.monotonic()-last_export>1:
                cv2.imwrite(str(OUT/"current_view.png"),canvas);last_export=time.monotonic()
            status=(telemetry.get('mission') or {}).get('state','IDLE')
            label={'TARGET_SELECTED':'target_selected','NAVIGATING':'navigating','SUCCESS':'mission_success','FAILED':'mission_failed'}.get(status)
            if label and label not in saved and map_image is not None:
                cv2.imwrite(str(OUT/f'{label}.png'),canvas);saved.add(label)
            key=cv2.waitKey(40)&0xff
            if key!=255:send(key)
            if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:send(ord('q'))
    finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
