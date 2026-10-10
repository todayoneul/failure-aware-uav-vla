"""Interactive overview and mission requests; file-only Windows observer."""
import hashlib
import json
import math
import sys
import time

def log_control(source,event):
    with (OUT/"control-events.jsonl").open("a",encoding="utf-8") as file:
        file.write(json.dumps({"source":source,"epoch":time.time(),**event})+"\n")
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.blur_demo_viewer import read_input_pair
from src.failures.control import read_control,write_control
from src.mission.control import mission_key,request_selection,request_start_pose,ARROWS
from src.mission.geometry import world_to_pixel

OUT=ROOT/'outputs/mission_demo';WINDOW='MISSION CONTROL | Map target + AeroVLA'
MAP_RECT=(20,190,900,506)
MISSION_BUTTONS=[(20,125,135,38,ord('g'),'G Start'),(170,125,135,38,ord('r'),'R Reset'),
                 (320,125,135,38,ord('n'),'N Landmark'),(470,125,135,38,ord('m'),'M Prompt mode')]
PROMPT_LABELS={'hint':'Prompt: direction hint + description','description':'Prompt: description only (no hint)',
               'instruction':'Prompt: instruction only (no hint)'}

VIEW_BUTTONS=[(620,125,135,38,ord('f'),'F Full map'),(770,125,135,38,ord('c'),'C Drone')]
BLUR_BUTTONS=[(940,125,220,38,ord('b'),'B Toggle blur')]+[(1180+i*90,125,80,38,ord(str(i+1)),label) for i,label in enumerate(('1 Low','2 Med','3 High'))]


def map_pixel(x,y,meta):
    left,top,width,height=MAP_RECT
    if not left<=x<left+width or not top<=y<top+height:return None
    return [int((x-left)*meta['width']/width),int((y-top)*meta['height']/height)]


def render_map(image,packet,telemetry):
    shown=image.copy();meta=packet['camera'];mission=telemetry.get('mission') or {}
    def project(point,outside=False):
        try:pixel=world_to_pixel(point,meta)
        except ValueError:return None
        if not all(math.isfinite(v) for v in pixel):return None
        if not outside and not (0<=pixel[0]<meta['width'] and 0<=pixel[1]<meta['height']):return None
        return tuple(int(round(max(-1000000,min(1000000,v)))) for v in pixel)
    path=telemetry.get('trajectory') or []
    for previous,current in zip(path,path[1:]):
        a,b=project(previous,True),project(current,True)
        if a is None or b is None:continue
        visible,start,end=cv2.clipLine((0,0,meta['width'],meta['height']),a,b)
        if visible:cv2.line(shown,start,end,(220,120,20),2,cv2.LINE_AA)
    target=mission.get('target')
    if target:
        pixel=project(target['surface_position'])
        if pixel is not None:
            cv2.drawMarker(shown,pixel,(20,60,230),cv2.MARKER_TILTED_CROSS,14,2)
            cv2.putText(shown,'Target',(pixel[0]+10,pixel[1]-5),0,.45,(20,60,230),1,cv2.LINE_AA)
            radius=project([target['surface_position'][0]+mission.get('success_radius_m',0),*target['surface_position'][1:]],True)
            if radius is not None and 2<abs(radius[1]-pixel[1])+abs(radius[0]-pixel[0])<2000:
                cv2.circle(shown,pixel,int(math.hypot(radius[0]-pixel[0],radius[1]-pixel[1])),(20,60,230),1,cv2.LINE_AA)
    state=telemetry.get('state')
    if state:
        p=state['position'];pixel=project(p)
        if pixel is not None:
            q=state['orientation'];yaw=math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
            end=project([p[0]+.7*math.cos(yaw),p[1]+.7*math.sin(yaw),p[2]])
            cv2.circle(shown,pixel,5,(20,160,30),-1)
            if end is not None:cv2.arrowedLine(shown,pixel,end,(20,160,30),2,tipLength=.5)
            cv2.putText(shown,'Drone',(pixel[0]+8,pixel[1]+15),0,.4,(20,140,30),1,cv2.LINE_AA)
    return shown


def read_preview_pair(telemetry):
    images={}
    for name in ('front','down'):
        filename=telemetry.get('preview_files',{}).get(name)
        if not filename:return None
        frame=cv2.imread(str(OUT/filename))
        if frame is None or hashlib.sha256(frame.tobytes()).hexdigest()!=telemetry.get('preview_hashes',{}).get(name):return None
        images[name]=frame
    return images


def render_canvas(telemetry,control,images):
    from src.mission.grounding import target_overlay
    import textwrap
    canvas=np.full((1040,1480,3),(247,243,238),dtype=np.uint8)
    ink=(65,47,30);blue=(170,98,35);purple=(150,75,115)
    mission=telemetry.get('mission') or {};status=mission.get('state','IDLE')
    def text(value,x,y,scale=.5,color=ink,thickness=1):
        cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)
    text('Full map / Mission control',20,35,.9,ink,2)
    text(status,20,75,.8,(40,140,35) if status=='SUCCESS' else (45,75,200) if status=='FAILED' else blue,2)
    errors=mission.get('errors');distance=f'{errors["horizontal_m"]:.2f} m' if errors else '--'
    label='Distance '+distance
    if status in ('SUCCESS','FAILED','ABORTED') and mission.get('target') and telemetry.get('state'):
        target=mission['target']['surface_position'];position=telemetry['state']['position']
        label=f'Final {distance} / live {math.hypot(target[0]-position[0],target[1]-position[1]):.2f} m'
    target_name=(mission.get('target') or {}).get('name') or 'Select a surface or landmark'
    text(f'{target_name[:44]} | {label} | success radius {mission.get("success_radius_m",0):.0f} m',20,105,.58)
    mode=control.get('prompt_mode','hint' if control.get('direction_hint',True) else 'description')
    text(PROMPT_LABELS.get(mode,mode),940,35,.62,purple,2)
    text('Front/Down RGB + prompt -> AeroVLA; the model ends the episode',940,65,.46)
    failure=telemetry.get('failure') or {};text('Applied: '+('BLUR' if failure.get('failure_enabled') else 'NORMAL'),940,100,.6)
    flight=telemetry.get('flight_limits') or {}
    text('Flight: '+('continuous' if flight.get('continuous') else 'step by step'),1180,100,.6)
    for left,top,width,height,key,label in MISSION_BUTTONS+VIEW_BUTTONS+BLUR_BUTTONS:
        enabled=key!=ord('g') or status=='TARGET_SELECTED'
        cv2.rectangle(canvas,(left,top),(left+width,top+height),blue if enabled else (205,205,205),-1)
        text(label,left+8,top+25,.45,(255,255,255))
    text('Map click | F: full map | C: drone | WASD: pan | +/-: zoom | V: view',20,182,.47)
    if images.get('chase') is not None:
        left,top,width,height=MAP_RECT;canvas[top:top+height,left:left+width]=cv2.resize(images['chase'],(width,height))
    else:text('Reading scene geometry...',30,240,.7)
    cv2.line(canvas,(938,165),(938,696),(205,195,185),1)
    report=telemetry.get('decision_grounding') or telemetry.get('grounding_preview')
    text('Target representation',960,184,.67,ink,2)
    if report:
        text(report.get('prompt_scope','preview')+f' / step {telemetry.get("input_step",0)} / '+report.get('mode','').lower(),960,212,.44,purple)
        fmt=lambda p:' / '.join(f'{v:+.2f}' for v in p)
        rows=[('World surface XYZ',fmt(report['world_surface'])),('Navigation goal XYZ',fmt(report['navigation_goal'])),
              ('Goal relative world XYZ',fmt(report['relative_world'])),('Goal relative body XYZ',fmt(report['relative_body']))]
        y=242
        for heading,value in rows:
            text(heading,960,y,.42);text(value,960,y+21,.56);y+=52
        text(f'Horizontal distance: {report["horizontal_distance"]:.2f} m',960,454,.55)
        text(f'Bearing {report["bearing_deg"]:+.1f} deg | body {report["body_bearing_deg"]:+.1f} deg',960,480,.51)
        sent=report.get('direction_hint',True)
        text('VLA direction hint' if sent else 'Direction (evaluation only, NOT sent)',960,510,.47,purple)
        text(report['semantic_direction'] or '(at goal)',960,539,.76,purple if sent else (150,150,150),2)
        text('XYZ tokens / distance / Overview / red X: NO',960,566,.44)
        text('Actual prompt' if report.get('prompt_scope')=='actual model input' else 'Preview prompt',960,597,.48)
        prompt=report['prompt'].replace('<image>','').replace('\n',' ').strip()
        for i,line in enumerate(textwrap.wrap(prompt,68)):text(line,960,620+i*22,.44)
    else:
        text('Click a flat surface to inspect its target.',960,242,.53)
        text('Exact coordinates stay in the evaluator.',960,274,.48)
    stop=mission.get('stop')
    note=telemetry.get('control_message') or mission.get('reason') or telemetry.get('phase','Waiting')
    if stop and not telemetry.get('control_message'):
        landed={True:'landed',False:'not landed'}.get(stop.get('landed'),'landing')
        if stop.get('landed') and 'on_target_surface' in stop and not (mission.get('target') or {}).get('landmark'):
            # Only a selected surface has a height to land on; a landmark is an object to reach.
            landed='landed on the selected surface' if stop['on_target_surface'] else \
                f'landed {abs(stop["below_target_surface_m"]):.1f} m {"below" if stop["below_target_surface_m"]>0 else "above"} the selected surface'
        note=f'Model LAND at step {stop["step"]}, {stop["distance_m"]:.2f} m from target ({landed}); success radius {mission.get("success_radius_m",0):.0f} m'
    text(str(note)[:108],20,716,.47)
    visibility=images.get('visibility') or {}
    scope=images.get('camera_scope','model input')
    for name,left in (('front',20),('down',310)):
        item=visibility.get(name) or {};text(f'{name.title()} / {item.get("status","--")}',left,743,.5)
        frame=images.get(name)
        if frame is not None:canvas[754:1010,left:left+256]=target_overlay(frame,item)
        else:cv2.rectangle(canvas,(left,754),(left+256,1010),(215,215,215),-1)
    inference=telemetry.get('inference') or {};action=inference.get('parsed_action') or {};command=telemetry.get('clipped_action') or {}
    text('Last model output',610,748,.57);text('Decoded action',900,748,.57);text('Executed command',1190,748,.57)
    raw=inference.get('raw_output','Waiting...').split('Action:')[-1].strip()
    for i,line in enumerate(textwrap.wrap(raw[:150],35)):text(line,610,780+i*23,.53)
    changed=(inference.get('decoder') or {}).get('interventions') or []
    if changed:
        # The font is ASCII only, so a rejected token is shown by its vocabulary id.
        first=changed[0]
        text(f'Grammar: token {first["rejected_id"]} -> {first["chosen"]} (p {first["chosen_probability"]:.2f})',610,858,.42,purple)
    if action:
        for i,line in enumerate([f'Forward {action["fwd"]:.3f} m',f'Down {action["down"]:.3f} m',f'Yaw {math.degrees(action["yaw"]):.1f} deg']):text(line,900,780+i*26,.55)
    if command:
        d=command['displacement_ned']
        for i,line in enumerate([f'Forward {math.hypot(*d[:2]):.3f} m',f'Down {d[2]:.3f} m',f'Yaw {math.degrees(command["yaw_delta_rad"]):.1f} deg']):text(line,1190,780+i*26,.55)
    state=telemetry.get('state') or {}
    if state:text('Last pose XYZ: '+' / '.join(f'{v:+.2f}' for v in state['position']),610,894,.5)
    text('Front / Down: '+scope,610,929,.47)
    text('Point visibility uses capture pose + depth; crosshair is a display copy.',610,956,.44)
    text('Actual input: '+('VERIFIED' if telemetry.get('input_verified') else 'not inferred yet'),610,985,.5)
    text('G start | R reset | N landmark | M prompt mode | B blur | 1/2/3 severity | Q/Esc abort and land',20,1030,.46)
    return canvas


def main(output=None):
    global OUT
    if output:OUT=Path(output) if Path(output).is_absolute() else ROOT/output
    control=read_control(OUT/'control.json');telemetry={};packet=None;map_image=None;images={};saved=set();last_export=0
    # An AeroVLA-OFT policy has its own canvas: what the model is given, apart from what only the evaluator knows.
    grounding=control.get('policy')=='grounding-film';overlay=False;chase_stamp=None;buttons=MISSION_BUTTONS+VIEW_BUTTONS+BLUR_BUTTONS;window=WINDOW
    drag=None
    if grounding:
        import scripts.grounding_mission_view as view
        buttons=view.BUTTONS;window=view.WINDOW;drag=view.StartDrag()
    cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1332,936);cv2.moveWindow(window,20,20)
    def send(key):
        nonlocal control,overlay
        if grounding and key in (ord('o'),ord('O')):overlay=not overlay;return      # a choice of this window; nothing is sent
        control=mission_key(read_control(OUT/'control.json'),key);write_control(OUT/'control.json',control)
        log_control('mission viewer key/button',{'key':key,'request_id':control.get('mission_request_id')})
    def mouse(event,x,y,*_):
        nonlocal control
        if grounding and control.get('start_mode') and packet is not None and control.get('main_view')!='chase':
            # A start is being placed: the map takes the press, the drag and the release; the buttons work as ever.
            pixel=map_pixel(x,y,packet['camera']);left,top,width,height=MAP_RECT;iw,ih=view.INSET
            if pixel is not None and x<left+iw+8 and y>=top+height-ih-8:pixel=None
            if event==cv2.EVENT_LBUTTONDOWN and pixel is not None:drag.press(pixel);return
            if event==cv2.EVENT_MOUSEMOVE:drag.move(pixel);return
            if event==cv2.EVENT_LBUTTONUP:
                gesture=drag.release(pixel)
                if gesture:
                    control=request_start_pose(read_control(OUT/'control.json'),packet['frame_id'],*gesture);write_control(OUT/'control.json',control)
                    log_control('mission viewer start placement',{'frame_id':packet['frame_id'],'pixel':gesture[0],'heading_pixel':gesture[1],
                                                                  'request_id':control.get('mission_request_id')})
                return
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if packet is not None and not (grounding and control.get('main_view')=='chase'):
            pixel=map_pixel(x,y,packet['camera'])
            if grounding and pixel is not None:
                # The corner of the map that the chase-camera inset covers is not the map.
                left,top,width,height=MAP_RECT;iw,ih=view.INSET
                if x<left+iw+8 and y>=top+height-ih-8:return
            if pixel is not None:
                control=request_selection(read_control(OUT/'control.json'),packet['frame_id'],pixel)
                write_control(OUT/'control.json',control)
                log_control('mission viewer map click',{'frame_id':packet['frame_id'],'pixel':pixel,'request_id':control.get('mission_request_id')});return
        for left,top,width,height,key,_ in buttons:
            if left<=x<left+width and top<=y<top+height:send(key);return
    cv2.setMouseCallback(window,mouse)
    try:
        while not control['quit']:
            try:
                control=read_control(OUT/'control.json');telemetry=json.loads((OUT/'telemetry.json').read_text())
                latest=json.loads((OUT/'overview.json').read_text())
                image=cv2.imread(str(OUT/latest['image_file']))
                if image is not None and hashlib.sha256(image.tobytes()).hexdigest()==latest['image_sha256']:
                    map_image,packet=image,latest
            except (OSError,ValueError):pass
            preview=(telemetry.get('mission') or {}).get('state') in ('IDLE','TARGET_SELECTED')
            pair=read_preview_pair(telemetry) if preview else read_input_pair(OUT,telemetry)
            if grounding:
                # A pair is shown only when its pixels hash to what the worker says it gave the model; until then the last one stays.
                if pair:images.update(pair,verified=not preview,shown_step=telemetry.get('input_step'))
                elif images.get('shown_step')!=telemetry.get('input_step'):images['verified']=False
                chase=OUT/'chase_latest.png'
                try:
                    stamp=chase.stat().st_mtime_ns
                    if stamp!=chase_stamp:
                        frame=cv2.imread(str(chase))
                        if frame is not None:images['chase_cam']=frame;chase_stamp=stamp
                except OSError:pass
            else:images.update(pair or {'front':None,'down':None})
            images['camera_scope']='preview (not model input)' if preview else f'model input step {telemetry.get("input_step",0)}'
            images['visibility']=(telemetry.get('preview_visibility') if preview else telemetry.get('target_visibility') if telemetry.get('visibility_step')==telemetry.get('input_step') else None) or {}
            if map_image is not None:
                images['chase']=render_map(map_image,packet,telemetry)
                if grounding:view.annotate_map(images['chase'],packet,telemetry,drag.preview() if control.get('start_mode') else None)
            canvas=view.render(telemetry,control,images,overlay) if grounding else render_canvas(telemetry,control,images)
            cv2.imshow(window,canvas)
            if map_image is not None and time.monotonic()-last_export>1:
                cv2.imwrite(str(OUT/"current_view.png"),canvas);last_export=time.monotonic()
            status=(telemetry.get('mission') or {}).get('state','IDLE')
            label={'TARGET_SELECTED':'target_selected','NAVIGATING':'navigating','SUCCESS':'mission_success','FAILED':'mission_failed'}.get(status)
            if label and label not in saved and map_image is not None:
                cv2.imwrite(str(OUT/f'{label}.png'),canvas);saved.add(label)
            if grounding:
                # The arrow keys pan the map (S places a start for this policy); they arrive as codes of more than one byte.
                code=cv2.waitKeyEx(40);key=code if code in ARROWS else 255 if code==-1 else code&0xff
            else:key=cv2.waitKey(40)&0xff
            if key!=255:send(key)
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:send(ord('q'))
    finally:cv2.destroyAllWindows()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',help='run folder; default outputs/mission_demo')
    main(parser.parse_args().output)
