"""Mission Control canvas for an AeroVLA-OFT policy: the map and the evaluator's numbers beside what the model is given.

Windows side; draws from files only. Everything about the target's place is drawn in the evaluator's colour under a
heading that says it is not sent to the model. The two camera panels show the arrays the policy received, with nothing
drawn on them unless the display-only overlay is switched on.
"""
import math
import textwrap
import cv2
import numpy as np
from src.failures.gaussian_blur import SEVERITIES
from src.mission.geometry import world_to_pixel

WINDOW='MISSION CONTROL | AeroVLA-OFT grounding_film'
MAP_RECT=(20,190,900,506)
INSET=(280,158)
def _row(items,left=20,top=125,height=38,gap=6):
    row=[]
    for key,label,width in items:
        row.append((left,top,width,height,ord(key),label));left+=width+gap
    return row


BUTTONS=_row([('g','G Start',92),('r','R Reset',92),('n','N Target',100),('t','T Land/Approach',160),('s','S Place start',134),('l','L Preset',96),
              ('f','F Map',80),('c','C Chase',92),('o','O Overlay',104),('b','B Blur',82),('1','1 Low',72),('2','2 Med',72),('3','3 High',76)])
FINALIZER_STATES=('FLYING','LANDING_DESCENT','CONTACT_CANDIDATE','STABLE_CONTACT','LANDED_LATCHED','DISARMED')
INK=(65,47,30);BLUE=(170,98,35);PURPLE=(150,75,115);EVALUATOR=(60,60,170);GREEN=(40,140,35);RED=(45,60,205);GRAY=(150,150,150)
START=(190,40,170);INVALID=(30,30,235)
START_ARROW_M=10.


class StartDrag:
    """The mouse on the map while a start is being placed: pressed where the vehicle is to stand, dragged toward what it is to
    face, released to send both. A press and a release without a drag send the place and keep the heading. Pixels are the
    map capture's own."""
    MINIMUM_PX=10

    def __init__(self):self.origin=None;self.current=None

    def press(self,pixel):self.origin=list(pixel);self.current=list(pixel)

    def move(self,pixel):
        if self.origin is not None and pixel is not None:self.current=list(pixel)

    def dragged(self,pixel):
        return self.origin is not None and math.hypot(pixel[0]-self.origin[0],pixel[1]-self.origin[1])>=self.MINIMUM_PX

    def release(self,pixel=None):
        """(place pixel, heading pixel or None) of the gesture that just ended, or None when none was begun."""
        if self.origin is None:return None
        end=list(pixel) if pixel is not None else self.current;origin=self.origin;far=self.dragged(end)
        self.origin=self.current=None
        return origin,(end if far else None)

    def preview(self):
        """The arrow to draw while the button is held, or None."""
        return (self.origin,self.current) if self.origin is not None and self.dragged(self.current) else None


def start_lines(start):
    return ['START' if start.get('applied') else 'START (not flown to yet)',f'Height {start["height_m"]:.1f} m',f'Yaw {start["yaw_deg"]:+.0f} deg']


def annotate_map(shown,packet,telemetry,drag=None):
    """Names of the semantic objects and the start pose on a map capture. Display only."""
    meta=packet['camera']
    def project(point):
        try:pixel=world_to_pixel(point,meta)
        except ValueError:return None
        if not all(math.isfinite(v) for v in pixel) or not (0<=pixel[0]<meta['width'] and 0<=pixel[1]<meta['height']):return None
        return int(round(pixel[0])),int(round(pixel[1]))
    selected=((telemetry.get('mission') or {}).get('target') or {}).get('object_id')
    for item in telemetry.get('targets') or []:
        pixel=project(item['position'])
        if pixel is None or item['object_id']==selected:continue
        cv2.circle(shown,pixel,3,(40,40,40),-1,cv2.LINE_AA)
        # A light plate keeps a name readable over a dark roof or a shadow.
        (width,height),_=cv2.getTextSize(item['name'],0,.36,1);x,y=pixel[0]+7,pixel[1]+4
        cv2.rectangle(shown,(x-2,y-height-2),(x+width+2,y+3),(235,235,235),-1)
        cv2.putText(shown,item['name'],(x,y),0,.36,(30,30,30),1,cv2.LINE_AA)
    start=telemetry.get('start');ground=telemetry.get('ground_z')
    if start:
        # The start of the next mission: a diamond with an arrow along its heading, apart from the target's cross and the
        # vehicle's dot. Red when a flight may not start there.
        level=ground if ground is not None else 0.;colour=START if start.get('valid',True) else INVALID
        pixel=project([start['x'],start['y'],level]);turn=math.radians(start['yaw_deg'])
        ahead=project([start['x']+START_ARROW_M*math.cos(turn),start['y']+START_ARROW_M*math.sin(turn),level])
        if pixel is not None:
            cv2.drawMarker(shown,pixel,colour,cv2.MARKER_DIAMOND,14,2,cv2.LINE_AA)
            if ahead is not None:cv2.arrowedLine(shown,pixel,ahead,colour,2,cv2.LINE_AA,tipLength=.3)
            lines=start_lines(start) if start.get('valid',True) else ['INVALID START']
            width=max(cv2.getTextSize(line,0,.36,1)[0][0] for line in lines);x,y=pixel[0]+10,pixel[1]+16
            cv2.rectangle(shown,(x-2,y-11),(x+width+2,y+13*(len(lines)-1)+4),(235,235,235),-1)
            for index,line in enumerate(lines):cv2.putText(shown,line,(x,y+13*index),0,.36,colour,1,cv2.LINE_AA)
    if drag:
        # While the button is held: from the place pressed toward where the pointer is.
        origin,current=drag
        cv2.arrowedLine(shown,tuple(map(int,origin)),tuple(map(int,current)),START,2,cv2.LINE_AA,tipLength=.25)
    return shown


def overlay(frame,pixel,seen):
    """A crosshair where the evaluator knows the target to be, on a copy. The array the model was given is not touched."""
    shown=frame.copy()
    if pixel and 0<=pixel[0]<frame.shape[1] and 0<=pixel[1]<frame.shape[0]:
        cv2.drawMarker(shown,(int(round(pixel[0])),int(round(pixel[1]))),(45,165,40) if seen else (30,145,235),cv2.MARKER_CROSS,13,1,cv2.LINE_AA)
    return shown


def render(telemetry,control,images,show_overlay=False):
    canvas=np.full((1040,1480,3),(247,243,238),dtype=np.uint8)
    mission=telemetry.get('mission') or {};policy=telemetry.get('policy') or {};status=mission.get('state','IDLE')
    target=mission.get('target');task=(mission.get('task') or 'land').upper()
    flying=status=='NAVIGATING';preview=status in ('IDLE','TARGET_SELECTED')
    def text(value,x,y,scale=.5,color=INK,thickness=1):
        cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)
    def wrapped(value,x,y,width,scale=.46,color=INK,thickness=1,step=21,limit=4):
        lines=textwrap.wrap(str(value),width)[:limit]
        for i,line in enumerate(lines):text(line,x,y+i*step,scale,color,thickness)
        return y+len(lines)*step

    # --- header ---------------------------------------------------------------------------------------------------------
    text('Mission Control',20,35,.9,INK,2)
    text(status,20,75,.8,GREEN if status=='SUCCESS' else RED if status in ('FAILED','ABORTED') else BLUE,2)
    text(f'Target: {target["name"] if target else "none"}',300,62,.62,INK,2)
    text(f'Task: {task}',300,88,.62,PURPLE,2)
    launch=telemetry.get('launch') or {};start=telemetry.get('start') or {};placing=bool(start.get('placing') or control.get('start_mode'))
    preset=launch.get('index')
    line='Start: '+(f'{launch.get("name","--")} (preset {preset+1}/{launch.get("count",1)})' if preset is not None else 'CUSTOM')
    if start:line+=f' | x {start["x"]:+.1f}, y {start["y"]:+.1f} | yaw {start["yaw_deg"]:+.0f} deg | height {start["height_m"]:.1f} m'
    if start and not start.get('valid',True):
        text(line,20,100,.45);text(('INVALID START. Reason: '+'; '.join(start.get('reasons') or []))[:118],20,116,.45,RED,1)
    else:
        text(line+('' if start.get('applied',True) else ' | not flown to yet: S, G or R moves the vehicle there'),20,100,.45)
        line=f'success radius {mission.get("success_radius_m",0):.0f} m | step {0 if preview else telemetry.get("step",0)}/{telemetry.get("max_steps",0)}'
        if target:
            inside=target.get('in_canonical_range')
            line=f'start distance {target["launch_distance_m"]:.1f} m ('+('within' if inside else 'BEYOND')+f' the validated {mission.get("canonical_range_m",0):.0f} m) | '+line
        text(line,20,116,.45)
    loaded=policy.get('loaded');kind=str(policy.get('grounding','none'))
    text(f'Model: {policy.get("model","--")}',940,35,.66,PURPLE,2)
    if loaded:
        parts=['LoRA','action head']+(['FiLM' if kind=='film' else kind] if loaded.get('grounding_loaded') else [])
        text('Loaded from checkpoint: '+' + '.join(parts)+f' | chunk {loaded["chunk_size"]}, execute {loaded["execute_horizon"]}',940,60,.45)
    else:text('Checkpoint loading...',940,60,.47,BLUE)
    frozen=str(policy.get('frozen','--')).upper()
    text(f'Grounding: {"FiLM" if kind=="film" else kind} | frozen baseline: {frozen}',940,82,.47,GREEN if frozen=='VERIFIED' else RED)
    # The failure the model's input is under now: what the last decision was given, or between flights what the keys ask for.
    failure=telemetry.get('failure') or {};blur=failure.get('failure_enabled') if not preview else control.get('enabled')
    severity=str(failure.get('severity') if not preview else control.get('severity','')).lower()
    text('Prompt mode: instruction-only (frozen)',940,104,.47,PURPLE)
    if blur:
        kernel,sigma=SEVERITIES.get(severity,(failure.get('kernel'),failure.get('sigma')))
        text('FAILURE: GAUSSIAN BLUR',640,62,.6,RED,2)
        text(f'Severity: {severity.upper()} | Kernel: {kernel} x {kernel} | Sigma: {sigma}',640,88,.42,RED,1)
    else:text('Failure: NORMAL',640,62,.6,GREEN,2)

    # --- buttons --------------------------------------------------------------------------------------------------------
    main=control.get('main_view','map')
    for left,top,width,height,key,label in BUTTONS:
        enabled=key!=ord('g') or (status!='NAVIGATING' and target is not None and not mission.get('refusal') and start.get('valid',True))
        lit=(key==ord('f') and main=='map') or (key==ord('c') and main=='chase') or (key==ord('o') and show_overlay) or (key==ord('b') and control.get('enabled')) \
            or (key==ord('s') and placing)
        cv2.rectangle(canvas,(left,top),(left+width,top+height),PURPLE if lit else BLUE if enabled else (205,205,205),-1)
        text(label,left+8,top+25,.45,(255,255,255))
    if placing:text('START PLACEMENT MODE | map click: where | drag: which way | [ ]: height | J K: yaw | S: done (the vehicle moves there) | L: back to a preset',20,182,.45,START,1)
    else:text('Map click selects an object | S: place the start | arrows or W A D: pan | +/-: zoom | V: view | map, markers and trajectory are for the viewer only',20,182,.45)

    # --- main panel: map or chase camera, the other one inset -----------------------------------------------------------------
    left,top,width,height=MAP_RECT;large,small=('chase_cam','chase') if main=='chase' else ('chase','chase_cam')
    if images.get(large) is not None:canvas[top:top+height,left:left+width]=cv2.resize(images[large],(width,height))
    else:
        cv2.rectangle(canvas,(left,top),(left+width,top+height),(225,225,225),-1)
        text('Waiting for the chase camera...' if main=='chase' else 'Reading scene geometry...',left+20,top+40,.7)
    label='CHASE CAMERA (simulator)' if main=='chase' else 'START PLACEMENT MODE - click the start, drag its heading' if placing else 'FULL MAP - click an object'
    if placing and main!='chase':cv2.rectangle(canvas,(left,top),(left+width-1,top+height-1),START,3)
    text(label,left+8,top+20,.5,(255,255,255),2)
    if images.get(small) is not None:
        iw,ih=INSET;x0,y0=left+8,top+height-ih-8          # bottom left: the corner of the map with nothing to select in it
        canvas[y0:y0+ih,x0:x0+iw]=cv2.resize(images[small],(iw,ih));cv2.rectangle(canvas,(x0-1,y0-1),(x0+iw,y0+ih),(255,255,255),2)
        text('map' if main=='chase' else 'chase camera',x0+6,y0+16,.42,(255,255,255),1)
    cv2.line(canvas,(938,165),(938,700),(205,195,185),1)

    # --- what the model is given ----------------------------------------------------------------------------------------
    text('MODEL INPUT',960,186,.7,PURPLE,2)
    text(policy.get('model_input','Front RGB + Down RGB + Instruction ONLY'),960,210,.52,PURPLE,2)
    sent=telemetry.get('model_input') if not preview else None
    sentence=(sent or {}).get('instruction') or mission.get('instruction')
    if mission.get('refusal') and target:
        # Nothing would be sent: G is refused, and the reason stands where the sentence would.
        y=wrapped(mission['refusal'],960,240,50,.5,RED,2,23,3);text('no sentence is sent; G is disabled',960,y+2,.43,RED)
    elif sentence:
        y=wrapped('"'+sentence+'"',960,240,44,.6,INK,2,25,3)
        scope=f'actual model input, step {sent["step"]}' if sent else 'preview: the sentence G would send'
        text(scope+' | no hint, no coordinates',960,y+2,.43,PURPLE)
    else:text('Select an object: the sentence appears here.',960,240,.5)
    cv2.line(canvas,(960,348),(1460,348),(205,195,185),1)

    # --- what only the evaluator and this display know ------------------------------------------------------------------------
    text('EVALUATOR / UI ONLY',960,372,.66,EVALUATOR,2)
    text('Target XYZ / Distance / Bearing  -  NOT SENT TO MODEL',960,395,.47,EVALUATOR,1)
    live=telemetry.get('evaluator') if not preview else None
    if target:
        fmt=lambda p:' / '.join(f'{v:+.2f}' for v in p)
        text('Target XYZ   '+fmt(target['surface_position']),960,422,.5,EVALUATOR)
        if live:
            text(f'Distance {live["distance_m"]:.1f} m | bearing {live["bearing_deg"]:+.1f} deg | height {live["height_m"]:.1f} m',960,446,.5,EVALUATOR)
            seen=lambda flag:'SEEN' if flag else 'not seen'
            text(f'Target in Front view: {seen(live["front_seen"])} | in Down view: {seen(live["down_seen"])}',960,470,.5,EVALUATOR)
        elif mission.get('errors'):
            errors=mission['errors']
            text(f'Distance {errors["horizontal_m"]:.1f} m | bearing {errors["bearing_deg"]:+.1f} deg',960,446,.5,EVALUATOR)
            visibility=telemetry.get('preview_visibility') or {}
            text('Front: '+str((visibility.get('front') or {}).get('status','--'))+' | Down: '+str((visibility.get('down') or {}).get('status','--')),960,470,.5,EVALUATOR)
        surface=target.get('surface') or {};kind=str(surface.get('surface','--')).upper()
        if surface.get('outline')=='circle':usable=f' | Usable radius: {surface["usable_radius_m"]:.1f} m'
        elif surface.get('outline')=='rectangle':usable=' | Usable half-size: '+' x '.join(f'{value:.1f}' for value in surface['usable_half_extent_m'])+' m'
        else:usable=''
        usable=usable if target.get('landable') else ''
        text(f'{target["object_id"]} | Surface: {kind} | Landable: {"YES" if target.get("landable") else "NO"}{usable}',960,494,.44,EVALUATOR)
    else:text('No target selected.',960,422,.5,EVALUATOR)
    wrapped('Not sent: '+', '.join(policy.get('not_sent') or []),960,511,80,.38,EVALUATOR,1,14,2)
    cv2.line(canvas,(960,534),(1460,534),(205,195,185),1)

    # --- result ---------------------------------------------------------------------------------------------------------
    text('RESULT',960,558,.62,INK,2)
    result=mission.get('result');contact=telemetry.get('contact') or {}
    if contact.get('collision'):text('COLLISION: '+str(contact['collision'])[:34],960,584,.62,RED,2)
    elif contact.get('touchdown_on'):
        offset=contact.get('touchdown_offset_m') or [0,0]
        text(f'Touchdown on {contact["touchdown_on"]}, {math.hypot(*offset):.1f} m from its centre',960,584,.5,GREEN if contact['touchdown_on']==(target or {}).get('object_id') else RED,2)
    elif flying:text('Flying...',960,584,.5)
    if result:
        y=wrapped(('SUCCESS: ' if result['success'] else 'FAILED: ')+str(result['text']),960,610,52,.48,GREEN if result['success'] else RED,1,21,3)
        ended=result.get('selected') or 'no object'
        text(f'Ended at: {ended} | policy stopped by itself: {"yes" if result.get("stopped") else "no"} | {result.get("steps")} decisions',960,y+4,.42)
        text('A demonstration, not a benchmark: the canonical test result (47/48) is unchanged.',960,y+26,.42,GRAY)
    elif mission.get('reason'):wrapped(mission['reason'],960,610,52,.48,RED,1,21,3)
    timing=telemetry.get('timing') or {}
    if timing.get('decision_s'):
        text(f'Decision period {timing["decision_s"]:.2f} s (tick {policy.get("tick_s",0.5)} s) | inference {timing.get("inference_ms") or 0:.0f} ms',960,698,.42,GRAY)

    # --- bottom row: the two inputs, the action, the command, the finalizer ---------------------------------------------------------
    note=telemetry.get('control_message') or telemetry.get('phase','Waiting')
    if telemetry.get('status') in ('FAIL','STOPPED'):note=telemetry['status']+' / '+str(note)
    text(str(note)[:120],20,720,.5,RED if telemetry.get('control_message') else BLUE,1)
    step=telemetry.get('input_step',0)
    for name,left in (('front',20),('down',310)):
        label=f'{name.title()} - preview (not model input)' if preview else f'{name.title()} - MODEL INPUT step {step}'
        text(label,left,746,.47,INK if preview else PURPLE,1 if preview else 2)
        frame=images.get(name)
        if frame is not None:
            if show_overlay:
                if live:frame=overlay(frame,live.get(f'{name}_pixel'),live.get(f'{name}_seen'))
                elif preview:
                    item=(telemetry.get('preview_visibility') or {}).get(name) or {}
                    frame=overlay(frame,item.get('pixel') if item.get('in_fov') else None,item.get('visible'))
            canvas[754:1010,left:left+256]=frame
        else:cv2.rectangle(canvas,(left,754),(left+256,1010),(215,215,215),-1)
    blurred=bool(failure.get('failure_enabled')) and not preview
    if blurred and not show_overlay:
        text('blurred frames are the MODEL INPUT',20,1023,.4,RED);text('RAW kept for evaluator/debug only',20,1037,.4,RED)
    else:text('overlay is display-only' if show_overlay else 'raw frames: nothing is drawn on them',20,1026,.42,EVALUATOR if show_overlay else GRAY)
    inference=telemetry.get('inference') or {};action=inference.get('action');executed=telemetry.get('executed')
    text('Model action (decoded)',610,748,.56,INK,2);text('Executed command',900,748,.56,INK,2);text('Landing finalizer',1190,748,.56,INK,2)
    def axes(values,x,y):
        for i,line in enumerate([f'Forward {values[0]:+.3f} m',f'Down    {values[1]:+.3f} m',f'Yaw     {math.degrees(values[2]):+.1f} deg']):text(line,x,y+i*26,.55)
    if action:
        axes(action,610,780)
        text('head output [-1,1]: '+' '.join(f'{v:+.2f}' for v in inference.get('normalised') or []),610,862,.42)
        text(f'chunk of {len(inference.get("chunk") or [])}, first action executed',610,882,.42)
        text('near-zero action (stop): '+('YES' if inference.get('stop') else 'no'),610,902,.45,PURPLE if inference.get('stop') else INK)
    else:text('Waiting...',610,780,.5)
    final=telemetry.get('finalizer') or {}
    if executed:
        axes(executed,900,780)
        if action and final.get('finalizer_triggered') and any(abs(v)>1e-9 for v in action) and not any(abs(v)>1e-9 for v in executed):
            text('landing latched: command suppressed',900,862,.42,PURPLE)
        text('continuous velocity command per tick',900,882,.42,GRAY)
    if task=='APPROACH' or (final and not final.get('active')):
        text('DISABLED',1190,780,.6,GRAY,2);text('the sentence asks for no landing',1190,804,.42,GRAY)
    else:
        current=final.get('state','FLYING') if final else None;path=final.get('path') or []
        for i,name in enumerate(FINALIZER_STATES):
            colour=PURPLE if name==current else INK if name in path else GRAY
            text(('> ' if name==current else '  ')+name,1190,778+i*23,.47,colour,2 if name==current else 1)
        if telemetry.get('disarmed') is not None:text('motors off: '+('yes' if telemetry['disarmed'] else 'FAILED'),1190,778+6*23,.45,GREEN if telemetry['disarmed'] else RED)
    text('descent flown by the policy; simulator landing routine NOT USED',1190,940,.36,GRAY)
    hashes=(telemetry.get('model_input') or {}).get('sha256') or {}
    if hashes and not preview:
        text(f'input sha256  front {hashes["front"][:12]}  down {hashes["down"][:12]}',610,940,.42)
        unchanged=(telemetry.get('failure') or {}).get('camera_frames_unchanged')
        text('shown image = model input: '+('VERIFIED' if images.get('verified') else 'checking')+' | camera frame unchanged: '+('YES' if unchanged else 'NO (blur on)'),610,962,.42,
             GREEN if images.get('verified') and unchanged else INK)
    if telemetry.get('input_verified') is not None:
        text('log check, all steps: evaluation loop hash = displayed hash: '+('YES' if telemetry['input_verified'] else 'NO'),610,984,.42,GREEN if telemetry['input_verified'] else RED)
    text('G start | R reset to the start | N target | T task | S place start | L preset | F map | C chase | O overlay | B blur | Q/Esc exit',330,1030,.44)
    return canvas
