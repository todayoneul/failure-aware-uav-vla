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
BUTTONS=[(20,125,110,38,ord('g'),'G Start'),(138,125,110,38,ord('r'),'R Reset'),(256,125,110,38,ord('n'),'N Target'),
         (374,125,170,38,ord('t'),'T Land / Approach'),(552,125,110,38,ord('l'),'L Launch'),(670,125,120,38,ord('f'),'F Full map'),
         (798,125,110,38,ord('c'),'C Chase'),(916,125,120,38,ord('o'),'O Overlay'),(1060,125,150,38,ord('b'),'B Toggle blur')]+\
        [(1220+i*84,125,76,38,ord(str(i+1)),label) for i,label in enumerate(('1 Low','2 Med','3 High'))]
FINALIZER_STATES=('FLYING','LANDING_DESCENT','CONTACT_CANDIDATE','STABLE_CONTACT','LANDED_LATCHED','DISARMED')
INK=(65,47,30);BLUE=(170,98,35);PURPLE=(150,75,115);EVALUATOR=(60,60,170);GREEN=(40,140,35);RED=(45,60,205);GRAY=(150,150,150)


def annotate_map(shown,packet,telemetry):
    """Names of the semantic objects and the launch pose on a map capture. Display only."""
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
    launch=telemetry.get('launch');ground=telemetry.get('ground_z')
    if launch and ground is not None:
        pixel=project([launch['xy'][0],launch['xy'][1],ground])
        if pixel is not None:
            cv2.drawMarker(shown,pixel,(150,75,115),cv2.MARKER_SQUARE,10,2)
            cv2.putText(shown,'Launch',(pixel[0]+8,pixel[1]-6),0,.36,(150,75,115),1,cv2.LINE_AA)
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
    launch=telemetry.get('launch') or {}
    line=f'Launch: {launch.get("name","--")} ({launch.get("index",0)+1}/{launch.get("count",1)})'
    if target:
        inside=target.get('in_canonical_range')
        line+=f' | start distance {target["launch_distance_m"]:.1f} m ('+('within' if inside else 'BEYOND')+f' the validated {mission.get("canonical_range_m",0):.0f} m)'
    line+=f' | success radius {mission.get("success_radius_m",0):.0f} m | step {0 if preview else telemetry.get("step",0)}/{telemetry.get("max_steps",0)}'
    text(line,20,110,.47)
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
        enabled=key!=ord('g') or (status!='NAVIGATING' and target is not None and not mission.get('refusal'))
        lit=(key==ord('f') and main=='map') or (key==ord('c') and main=='chase') or (key==ord('o') and show_overlay) or (key==ord('b') and control.get('enabled'))
        cv2.rectangle(canvas,(left,top),(left+width,top+height),PURPLE if lit else BLUE if enabled else (205,205,205),-1)
        text(label,left+8,top+25,.45,(255,255,255))
    text('Map click selects an object | WASD: pan | +/-: zoom | V: view | the map, its markers and the trajectory are for the viewer only',20,182,.45)

    # --- main panel: map or chase camera, the other one inset -----------------------------------------------------------------
    left,top,width,height=MAP_RECT;large,small=('chase_cam','chase') if main=='chase' else ('chase','chase_cam')
    if images.get(large) is not None:canvas[top:top+height,left:left+width]=cv2.resize(images[large],(width,height))
    else:
        cv2.rectangle(canvas,(left,top),(left+width,top+height),(225,225,225),-1)
        text('Waiting for the chase camera...' if main=='chase' else 'Reading scene geometry...',left+20,top+40,.7)
    text('CHASE CAMERA (simulator)' if main=='chase' else 'FULL MAP - click an object',left+8,top+20,.5,(255,255,255),2)
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
        text(f'Object {target["object_id"]} | {target.get("color")} {target.get("shape")} | landable: {"yes" if target.get("landable") else "no"}',960,494,.47,EVALUATOR)
    else:text('No target selected.',960,422,.5,EVALUATOR)
    text('Not sent: '+', '.join(policy.get('not_sent') or []),960,518,.42,EVALUATOR)
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
    text('G start | R reset to launch | N target | T task | L launch pose | F map | C chase | O overlay | B blur | Q/Esc exit',330,1030,.44)
    return canvas
