"""The pilot's keyboard and game pad on Windows, read as states (held or not), not as typed characters.

A typed character only says a key went down, and repeats after half a second; a held key has to fly the vehicle for as
long as it is held. `GetAsyncKeyState` gives that. Keys are read only while the pilot window is the window in front, so
typing in another window never flies the vehicle. A game pad is read through XInput (an Xbox-style pad); without one the
keyboard alone flies.

Keys (the policy's three axes only):
    W / Up          forward              A D / Left Right   turn
    R / PageUp      up                   F / PageDown       down
    1 2 3           speed tier           Enter              done (approach: here; land: give up)
    Backspace       discard this flight  Esc                quit
Before a flight: N / P next or previous object, T land or approach, L next layout, M next scene, Enter fly.
Pad: left stick up forward, left stick sideways turn, right stick up and down; A done, B discard, Start fly, LB / RB tier.
"""
import ctypes
import sys
from .controls import Inputs

KEYS={'w':0x57,'a':0x41,'d':0x44,'r':0x52,'f':0x46,'n':0x4E,'p':0x50,'t':0x54,'l':0x4C,'m':0x4D,'1':0x31,'2':0x32,'3':0x33,
      'up':0x26,'down':0x28,'left':0x25,'right':0x27,'pageup':0x21,'pagedown':0x22,'enter':0x0D,'space':0x20,'backspace':0x08,'escape':0x1B}
PAD_BUTTONS={'a':0x1000,'b':0x2000,'start':0x0010,'lb':0x0100,'rb':0x0200}


class _Gamepad(ctypes.Structure):
    _fields_=[('wButtons',ctypes.c_ushort),('bLeftTrigger',ctypes.c_ubyte),('bRightTrigger',ctypes.c_ubyte),
              ('sThumbLX',ctypes.c_short),('sThumbLY',ctypes.c_short),('sThumbRX',ctypes.c_short),('sThumbRY',ctypes.c_short)]


class _State(ctypes.Structure):
    _fields_=[('dwPacketNumber',ctypes.c_uint),('Gamepad',_Gamepad)]


class Devices:
    """Keyboard and pad states, and the edges (just pressed) of the buttons that act once."""
    def __init__(self,window_title):
        self.title=window_title;self.windows=sys.platform=='win32';self.previous=set();self.pad=None
        if self.windows:
            self.user32=ctypes.windll.user32
            for name in ('xinput1_4','xinput1_3','xinput9_1_0'):
                try:self.pad=getattr(ctypes.windll,name);break
                except OSError:continue

    def focused(self):
        """Is the pilot window the window in front?"""
        if not self.windows:return False
        return self.user32.GetForegroundWindow()==self.user32.FindWindowW(None,self.title)

    def _keys(self):
        if not self.focused():return set()
        return {name for name,code in KEYS.items() if self.user32.GetAsyncKeyState(code)&0x8000}

    def _pad(self):
        if self.pad is None:return None
        state=_State()
        if self.pad.XInputGetState(0,ctypes.byref(state))!=0:return None
        return state.Gamepad

    def read(self):
        """(Inputs of the axes, set of buttons pressed since the last read, whether a pad is connected)."""
        keys=self._keys();pad=self._pad();held=set(keys)
        inputs=Inputs(forward='w' in keys or 'up' in keys,up='r' in keys or 'pageup' in keys,down='f' in keys or 'pagedown' in keys,
                      left='a' in keys or 'left' in keys,right='d' in keys or 'right' in keys)
        if pad is not None and self.focused():
            scale=lambda value:max(-1.,min(1.,value/32767.))
            inputs.stick_forward=scale(pad.sThumbLY);inputs.stick_turn=scale(pad.sThumbLX);inputs.stick_vertical=scale(pad.sThumbRY)
            held|={f'pad_{name}' for name,mask in PAD_BUTTONS.items() if pad.wButtons&mask}
        pressed=held-self.previous;self.previous=held
        return inputs,pressed,pad is not None
