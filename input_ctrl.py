import ctypes
import time

user32 = ctypes.windll.user32
PUL = ctypes.POINTER(ctypes.c_ulong)

MOUSEEVENTF_MOVE      = 0x0001
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP   = 0x0010
KEYEVENTF_KEYUP       = 0x0002

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL))

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort))

class _INPUTunion(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))

class INPUT(ctypes.Structure):
    _fields_ = (("type", ctypes.c_ulong), ("union", _INPUTunion))


def _send(inp):
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def mouse_move(dx, dy):
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi.dx = int(dx)
    inp.union.mi.dy = int(dy)
    inp.union.mi.dwFlags = MOUSEEVENTF_MOVE
    _send(inp)


def mouse_button(down: bool, left: bool = True):
    if left:
        flag = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    else:
        flag = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi.dwFlags = flag
    _send(inp)


def key_press(vk: int, hold: float = 0.05):
    down = INPUT(type=INPUT_KEYBOARD)
    down.union.ki.wVk = vk
    _send(down)
    time.sleep(hold)
    up = INPUT(type=INPUT_KEYBOARD)
    up.union.ki.wVk = vk
    up.union.ki.dwFlags = KEYEVENTF_KEYUP
    _send(up)