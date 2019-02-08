import webbrowser
import psutil
import logging
import win32gui
import win32api
import win32con
import numpy as np


from time import sleep

logger = logging.getLogger(__name__)


def launch_game():
    pname = "GemsOfWar.exe"
    process = list(filter(lambda p: p.name() == pname, psutil.process_iter()))
    try:
        pid = process[0].pid
        logger.info('"%s" PID(%s) detected.' % (pname, pid))
        return False
    except IndexError:
        logger.warning('No %s instance detected!' % pname)
        logger.info('Attempting to launch the game: "%s"' % pname)
        webbrowser.open('steam://rungameid/329110')
        sleep(5)
        return True


def left_click(hwnd, pos):
    x = 0.005
    client_pos = win32gui.ClientToScreen(hwnd, pos)
    cur_pos = win32api.GetCursorPos()
    win32api.SetCursorPos(client_pos)
    tmp = win32api.MAKELONG(client_pos[0], client_pos[1])
    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, tmp)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, tmp)
    sleep(x)
    win32api.SetCursorPos(cur_pos)


def GetPos(hwnd):
    print(win32gui.ScreenToClient(hwnd, win32api.GetCursorPos()))


def press_esc(hwnd):
    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_INACTIVE, 0)
    win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
    sleep(0.01)
    win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)


def get_hwnd(window_name):
    logger.debug('getting handle for window: "%s".' % window_name)
    hwnd = win32gui.FindWindow(None, window_name)
    x1, y1, x2, y2 = -7, 0, 974, 527
    win32gui.MoveWindow(hwnd, x1, y1, x2, y2, True)
    return hwnd


logger.debug("Module 'GameConroler' imported.")
