import win32gui
import cv2
import glob
import numpy as np
import logging
import pickle
import MLClassifier as ml

from PIL import ImageGrab
from os import listdir
from random import randrange
from pytesseract import image_to_data, image_to_string
from time import sleep

logger = logging.getLogger(__name__)
template_path = './ScreenDetection/Templates/'

# Pickle vars
logger.debug('Loading pickled variables.')
with open('fullScreenClicks', 'rb') as f:
    clicks = pickle.load(f)

areas = [
    ((340, 126), (304, 251)),
    ((563, 441), (241, 41))
]

class GameView:
    def __init__(self, hwnd):
        self.name = "GemsofWar"
        self.hwnd = hwnd
        self.dimension = win32gui.GetWindowRect(hwnd)
        self.scope = 30
        self.game_states = {
                                0: 'Just started',
                                1: 'Unknown',
                                2: 'In puzzle',
                                3: 'In main window'
                            }
        self.template_path = './ScreenDetection/Templates/'
        self.templates = []
        self.last_frame = None
        self.load_ss_templates()
        self.sub = cv2.createBackgroundSubtractorKNN(history=5)

    def get_ss(self, x, y):
        def nothing(x):
            pass
        offset = 8
        header = 31
        l, t, r, b = self.dimension
        d = (l + offset, t + header, r - offset, b - offset)
        img = ImageGrab.grab(d).convert('RGB')
        img_np = np.array(img)
        y -= header
        x -= 1
        img = img_np[y:y + self.scope, x:x + self.scope]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow('ss', flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ss', 600, 600)
        cv2.createTrackbar('threshold', 'ss', 0, 255, nothing)
        t = 0
        while True:
            _, img_tresh = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            cv2.imshow('ss', img_tresh)
            t = cv2.getTrackbarPos('threshold', 'ss')
            key = cv2.waitKey(27)
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        _l = len(listdir(self.template_path))
        temp_name = '-'.join([str(_l), str(x), str(y), str(t)])
        temp_name = self.template_path + temp_name + '.png'
        cv2.imwrite(temp_name, img_tresh)

    def load_ss_templates(self):
        self.templates = []
        pickles = glob.glob(self.template_path + '*')
        for p in pickles:
            tmp = p.replace('.png', '').split('-')
            tmp.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
            self.templates.append(tmp)
        logging.debug('%s templates loaded.' % len(self.templates))

    def ss_match(self, frame):
        for tmpl in self.templates:
            x = int(tmpl[1])
            y = int(tmpl[2])
            t = int(tmpl[3])
            img1 = frame[y:y + self.scope, x:x + self.scope]
            _, img1 = cv2.threshold(img1, t, 255, cv2.THRESH_BINARY)
            img2 = tmpl[-1]
            result = np.subtract(img1, img2)
            match_score = np.count_nonzero(result==0) / np.size(img1)
            if match_score >= 0.97:
                tmp = tmpl[0].split('\\')[-1]
                logger.debug('Screen matched to template: %s, Score: %s' % (tmp, match_score))
                return int(tmp)
        else:
            return None

    def yield_mask(self):
        frame = self.last_frame
        mask = self.sub.apply(frame)
        return mask


    def get_frame(self):
        offset = 8
        header = 31
        l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        d = (l + offset, t + header, r - offset, b - offset)
        img = ImageGrab.grab(d).convert('RGB')
        img_np = np.array(img)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        self.last_frame = img


def get_rand_pos(p1, size):
    margin = 2
    w, h = size
    w -= margin
    h -= margin
    x1, y1 = p1[0] - 1, p1[1] - 31
    if w > margin and h > margin:
        x_offset = margin if randrange(w) < margin else randrange(w)
        y_offset = margin if randrange(h) < margin else randrange(h)
        return x1 + x_offset, y1 + y_offset
    else:
        return False


class Puzzle:
    def __init__(self):
        self.unit_dim = (144, 107)
        self.grid = {
            'base_y': 63 - 31,
            'base_x': 116 - 1,
            'x_step': 584,
            'y_step': 115,
            'own_mana_ROI': (0, 0, 30, 19),
            'own_attack_ROI': (24, 80, 44, 19),
            'own_def_ROI': (80, 55, 44, 19),
            'own_life_ROI': (80, 80, 44, 19)
        }
        self.units = []
        self.enemy = []
        self.layout = []
        self.pred = []
        self.load_classifiers()

    def get_units(self, frame):
        x = self.grid['base_x']
        y = self.grid['base_y']
        images = []
        for n in range(2):
            for m in range(4):
                x1 = x + self.grid['x_step'] * n
                x2 = x1 + self.unit_dim[0]
                y1 = y + self.grid['y_step'] * m
                y2 = y1 + self.unit_dim[1]
                images.append(frame[y1:y2, x1:x2])
        return images

    def get_layout(self):
        x, y = (264, 67)
        z = 54
        d = (x, y, x + z * 8, y + z * 8)
        img = ImageGrab.grab(d).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.hsplit(img, 8)
        result = []
        for arr in img:
            result.append(np.vsplit(arr, 8))
        self.layout = result

    def load_classifiers(self):
        self.pred = []
        for i in range(1):
            self.pred.append(ml.ImgRecognizer())
            self.pred[i].train(i)

    def parse_layout(self, mode):
        temp = []
        for x in range(8):
            col = []
            for y in range(7, -1, -1):
                img = cv2.resize(self.layout[x][y], (30, 30))
                col.append(self.pred[mode].predict(img))
            temp.append(col)
        self.layout = temp



logger.debug("Module 'CVision' imported")
