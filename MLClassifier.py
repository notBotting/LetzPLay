from sklearn import svm
from sklearn.externals import joblib
from PIL import Image
from cv2 import imwrite, imread

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImgRecognizer:
    def __init__(self):
        self.base_path = './TrainingData/'
        self.training_data = []
        self.target_values = []
        self.svc = svm.SVC(gamma=0.001, kernel='linear', C=100)
    
    def _load(self, path, target_value):
        training_imgs = os.listdir(path)
        for f in training_imgs:
            img = imread(path+'/'+f)
            np_arr = img.flatten()
            self.training_data.append(np_arr)
            self.target_values.append(target_value)
    
    def load(self, mode):
        if mode == 0:
            for _dir in os.listdir(self.base_path):
                _path = self.base_path + str(_dir)
                self._load(_path, int(_dir))

    def train(self, mode=0):
        a_file = 'svc' + str(mode) + '.dat'
        if os.path.isfile(a_file):
            self.svc = joblib.load(a_file)
        else:
            self.load(mode)
            np_data = np.array(self.training_data)
            np_values = np.array(self.target_values)
            self.svc.fit(np_data, np_values)
            joblib.dump(self.svc, a_file, compress=9)

    def predict(self, img):
        np_img = img.flatten().reshape(1, -1)
        return int(self.svc.predict(np_img))

    def save_img(self, img, dest):
        path = self.base_path + str(dest) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        counter = 0
        while True:
            a_file = path + str(counter) + '.png'
            if not os.path.isfile(a_file):
                imwrite(a_file, img)
                logger.info('File save as "%s"' % a_file)
                break
            counter += 1

logger.debug("Module '%s' imported" % __name__)