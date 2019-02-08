from PIL import Image, ImageGrab, ImageFilter, ImageOps, ImageDraw, ImageEnhance
from os import path, listdir, mkdir
from MLClassifier import ImgRecognizer
from pytesseract import image_to_string, image_to_boxes

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

offset = 30
cPlayArea = (0, 0, 960, 1040)
data_loc = './ScreenDetection/'
base_point = (199, 264)
step = 70
pred = []

################################################################################################################


def load_classifiers():
    logging.debug("Loading classifiers for gem recognition.")
    for i in range(1):
        pred.append(ImgRecognizer())
        pred[i].train(i)
        logging.debug("Classifier %s loaded." % i)


load_classifiers()




def load_image_array():
    logging.info('Loading image data from "%s"' % data_loc)
    existing = listdir(data_loc)
    result = []
    for image in existing:
        nArray = image.replace('.png', '').split('-')
        logging.debug('Loading "%s".' % image)
        target_img = data_loc + image
        nArray.append(Image.open(target_img))
        result.insert(int(nArray[0]), [int(nArray[1]), int(nArray[2]), nArray[3]])
    logging.info("%s images loaded." % str(len(result)))
    return result


def extract_points(dataset):
    result = []
    for entry in dataset:
        result.append((entry[0], entry[1]))
    return result


def get_match(dataset, img, p):
    logging.debug('Get match logic initiated')
    for datapoint in dataset:
        logging.debug('dataset point - %s' % str((datapoint[0], datapoint[1])))
        if (datapoint[0], datapoint[1]) == p:
            logging.debug('Point found in dataset.')
            match_score = percent_same(img, datapoint[2])
            if percent_same(img, datapoint[2]) > 0.9:
                logging.debug('Found match - %s' % dataset.index(datapoint))
                result = dataset.index(datapoint)
                break
    else:
        result = None
    return result


dataset = load_image_array()


def guess_screen(img):
    logging.debug('Loading known points')
    points = extract_points(dataset)
    for p in points:
        logging.debug('Trying point - %s' % str(p))
        point = (p[0], p[1], p[0] + offset, p[1] + offset)
        result = get_match(dataset, img.crop(point), p)
        if result is not None:
            break
    else:
        result = None
    return result


def save_point(p, index):
    sc = get_screenshot()
    point = (p[0], p[1], p[0] + offset, p[1] + offset)
    tmp_img = sc.crop(point)
    tmp_name = str(index) + '-' + str(p[0]) + '-' + str(p[1])
    tmp_name += '.png'
    logging.debug('Saving point as %s' % tmp_name)
    tmp_img.save(data_loc + tmp_name, 'PNG')


def cut_game_board():
    sc = get_screenshot()
    for pos in yield_pos():
        img = sc.crop(pos_to_coords(pos))
        img = img.thumbnail((25, 25), Image.ANTIALIAS)
        img.show()


def prompt_save(img):
    base_loc = './TrainingData/'
    img.show()
    base_name = raw_input("Color nr.:")
    loc = base_loc + str(base_name) + '/'
    if not path.exists(loc):
        mkdir(loc)
    existing = listdir(loc)
    img.save(loc + str(len(existing)) + '.png', 'PNG')


def get_board(mode):
    screenshot = get_screenshot()
    result = []
    for n in xrange(8):
        result.append([])
    for cell in yield_pos():
        cimg = screenshot.crop(pos_to_coords(cell))
        cimg.thumbnail((25, 25), Image.ANTIALIAS)
        result[cell[0]].append(pred[mode].predict(cimg))
    return result


def yield_pos():
    for x in xrange(8):
        for y in xrange(8):
            yield (x, y)


def pos_to_coords(pos):
    bx = base_point[0] + pos[0] * step
    by = base_point[1] + (7 - pos[1]) * step
    bdx = bx + step
    bdy = by + step
    return (bx, by, bdx, bdy)


def get_image_for_OCR(img):
    im = ImageOps.invert(img)
    im = ImageEnhance.Contrast(im).enhance(1.4)
    im = ImageEnhance.Brightness(im).enhance(2.0)
    return im


def draw_hitboxes(img):
    draw = ImageDraw.Draw(img)
    for val in cDimensions.mainScreen.values():
        draw.rectangle(cDimensions.translate_pos(val), outline='red')
    del draw
    return img


def get_main_intel(img):
    result = {}
    d = cDimensions.mainScreen
    for k, v in d.iteritems():
        t_img = img.crop(cDimensions.translate_pos(v))
        t_img = get_image_for_OCR(t_img)
        text = image_to_string(t_img)
        result[k] = text
    return result