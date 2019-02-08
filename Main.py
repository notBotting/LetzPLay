from time import sleep

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s- %(name)-12s - %(message)s',
    level="DEBUG")

logger = logging.getLogger('Main')


def main():
    import CVision as cv
    import GameControl as game

    game.launch_game()
    hwnd = game.get_hwnd('GemsofWar')
    instance = cv.GameView(hwnd)
    ingame = 1
    puzel = cv.Puzzle()
    while True:
        instance.get_frame()
        sc = instance.ss_match(instance.last_frame)
        if sc is not False:
            if sc in [0, 1]:
                pos = cv.areas[0]
                pos = cv.get_rand_pos(pos[0], pos[1])
            elif sc in [2]:
                pos = cv.areas[1]
                pos = cv.get_rand_pos(pos[0], pos[1])
                ingame = 1
            else:
                pos = None
            if pos:
                game.left_click(hwnd, pos)
                sleep(2)
        cv.cv2.imshow('Lolzers', instance.last_frame)
        key = cv.cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            cv.cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    logger.debug('Starting "Main" program.')
    main()