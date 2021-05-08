import time

from tetris.tetris import Tetris

if __name__ == '__main__':

    env = Tetris(16, 30)
    env.init_env()
    while True:
        env.step(0)
        # input()
        time.sleep(0.05)
