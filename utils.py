import os
import time
import imageio
import numpy as np


def read_image(filename: str):

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    img = imageio.imread(filename)
    if img.dtype == np.float32:
        return img
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        raise RuntimeError('Unexpected image data type.')


def write_image(filename: str, img, check=False):

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    if check:
        assert (np.all(img >= 0) and np.all(img <= 1))

    extension = os.path.splitext(filename)[1]
    if extension == '.exr':
        imageio.imwrite(filename, img)
    elif extension in ['.png', '.jpg']:
        imageio.imwrite(filename, (img * 255.0).astype(np.uint8))
    else:
        raise RuntimeError(f'Unexpected image filename extension {extension}.')


class Timer:
    def __init__(self):
        self.start_time = []

    def begin(self, output=''):
        if output != '':
            print(output)
        self.start_time.append(time.time())

    def end(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time.pop()
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def lap(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time[-1]
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))