import imgarr as ir
import numpy as np
from PIL import Image


def test_01():
    # numpy image case
    imgs = [np.zeros((150, 100 + i * 10, 3)) + i / 5 for i in range(5)]
    ir.get_concat_horizontal(imgs)
    ir.get_concat_vertical(imgs)


def test_02():
    # numpy uint8 image case
    imgs = [np.zeros((150, 100 + i * 10, 3), dtype=np.uint8) + i / 5 * 255 for i in range(5)]
    ir.get_concat_horizontal(imgs)
    ir.get_concat_vertical(imgs)


def test_03():
    # PIL image case
    imgs = [Image.fromarray((np.zeros((150, 100 + i * 10, 3)) + i / 5 * 255).astype(np.uint8)) for i in range(5)]
    ir.get_concat_horizontal(imgs)
    ir.get_concat_vertical(imgs)
