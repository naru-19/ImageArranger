import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import imgarr.interactive_plot as iplt


# test with PIL type imgs
def test_01():
    imgs0 = [
        Image.fromarray((np.zeros((150, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs1 = [
        Image.fromarray(((np.ones((100, 150, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    ifig = iplt.show([imgs0, imgs1], setFrame=True)
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)
    
    # test with PIL odd size imgs
def test_01_odd():
    imgs0 = [
        Image.fromarray((np.zeros((151, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs1 = [
        Image.fromarray(((np.ones((100, 150, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    ifig = iplt.show([imgs0, imgs1], setFrame=True)
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# test with numpy type imgs
def test_02():
    imgs0 = [np.zeros((150, 100, 3)) + i / 5 for i in range(5)]
    imgs1 = [np.ones((100, 150, 3)) - 0.2 - i / 5 for i in range(5)]
    ifig = iplt.show([imgs0, imgs1], setFrame=True)
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# test with numpy type imgs v2
def test_03():
    imgs0 = [np.zeros((150, 100, 3), dtype=np.uint8) + i / 5 * 255 for i in range(5)]
    imgs1 = [
        (np.ones((100, 150, 3), dtype=np.uint8) * 255 - 30 - i * 30) for i in range(5)
    ]
    ifig = iplt.show([imgs0, imgs1], setFrame=True)
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# layout test with pil img type.
def test_04():
    imgs0 = [
        Image.fromarray((np.zeros((150, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs1 = [
        Image.fromarray(((np.ones((100, 150, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs2 = [
        Image.fromarray(((np.ones((80, 80, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs3 = [
        Image.fromarray((np.zeros((80, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    ifig = iplt.show([imgs0, imgs1, imgs2, imgs3], setFrame=False, layout=(2, 2))
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# layout test with pil img type v2.
def test_05():
    imgs0 = [
        Image.fromarray((np.zeros((150, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs1 = [
        Image.fromarray(((np.ones((100, 150, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs2 = [
        Image.fromarray(((np.ones((80, 80, 3)) - 0.2 - i / 5) * 255).astype(np.uint8))
        for i in range(5)
    ]
    imgs3 = [
        Image.fromarray((np.zeros((80, 100, 3)) + i / 5 * 255).astype(np.uint8))
        for i in range(5)
    ]
    ifig = iplt.show([imgs0, imgs1, imgs2, imgs3], setFrame=True, layout=(3, 2))
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# layout test with numpy type imgs.
def test_06():
    imgs0 = [np.zeros((150, 100, 3)) + i / 5 for i in range(5)]
    imgs1 = [np.ones((100, 150, 3)) - 0.2 - i / 5 for i in range(5)]
    imgs2 = [np.zeros((100, 100, 3)) + i / 5 for i in range(5)]
    imgs3 = [np.ones((150, 150, 3)) - 0.2 - i / 5 for i in range(5)]
    ifig = iplt.show([imgs0, imgs1, imgs2, imgs3], setFrame=False, layout=(2, 2))
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)


# layout test with numpy type imgs.
def test_07():
    imgs0 = [np.zeros((150, 100, 3)) + i / 5 for i in range(5)]
    imgs1 = [np.ones((100, 150, 3)) - 0.2 - i / 5 for i in range(5)]
    imgs2 = [np.zeros((100, 100, 3)) + i / 5 for i in range(5)]
    imgs3 = [np.ones((150, 150, 3)) - 0.2 - i / 5 for i in range(5)]
    ifig = iplt.show([imgs0, imgs1, imgs2, imgs3], setFrame=True, layout=(2, 2))
    assert ifig.save_as_gif("./imgs/test.gif")
    assert ifig.save_as_video("./imgs/test.mp4", fps=2.0)
