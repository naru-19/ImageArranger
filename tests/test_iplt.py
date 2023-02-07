import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import imgarr.interactive_plot as iplt


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