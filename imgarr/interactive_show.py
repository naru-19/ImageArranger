from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (
    FloatSlider,
    HBox,
    IntSlider,
    Select,
    interact,
    interactive,
    interactive_output,
    jslink,
)
from PIL import Image
from tqdm import tqdm

import imgarr.digital_number as digitn
from imgarr.image_manipulation import get_concat_horizontal, get_concat_vertical

___all__ = ["InteractiveFigure", "ishow"]


class InteractiveFigure:
    def __init__(self, frame_imgs: List[Union[Image.Image, np.ndarray]]) -> None:
        self.imgs = frame_imgs

    def show(self) -> None:
        frame_length = len(self.imgs)
        slider = IntSlider(min=0, max=frame_length - 1)
        ui = HBox([slider])
        if type(self.imgs[0]) == np.ndarray:
            out = interactive_output(self._showNP, {"t": slider})
        elif type(self.imgs[0]) == Image.Image:
            out = interactive_output(self._showPIL, {"t": slider})
        else:
            raise NotImplementedError(
                "The images type should be np.ndarray or Image.Image"
            )
        try:
            display(ui, out)  # type: ignore
        except:
            print("Can't show the figure.")

    def _showNP(self, t: int) -> None:
        plt.imshow(self.imgs[t])
        plt.axis("off")

    def _showPIL(self, t: int) -> None:
        display(self.imgs[t])  # type: ignore


# Show images interactively
def ishow(
    imgs: List[List[Union[Image.Image, np.ndarray]]],
    layout: Optional[Tuple[int, int]] = None,
    setFrame: bool = False,
) -> InteractiveFigure:
    """
    params:
        imgs: [imgs_1, imgs_2, ..., imgs_n]
        layout: How to arrange the images(w,h). 'None' is horizontal.
    return:

    """
    for i in range(len(imgs)):
        if len(imgs[i]) != len(imgs[0]):
            raise ValueError(f"imgs[{i}] is a different length from imgs[0].")
    assert type(imgs[0][0]) == Image.Image or type(imgs[0][0]) == np.ndarray
    mode = "pil" if type(imgs[0][0]) == Image.Image else "np"
    frame_length = len(imgs[0])
    frame_imgs = []
    # Set visualize frame index
    if setFrame:
        if mode == "pil":
            size = min(imgs[0][0].size)
        else:
            size = min(imgs[0][0].shape[:2])
        digit_imgs = [
            digitn.num2img(n=i, fix_digit=len(str(frame_length)), size=(size, size))
            for i in range(frame_length + 1)
        ]
        if mode == "pil":
            digit_imgs = [
                Image.fromarray((digit_img * 255).astype(np.uint8))
                for digit_img in digit_imgs
            ]
        imgs.append(digit_imgs)

    # Arrange in row x col
    if layout:
        col, row = layout
        while col * (row - 1) > len(imgs):
            row -= 1
        for i in range(frame_length):
            frame_imgs.append(
                get_concat_vertical(
                    [
                        get_concat_horizontal(
                            [
                                img[i]
                                for img in imgs[j * col : min(col * (j + 1), len(imgs))]
                            ],
                            margin=20,
                        )
                        for j in range(row)
                    ],
                    margin=20,
                )
            )
    else:
        for i in range(frame_length):
            frame_imgs.append(
                get_concat_horizontal([img[i] for img in imgs], margin=20)
            )
    ifig = InteractiveFigure(frame_imgs)
    ifig.show()
    return ifig
