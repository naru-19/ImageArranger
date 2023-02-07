from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import HBox, IntSlider, interactive_output
from PIL import Image

from imgarr.digital_number import num2img
from imgarr.image_manipulation import (align_horizontal_center,
                                       get_concat_horizontal,
                                       get_concat_vertical, save_as_gif,
                                       save_as_video)

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

    def save_as_gif(self, save_path: Union[str, Path], loop: int = 0):
        return save_as_gif(self.imgs, save_path, loop=loop)

    def save_as_video(self, save_path: Union[str, Path], fps: float = 2.0):
        return save_as_video(self.imgs, save_path, fps)


# Show images interactively
def show(
    imgs: List[List[Union[Image.Image, np.ndarray]]],
    layout: Optional[Tuple[int, int]] = None,
    setFrame: bool = False,
) -> InteractiveFigure:
    """
    params:
        imgs: [imgs_1, imgs_2, ..., imgs_n]
        layout: Layout of the images(w,h). 'None' is horizontal.
    return:
        interactive figure
    """
    # Preprocess. np.ndarray is handled as float only in this function.
    for i in range(len(imgs)):
        if len(imgs[i]) != len(imgs[0]):
            raise ValueError(f"imgs[{i}] is a different length from imgs[0].")
        assert type(imgs[i][0]) == Image.Image or type(imgs[i][0]) == np.ndarray    
        if type(imgs[i][0]) == np.ndarray and imgs[i][0].dtype == np.uint8:
            for j in range(len(imgs[i])):
                imgs[i][j] = imgs[i][j] / 255
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
            num2img(n=i, fix_digit=len(str(frame_length)), size=(size, size))
            for i in range(frame_length + 1)
        ]
        if mode == "pil":
            digit_imgs = [
                Image.fromarray((digit_img * 255).astype(np.uint8))
                for digit_img in digit_imgs
            ]
        imgs.append(digit_imgs)

    # Get row, col
    if layout:
        col, row = layout
    else:
        col, row = len(imgs), 1
    while col * (row - 1) > len(imgs):
        row -= 1

    # add dammy
    if mode == "pil":
        dammy = Image.new("RGB", (1, 1))
    else:
        dammy = np.ones((1, 1, 3))
    for i in range(len(imgs), col * row):
        imgs.append([dammy for _ in range(len(imgs[0]))])

    # Width of each column. Used to align each column.
    col_width = [
        max([np.array(imgs[i + j * col][0]).shape[1] for j in range(row)])
        for i in range(col)
    ]
    for i in range(frame_length):
        # Treat each frame as a single image.
        frame_imgs.append(
            get_concat_vertical(
                [
                    get_concat_horizontal(
                        [
                            align_horizontal_center(
                                imgs[k][i], w=col_width[k - j * col]
                            )
                            for k in range(j * col, col * (j + 1))
                        ],
                        margin=20,
                    )
                    for j in range(row)
                ],
                margin=20,
            )
        )
    ifig = InteractiveFigure(frame_imgs)
    ifig.show()
    return ifig
