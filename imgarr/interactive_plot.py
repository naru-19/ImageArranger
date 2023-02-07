import warnings
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


def _check(imgs: List[List[Union[Image.Image, np.ndarray]]]) -> None:
    # Check the length of imgs_i.
    for i in range(len(imgs)):
        assert len(imgs[0]) == len(
            imgs[i]
        ), f"imgs[{i}] is a different length from imgs[0]."

    # Check the type of all imgs.
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            assert type(imgs[i][j]) == type(
                imgs[0][0]
            ), f"All image type must be same! Type of imgs[{i}][{j}] and imgs[0][0] is not the same."


def _preprocess(
    imgs: List[List[Union[Image.Image, np.ndarray]]]
) -> List[List[Union[Image.Image, np.ndarray]]]:
    # Preprocess. np.ndarray is handled as float only.
    _imgs = imgs.copy()
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            if type(imgs[i][j]) == np.ndarray and imgs[i][j].dtype == np.uint8:
                _imgs[i][j] = _imgs[i][j] / 255
    return _imgs


def _get_digit_img(
    num: int, frame_length: int, size: int, mode: str = ""
) -> Union[Image.Image, np.ndarray]:
    # Get digital number.
    digit_img = num2img(n=num, fix_digit=len(str(frame_length)), size=(size, size))
    if mode == "pil":
        digit_img=Image.fromarray((digit_img * 255).astype(np.uint8))
    return digit_img


# Show images interactively
def show(
    imgs: List[List[Union[Image.Image, np.ndarray]]],
    layout: Optional[Tuple[int, int]] = None,
    setFrame: bool = False,
) -> InteractiveFigure:
    """
    params:
        imgs: [imgs_1, imgs_2, ..., imgs_n].
        layout: Layout of the images(w,h). 'None' is horizontal.
    return:
        interactive figure
    """

    # Check the iamges and preprocess
    _check(imgs)
    _imgs = _preprocess(imgs)
    frame_length, frame_imgs = len(_imgs[0]), []
    # Image type information.
    mode = "pil" if type(imgs[0][0]) == Image.Image else "np"

    # Get row, col
    if layout:
        col, row = layout
    else:
        col, row = len(_imgs), 1
    assert col * row >= len(
        _imgs
    ), f"row x col ({row}x{col}) is smaller than the length of imgs."
    while col * (row - 1) > len(_imgs):
        row -= 1

    # add dammy
    if mode == "pil":
        dammy = Image.new("RGB", (1, 1))
    else:
        dammy = np.ones((1, 1, 3))
    for i in range(len(_imgs), col * row):
        _imgs.append([dammy for _ in range(len(_imgs[0]))])

    # Width of each column. Used to align each column.
    col_width = [
        max([np.array(_imgs[i + j * col][0]).shape[1] for j in range(row)])
        for i in range(col)
    ]
    # Before concat, align each image horizontal center.
    _imgs = [
        [
            align_horizontal_center(_imgs[i][j], w=col_width[i % col])
            for j in range(frame_length)
        ]
        for i in range(len(_imgs))
    ]
    # Generate each frame img.
    for i in range(frame_length):
        # Treat each frame as a single image.
        frame_parts = [_imgs[j][i] for j in range(len(_imgs))]
        line_imgs = [
            get_concat_horizontal(frame_parts[j * col : (j + 1) * col], margin=20)
            for j in range(row)
        ]
        frame_img = get_concat_vertical(line_imgs, margin=20)
        # Set visualize frame index
        if setFrame:
            digit_size = int(np.array(frame_img).shape[0] * 0.8)
            digit_img = _get_digit_img(
                num=i, frame_length=frame_length, size=digit_size, mode=mode
            )
            frame_img = get_concat_horizontal([frame_img, digit_img], margin=20)
        frame_imgs.append(frame_img)
    ifig = InteractiveFigure(frame_imgs)
    ifig.show()
    return ifig
