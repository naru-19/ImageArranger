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
from imgarr.image_manipulation import (
    align_center,
    align_horizontal_center,
    get_concat_horizontal,
    get_concat_vertical,
    save_as_gif,
    save_as_video,
)

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

def resize_max(img,max_size):
    if isinstance(img,Image.Image):
        w,h = img.size
        if w>h:
            nw = max_size
            nh = int(max_size*h/w)
        else:
            nh = max_size
            nw = int(max_size*w/h)
        img = img.resize((nw,nh))
        return img

    h,w = img.shape[:2]
    if h > w:
        img = cv2.resize(img,(max_size,int(max_size*h/w)))
    else:
        img = cv2.resize(img,(int(max_size*w/h),max_size))
    return img

def _preprocess(
    imgs: List[List[Union[Image.Image, np.ndarray]]], mode: str, row: int, col: int,max_size:int=256
) -> List[List[Union[Image.Image, np.ndarray]]]:

    frame_length = len(imgs[0])
    # Preprocess. np.ndarray is handled as float only.
    _imgs = imgs.copy()
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            if type(imgs[i][j]) == np.ndarray and imgs[i][j].dtype == np.uint8:
                _imgs[i][j] = _imgs[i][j] / 255

    # add dammy
    if mode == "pil":
        dammy = Image.new("RGB", (1, 1))
    else:
        dammy = np.ones((1, 1, 3))
    for i in range(len(_imgs), col * row):
        _imgs.append([dammy for _ in range(frame_length)])

    _imgs = [
        [
            resize_max(img,max_size)
            for img in imgs_col
        ]
        for imgs_col in _imgs
    ]

    # Width of each column. Used to align each column.
    max_width = (
        [
            max([
                img.shape[1] if type(img) == np.ndarray else img.size[0]
                for img in imgs_col
            ])
            for imgs_col in _imgs
        ]
    )
    max_height = (
        [
            max([
                img.shape[0] if type(img) == np.ndarray else img.size[1]
                for img in imgs_col
            ])
            for imgs_col in _imgs
        ] 
    )



    # Align each image horizontal center.
    _imgs = [
        [
            align_center(_imgs[i][j], w=max_width[i], h=max_height[i])
            for j in range(frame_length)
        ]
        for i in range(len(_imgs))
    ]
    return _imgs


def _get_digit_img(
    num: int, frame_length: int, size: int, mode: str = ""
) -> Union[Image.Image, np.ndarray]:
    # Get digital number.
    digit_img = num2img(n=num, fix_digit=len(str(frame_length)), size=(size, size))
    if mode == "pil":
        digit_img = Image.fromarray((digit_img * 255).astype(np.uint8))
    return digit_img


# Show images interactively
def show(
    imgs: List[List[Union[Image.Image, np.ndarray]]],
    layout: Optional[Tuple[int, int]] = None,
    setFrame: bool = False,
    max_size:int = 512
) -> InteractiveFigure:
    """
    params:
        imgs: [imgs_1, imgs_2, ..., imgs_n].
        layout: Layout of the images(w,h). 'None' is horizontal.
    return:
        interactive figure
    """

    # Get row, col
    if layout is None:
        layout = (len(imgs), 1)
    col, row = layout
    assert col * row >= len(
        imgs
    ), f"row x col ({row}x{col}) is smaller than the length of imgs."
    while col * (row - 1) > len(imgs):
        row -= 1

    # Get image type
    mode = "pil" if type(imgs[0][0]) == Image.Image else "np"

    # Check the iamges and preprocess
    _check(imgs)
    _imgs = _preprocess(imgs=imgs, mode=mode, row=row, col=col,max_size=max_size)
    frame_length, frame_imgs = len(_imgs[0]), []

    # Generate each frame img.
    for i in range(frame_length):
        # Treat each frame as a single image.
        # Collect images at frame i.
        frame_parts = [_imgs[j][i] for j in range(len(_imgs))]
        line_imgs = [
            get_concat_horizontal(frame_parts[j * col : (j + 1) * col], margin=20)
            for j in range(row)
        ]
        frame_img = get_concat_vertical(line_imgs, margin=20)
        # Visualize frame index
        if setFrame:
            digit_size = int(np.array(frame_img).shape[0] * 0.8)*0.5
            digit_img = _get_digit_img(
                num=i, frame_length=frame_length, size=digit_size, mode=mode
            )
            frame_img = get_concat_horizontal([frame_img, digit_img], margin=20)
        frame_imgs.append(frame_img)
    ifig = InteractiveFigure(frame_imgs)
    ifig.show()
    return ifig
