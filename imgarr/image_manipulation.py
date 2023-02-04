from typing import List, Union

import numpy as np
from PIL import Image

__all__ = [
    "get_concat_horizontal",
    "get_concat_vertical",
    "align_center",
    "align_horizontal_center",
    "align_vertical_center",
]


def get_concat_horizontal(
    imgs: List[Union[Image.Image, np.ndarray]], alignAuto: bool = True, margin: int = 0
) -> Union[Image.Image, np.ndarray]:
    """
    params:
        imgs: List of image to concat.
        alignAuto: Automatically align each image's vertical center.
        margin: Margin beween each image.
    return:
        dst: Concatenated image.
    """
    imgs_np = [np.array(img) for img in imgs]
    margin_np = np.ones((1, margin, 3))
    if type(imgs[0]) == Image.Image:
        margin_np = (margin_np * 255).astype(np.uint8)
    # Insert margin between each image.
    imgs_np = [margin_np] + [
        imgs_np[i // 2] if i % 2 == 0 else margin_np for i in range(len(imgs_np) * 2)
    ]
    # The height/width of the concatenated iamge.
    H = max([img.shape[0] for img in imgs_np])
    W = sum([img.shape[1] for img in imgs_np])
    dst = np.ones((H, W, 3))
    pivotW = 0
    # Concatenate each image.
    for img in imgs_np:
        if alignAuto:
            dst[:, pivotW : pivotW + img.shape[1]] = align_vertical_center(img, H)
        else:
            dst[: img.shape[0], pivotW : pivotW + img.shape[1]] = img
        pivotW += img.shape[1]
    if type(imgs[0]) == Image.Image:
        return Image.fromarray(dst.astype(np.uint8))
    else:
        return dst


def get_concat_vertical(
    imgs: List[Union[Image.Image, np.ndarray]], alignAuto: bool = True, margin: int = 0
) -> Union[Image.Image, np.ndarray]:
    """
    params:
        imgs: List of image to concat.
        alignAuto: Automatically align each image's vertical center.
        margin: Margin beween each image.
    return:
        dst: Concatenated image.
    """
    imgs_np = [np.array(img) for img in imgs]
    margin_np = np.ones((margin, 1, 3))
    if type(imgs[0]) == Image.Image:
        margin_np = (margin_np * 255).astype(np.uint8)
    # Insert margin between each image.
    imgs_np = [margin_np] + [
        imgs_np[i // 2] if i % 2 == 0 else margin_np for i in range(len(imgs_np) * 2)
    ]
    # The height/width of the concatenated iamge.
    H = sum([img.shape[0] for img in imgs_np])
    W = max([img.shape[1] for img in imgs_np])
    dst = np.ones((H, W, 3))
    pivotH = 0
    # Concatenate each image.
    for img in imgs_np:
        if alignAuto:
            dst[pivotH : pivotH + img.shape[0], :] = align_horizontal_center(img, W)
        else:
            dst[
                pivotH : pivotH + img.shape[0],
                : img.shape[0],
            ] = img
        pivotH += img.shape[0]
    if type(imgs[0]) == Image.Image:
        return Image.fromarray(dst.astype(np.uint8))
    else:
        return dst


def align_vertical_center(
    img: Union[Image.Image, np.ndarray], h: int
) -> Union[Image.Image, np.ndarray]:
    """
    params:
        img: Target image to align.
        h: The hight of the area.
    return:
        dst: Aligned image.
    """
    img_np = np.array(img)
    _h, _w, _ = img_np.shape
    if _h == h:
        return img
    padding_top = (h - _h) // 2
    dst = np.ones((h, _w, 3))
    if img_np.dtype == np.uint8:
        dst = (dst * 255).astype(np.uint8)
        dst[padding_top : padding_top + _h, :_w] = img_np
        return Image.fromarray(dst)
    else:
        dst[padding_top : padding_top + _h, :_w] = img_np
        return dst


def align_horizontal_center(
    img: Union[Image.Image, np.ndarray], w: int
) -> Union[Image.Image, np.ndarray]:
    """
    params:
        img: Target image to align.
        w: The width of the area.
    return:
        dst: Aligned image.
    """
    img_np = np.array(img)
    _h, _w, _ = img_np.shape
    if _w == w:
        return img
    padding_left = (w - _w) // 2
    dst = np.ones((_h, w, 3))
    if img_np.dtype == np.uint8:
        dst = (dst * 255).astype(np.uint8)
        dst[:_h, padding_left : padding_left + _w] = img_np
        return Image.fromarray(dst)
    else:
        dst[:_h, padding_left : padding_left + _w] = img_np
        return dst


def align_center(
    img: Union[Image.Image, np.ndarray], w: int, h: int
) -> Union[Image.Image, np.ndarray]:
    """
    params:
        img: Target image to align.
        w: The width of the area.
        h: The height of the area.
    return:
        dst: Aligned image.
    """
    return align_horizontal_center(align_vertical_center(img, h), w)
