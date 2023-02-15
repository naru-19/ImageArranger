from pathlib import Path
from typing import List, Union

import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    # Preprocess. Convert each image to np.ndarray.
    imgs_np = [np.array(img) for img in imgs]
    # The margin between each image.
    margin_np = np.ones((1, margin, 3))
    if type(imgs[0]) == Image.Image:
        margin_np = (margin_np * 255).astype(np.uint8)
    # Insert margin.
    imgs_np = [margin_np] + [
        imgs_np[i // 2] if i % 2 == 0 else margin_np for i in range(len(imgs_np) * 2)
    ]
    # The height/width of the concatenated iamge.
    H = max([img.shape[0] for img in imgs_np])
    W = sum([img.shape[1] for img in imgs_np])
    dst = np.ones((H, W, 3))
    pivotW = 0
    # Fill pixels by each image.
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


# TODO implement insert margin ↓↑


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
    # Preprocess. Convert each image to np.ndarray.
    imgs_np = [np.array(img) for img in imgs]
    # The margin between each image.
    margin_np = np.ones((margin, 1, 3))
    if type(imgs[0]) == Image.Image:
        margin_np = (margin_np * 255).astype(np.uint8)
    # Insert margin.
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
    elif _h > h:
        raise ValueError(
            f"Target image height {_h} is larger than the draw area height {h}."
        )
    padding_top = (h - _h) // 2
    dst = np.ones((h, _w, 3))
    if img_np.dtype == np.uint8:
        dst = (dst * 255).astype(np.uint8)
        dst[padding_top : padding_top + _h, :_w] = img_np
        if type(img) == Image.Image:
            return Image.fromarray(dst)
        else:
            return dst
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
    elif _w > w:
        raise ValueError(
            f"Target image width {_w} is larger than the draw area width {w}."
        )
    padding_left = (w - _w) // 2
    dst = np.ones((_h, w, 3))
    if img_np.dtype == np.uint8:
        dst = (dst * 255).astype(np.uint8)
        dst[:_h, padding_left : padding_left + _w] = img_np
        if type(img) == Image.Image:
            return Image.fromarray(dst)
        else:
            return dst
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


def save_as_gif(
    imgs: List[Union[np.ndarray, Image.Image]],
    save_path: Union[str, Path],
    loop: int = 0,
) -> bool:
    save_path = Path(save_path)
    assert save_path.suffix in {".gif", ".GIF"}, f"Unkown file type {save_path.suffix}"
    # Convert np.ndarry to PIL.
    if type(imgs[0]) == np.ndarray:
        if imgs[0].dtype == np.uint8:
            imgs = [Image.fromarray((img).astype(np.uint8)) for img in imgs]
        else:
            imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], loop=loop)
    print(f"GIF saved {save_path}")
    return True


def _resize_for_video(
    imgs: List[Union[Image.Image, np.ndarray]]
) -> List[Union[Image.Image, np.ndarray]]:
    _imgs = []
    # imageio can't save h x w video which h%2=1 or w%2=1
    if type(imgs[0]) == Image.Image:
        for img in imgs:
            w, h = img.size
            _img = Image.new("RGB", (w + w % 2, h + h % 2))
            _img.paste(img, (0, 0))
            _imgs.append(_img)
    else:
        for img in imgs:
            h, w, _ = img.shape
            _img = np.zeros((h + h % 2, w + w % 2, 3), dtype=img.dtype)
            _img[:h, :w, :] = img
            _imgs.append(_img)
    return _imgs


def save_as_video(
    imgs: List[Union[Image.Image, np.ndarray]],
    save_path: Union[str, Path],
    fps: float = 2.0,
) -> bool:
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    if type(imgs[0]) == Image.Image:
        return _pil_imgs_to_video(
            imgs=_resize_for_video(imgs), save_path=save_path, fps=fps
        )
    else:
        if imgs[0].dtype == np.uint8:
            return _pil_imgs_to_video(
                imgs=_resize_for_video(imgs), save_path=save_path, fps=fps
            )
        else:
            return _np_imgs_to_video(
                imgs=_resize_for_video(imgs), save_path=save_path, fps=fps
            )


def _pil_imgs_to_video(
    imgs: List[Image.Image], save_path: Union[str, Path], fps: float
) -> bool:
    imageio.mimsave(
        str(save_path), [np.array(img) for img in imgs], fps=fps, macro_block_size=1
    )
    print(f"Video saved {save_path}")
    return True


def _np_imgs_to_video(
    imgs: List[Image.Image], save_path: Union[str, Path], fps: float
) -> bool:
    imgs_pil = [
        img if img.dtype == np.uint8 else (img * 255).astype(np.uint8) for img in imgs
    ]
    return _pil_imgs_to_video(imgs=imgs_pil, save_path=save_path, fps=fps)
