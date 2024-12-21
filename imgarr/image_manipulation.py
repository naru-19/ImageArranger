from pathlib import Path
from typing import List, Union
import imageio
import numpy as np
from PIL import Image

__all__ = [
    "get_concat_horizontal",
    "get_concat_vertical",
    "align_center",
    "align_horizontal_center",
    "align_vertical_center",
]


# np.uint8でデータを扱うための前処理
class ImagePrerocessor:
    preprocess_funcs: List[callable] = ["to_numpy_uint_255", "to_three_dimensional"]

    @classmethod
    def execute(cls, img: Image.Image) -> np.ndarray:
        for func_name in cls.preprocess_funcs:
            img = getattr(cls, func_name)(img)
        return img

    @classmethod
    def to_numpy_uint_255(cls, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.array(img)
        else:
            if img.dtype == np.uint8:
                return img
            else:
                return (img * 255).astype(np.uint8)

    @classmethod
    def to_three_dimensional(cls, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img[:, :, np.newaxis]
        else:
            return img


# np.uint8から、元のデータ型に戻すための後処理
class ImagePostprocessor:
    postprocess_funcs: List[callable] = ["to_input_type_image"]

    @classmethod
    def execute(
        cls, input_img: Union[np.ndarray, Image.Image], output_img: np.ndarray
    ) -> Union[np.ndarray, Image.Image]:
        for func_name in cls.postprocess_funcs:
            img = getattr(cls, func_name)(input_img, output_img)
        return img

    @classmethod
    def to_input_type_image(
        cls, input_img: Union[np.ndarray, Image.Image], output_img: np.ndarray
    ) -> Union[np.ndarray, Image.Image]:
        if isinstance(input_img, Image.Image):
            return Image.fromarray(ImagePrerocessor.to_numpy_uint_255(output_img))
        else:
            if input_img.dtype == np.uint8:
                return output_img
            else:
                return output_img / 255


def align_horizontal_center(
    img: Union[np.ndarray, Image.Image], w: int, bg_color: Tuple[int] = (255, 255, 255, 0)
) -> Union[np.ndarray, Image.Image]:
    """
    params:
        img: Target image to align.
        w: The width of the area.
    return:
        dst: Aligned image.
    """
    img_np = ImagePrerocessor.execute(img)
    ch, cw, c = img_np.shape
    bg_color = np.array(bg_color, dtype=np.uint8)[:c]
    if cw == w:
        return img
    elif cw > w:
        raise ValueError(f"Target image width {cw} is larger than the draw area width {w}.")
    padding_left = (w - cw) // 2
    dst = np.ones((ch, w, c), dtype=np.uint8) * bg_color
    dst[:, padding_left : padding_left + cw] = img_np
    return ImagePostprocessor.execute(img, dst)


def align_vertical_center(
    img: Union[np.ndarray, Image.Image], h: int, bg_color: Tuple[int] = (255, 255, 255, 0)
) -> Union[np.ndarray, Image.Image]:
    """
    params:
        img: Target image to align.
        h: The height of the area.
    return:
        dst: Aligned image.
    """
    img_np = ImagePrerocessor.execute(img)
    ch, cw, c = img_np.shape
    bg_color = np.array(bg_color, dtype=np.uint8)[:c]
    if ch == h:
        return img
    elif ch > h:
        raise ValueError(f"Target image height {ch} is larger than the draw area height {h}.")
    padding_top = (h - ch) // 2
    dst = np.ones((h, cw, c), dtype=np.uint8) * bg_color
    dst[padding_top : padding_top + ch] = img_np
    return ImagePostprocessor.execute(img, dst)


def align_center(
    img: Union[np.ndarray, Image.Image], w: int, h: int, bg_color: Tuple[int] = (255, 255, 255, 0)
) -> Union[np.ndarray, Image.Image]:
    """
    params:
        img: Target image to align.
        w: The width of the area.
        h: The height of the area.
    return:
        dst: Aligned image.
    """
    return align_horizontal_center(align_vertical_center(img, h, bg_color), w, bg_color)


def get_concat_horizontal(
    imgs: List[Union[np.ndarray, Image.Image]], margin: int = 0, margin_color: Tuple[int] = (255, 255, 255, 0)
):
    """
    params:
        imgs: List of images to concatenate.
        margin: Margin between images.
        bg_color: Background color.
    """
    imgs_np = [ImagePrerocessor.execute(img) for img in imgs]
    c = imgs_np[0].shape[2]
    margin_np = np.ones((1, margin, c), dtype=np.uint8) * np.array(margin_color, dtype=np.uint8)[:c]
    imgs_np = [imgs_np[i // 2] if i % 2 == 0 else margin_np for i in range(len(imgs_np) * 2 - 1)]
    H = max([img.shape[0] for img in imgs_np])
    W = sum([img.shape[1] for img in imgs_np])
    dst = np.ones((H, W, c), dtype=np.uint8) * np.array(margin_color, dtype=np.uint8)[:c]
    pivotW = 0
    for img in imgs_np:
        dst[:, pivotW : pivotW + img.shape[1]] = align_vertical_center(img, H, margin_color)
        pivotW += img.shape[1]
    return ImagePostprocessor.execute(imgs[0], dst)


def get_concat_vertical(
    imgs: List[Union[np.ndarray, Image.Image]], margin: int = 0, margin_color: Tuple[int] = (255, 255, 255, 0)
):
    """
    params:
        imgs: List of images to concatenate.
        margin: Margin between images.
        bg_color: Background color.
    """
    imgs_np = [ImagePrerocessor.execute(img) for img in imgs]
    c = imgs_np[0].shape[2]
    margin_np = np.ones((margin, 1, c), dtype=np.uint8) * np.array(margin_color, dtype=np.uint8)[:c]
    imgs_np = [imgs_np[i // 2] if i % 2 == 0 else margin_np for i in range(len(imgs_np) * 2 - 1)]
    H = sum([img.shape[0] for img in imgs_np])
    W = max([img.shape[1] for img in imgs_np])
    dst = np.ones((H, W, c), dtype=np.uint8) * np.array(margin_color, dtype=np.uint8)[:c]
    pivotH = 0
    for img in imgs_np:
        dst[pivotH : pivotH + img.shape[0]] = align_horizontal_center(img, W, margin_color)
        pivotH += img.shape[0]
    return ImagePostprocessor.execute(imgs[0], dst)


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


def _resize_for_video(imgs: List[Union[Image.Image, np.ndarray]]) -> List[Union[Image.Image, np.ndarray]]:
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
        return _pil_imgs_to_video(imgs=_resize_for_video(imgs), save_path=save_path, fps=fps)
    else:
        if imgs[0].dtype == np.uint8:
            return _pil_imgs_to_video(imgs=_resize_for_video(imgs), save_path=save_path, fps=fps)
        else:
            return _np_imgs_to_video(imgs=_resize_for_video(imgs), save_path=save_path, fps=fps)


def _pil_imgs_to_video(imgs: List[Image.Image], save_path: Union[str, Path], fps: float) -> bool:
    imageio.mimsave(str(save_path), [np.array(img) for img in imgs], fps=fps, macro_block_size=1)
    print(f"Video saved {save_path}")
    return True


def _np_imgs_to_video(imgs: List[Image.Image], save_path: Union[str, Path], fps: float) -> bool:
    imgs_pil = [img if img.dtype == np.uint8 else (img * 255).astype(np.uint8) for img in imgs]
    return _pil_imgs_to_video(imgs=imgs_pil, save_path=save_path, fps=fps)
