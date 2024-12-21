from typing import Optional, Tuple
from PIL import Image
import numpy as np

___all__ = ["num2img"]


def num2img(
    n: int,
    fix_digit: int = -1,
    size: Optional[Tuple] = None,
) -> np.ndarray:
    """
    params:
        n: Target number to rendering.
        fix_digit: Fix the number of display digit.
        size: size (w x h)
    return:
        digit: The digital number of n
    """
    # Number of display digits
    if fix_digit != -1:
        number_of_digits = fix_digit
    else:
        number_of_digits = 1 if n == 0 else int(np.log10(n)) + 1

    # margin around each number.
    digit = np.zeros((7, number_of_digits * 4 + 1, 3))
    n_s = str(n)
    n_s = n_s.zfill(number_of_digits)
    for i, num in enumerate(n_s):
        # margin top/bottom 1px around of each number
        # digit[:,4*i,:] is margin
        digit[1:6, 4 * i + 1 : 4 * (i + 1)] = _get_digit(int(num))
    if size:
        digit = np.array(Image.fromarray((digit * 255).astype(np.uint8)).resize(size))
    return digit


def _get_digit(x):
    block = np.zeros((5, 3, 3))
    lines = {
        0: list(range(0, 3)),
        1: list(range(6, 9)),
        2: list(range(12, 15)),
        3: list(range(0, 7, 3)),
        4: list(range(2, 9, 3)),
        5: list(range(6, 13, 3)),
        6: list(range(8, 15, 3)),
    }

    # Line numbers that make up the digital number
    digit_num = {
        0: [0, 2, 3, 4, 5, 6],
        1: [4, 6],
        2: [0, 1, 2, 4, 5],
        3: [0, 1, 2, 4, 6],
        4: [1, 3, 4, 6],
        5: [0, 1, 2, 3, 6],
        6: [1, 2, 3, 5, 6],
        7: [0, 4, 6],
        8: list(range(7)),
        9: [0, 1, 3, 4, 6],
    }
    for line_id in digit_num[x]:
        for px in lines[line_id]:
            block[px // 3, px % 3, :] = 1
    return block


num2img(123, fix_digit=3, size=(100, 100))
