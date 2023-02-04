from imgarr.digital_number import num2img


def test_num2img_01():
    num2img(5)


def test_num2img_02():
    num2img(5, fix_digit=2)


def test_num2img_03():
    num2img(5, fix_digit=2, size=(15, 15))
