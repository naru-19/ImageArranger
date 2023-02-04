# import os
# import sys

# print(os.listdir("../"),"1")
# print(os.listdir("./"),"2")
# print(os.listdir("../lib"),"3")

# print(sys.path)
from imgarr.digital_number import num2img


def test_num2img(capsys):
    num2img(5)
    num2img(5,fix_digit=2)
    num2img(5,fix_digit=2,size=(15,15))
