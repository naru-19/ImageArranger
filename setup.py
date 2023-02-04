import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_install_requires():
    install_requires = [
        "tqdm",
        "numpy",
        "matplotlib",
        "pillow",
        "opencv-python"
    ]
    return install_requires


setuptools.setup(
    name="narutils",
    version="1.0.0",
    author="naru-19",
    author_email="narururu19@gmail.com",
    description="Complete",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naru-19/ImageArranger",
    install_requires=get_install_requires(),
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
