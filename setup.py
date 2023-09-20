from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning==2.0.9",
        "blobfile==2.0.2",
        "numpy==1.26.0",
        "torch==2.0.1",
        "tqdm==4.66.1",
        "scipy==1.11.2",
        "mod==0.3.0",
        "matplotlib==3.8.0",
    ],
)
