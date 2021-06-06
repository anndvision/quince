from setuptools import setup, find_packages

setup(
    name="quince",
    version="0.0.0",
    description="Exploring CATE under hidden confounding",
    long_description_content_type="text/markdown",
    url="https://github.com/anndvision/quince",
    author="Andrew Jesson",
    author_email="andrew.jesson@cs.ox.ac.uk",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.1",
        "torch>=1.8.1",
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "pandas>=1.2.4",
        "pyreadr>=0.4.1",
        "seaborn>=0.11.1",
        "hyperopt>=0.2.5",
        "ray[tune]>=1.3.0",
        "matplotlib>=3.4.2",
        "tensorboard>=2.5.0",
        "torchvision>=0.9.1",
        "scikit-learn>=0.24.2",
        "pytorch-ignite>=0.4.4",
    ],
    entry_points={
        "console_scripts": ["quince=quince.application.main:cli"],
    },
)
