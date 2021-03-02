from setuptools import setup, find_packages

setup(
    name="quince",
    version="0.0.0",
    description="Exploring CATE under hidden confounding",
    long_description_content_type="text/markdown",
    url="https://github.com/anon/quince",
    author="anon",
    author_email="anon@anon.anon",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "black",
        "pyreadr",
        "ray>=1.2.0",
        "click>=7.1.2",
        "torch>=1.7.1",
        "numpy>=1.20.1",
        "scipy>=1.6.1",
        "pandas>=1.2.2",
        "jupyter>=1.0.0",
        "seaborn>=0.11.1",
        "matplotlib>=3.3.4",
        "tensorboard>=2.4.1",
        "torchvision>=0.8.2",
        "scikit-learn>=0.24.1",
        "pytorch-ignite>=0.4.3",
    ],
    entry_points={
        "console_scripts": ["quince=quince.application.main:cli"],
    },
)
