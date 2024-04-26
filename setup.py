from setuptools import setup, find_packages

setup(
    name='skyline_extraction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'transformers',
        'networkx',
        'scikit-image',
        'opencv-python',
    ],
)