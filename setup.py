# coding=utf-8
from setuptools import setup, find_packages

setup(
    name="multilingual_title_classifier",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=0.24.2',
        'numpy>=1.16.2',
        'tensorflow-gpu',
        'keras>=2.3.0',
        'scikit-learn',
        'unidecode',
        'symspellpy'
    ],
    author="Martin Baigorria",
    author_email="martinbaigorria@gmail.com",
    description="Solution for MercadoLibre's multilingual title classification challenge",
)
