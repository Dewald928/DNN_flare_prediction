#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='DNN flare prediction',
    version='0.0.0',
    description='Flare Prediction with lightning',
    author='Dewald Krynauw',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Dewald928/DNN_flare_prediction.git',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

