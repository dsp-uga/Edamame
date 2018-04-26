#!/usr/bin/env python3
from setuptools import setup
required = open('requirements.txt').read().split('\n')
setup(
  packages = ['ARIMA','LSTM'],
  install_requires = required
)
