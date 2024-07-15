from setuptools import setup, find_packages
import os

setup(name='alaya',
      version='0.1',
      packages=find_packages(),
      package_data={'alaya': ['alaya.cpython-38-x86_64-linux-gnu.so']},
      include_package_data=True,
      zip_safe=False,
    #   data_files=[('alaya', ['alaya.cpython-38-x86_64-linux-gnu.so'])])
)