# from distutils.core import setup

from setuptools import find_packages
from setuptools import setup, find_packages  # Changed from distutils.core

setup_requires = []
install_requires = [
    'numpy',          # Removed ==1.15
    'pandas',         # Removed ==0.23
    'six',            # Removed ==1.11
    'scikit-learn',   # Removed ==0.19
    'scipy'           # Removed ==1.1
]

setup(name='AMED miRNA project preprocessor',
      version='0.0.1',  # NOQA
      description='',
      author='Kenta Oono',
      author_email='oono@preferred.jp',
      packages=find_packages(),
      setup_requires=setup_requires,
      install_requires=install_requires
      )
