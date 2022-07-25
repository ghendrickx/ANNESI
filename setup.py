"""
ANNESI: An open-source artificial neural network for estuarine salt intrusion.

Author: Gijs G. Hendrickx
"""
from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='ANNESI',
    version='0.1',
    author='Gijs G. Hendrickx',
    author_email='G.G.Hendrickx@tudelft.nl',
    description='An open-source artificial neural network for estuarine salt intrusion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=[
        'application', 'src', 'utils',
    ],
    license='Apache-2.0',
    keywords=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit_learn',
        'joblib',
        'dash',
        'plotly',
        'Shapely',
    ],
    python_requires='>=3.7'
)
