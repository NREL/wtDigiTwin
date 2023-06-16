from setuptools import setup, find_packages

VERSION='1.0'

setup(
    name='wtDigiTwin',
    version=VERSION,
    description='wtDigiTwin a digital twin model based on YAMS',
    url='https://github.com/NREL/wtDigiTwin',
    author='Emmanuel Branlard',
    author_email='lastname@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False
)
