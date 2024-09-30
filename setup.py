from setuptools import setup, find_packages

setup(
    name='TOF_ML',
    version='0.0.1',
    description='Package for generating simion data '
                'and training a machine learning model',
    license='MIT',
    packages=find_packages(where='src'),  # 'src' is your main package directory
    package_dir={'': 'src'},  # Source code is inside the src directory
    author='Amanda Shackelford',
    author_email='ajshack4@gmail.com',
    keywords=['example'],
    url='https://github.com/amandashack/TOF_ML'
)