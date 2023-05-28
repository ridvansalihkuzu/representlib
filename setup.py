from setuptools import find_packages, setup

setup(
    name='represent',
    packages=find_packages(include=['represent']),
    version='0.0.1',
    description='Common code for the ESA RepreSent project',
    author='Lloyd Hughes, Marc Russwurm',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner', 'torch', 'torchgeo', 'torchtyping', 'wget', 'rasterio', 'torchvision'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
