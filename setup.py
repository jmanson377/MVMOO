from setuptools import setup, find_packages

setup(
    name='MVMOO',
    version='0.1.1',
    packages=find_packages(exclude=['tests*']),
    license='Apache',
    description='Mixed Variable Multi-Objective Optimization',
    long_description=open('README.md').read(),
    install_requires=['gpflow>=2.0.5',
'matplotlib>=3.2.2',
'numpy>=1.18.5',
'pyDOE2>=1.3.0',
'scipy>=1.4.1',
'sobol-seq>=0.2.0',
'tensorflow>=2.3.0',
'tensorflow-probability>=0.11.0'],
    url='https://github.com/jmanson377/MVMOO',
    author='Jamie Manson',
    author_email='jamie.manson377@gmail.com'
)