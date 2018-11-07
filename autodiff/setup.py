from distutils.core import setup

setup(
    name='AutoDiff',
    version='0.1.0',
    author='B Groever, M. Fekete, P. Wong, P. Sukhum',
    author_email='groever@seas.harvard.edu',
    packages=['towelstuff', 'towelstuff.test'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',
    description='Automatic differentiaion package.',
    long_description=open('README.txt').read(),
    #install_requires=[
    #    "math >= 1.1.1",
    #    "numpy == 0.1.4",
    #],
)