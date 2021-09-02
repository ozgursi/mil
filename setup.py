from setuptools import setup

setup(
    name='PackageName',
    version='0.1.0',
    author='An Awesome Coder',
    author_email='aac@example.com',
    packages=['gpy'],
    url='http://pypi.python.org/pypi/PackageName/',
    description='An awesome package that does something',
    install_requires=[
        "pandas==1.2.3",
        "numpy==1.20.2",
        "torch==1.7.1",
        "sklearn",
        "scipy==1.6.3",
        "matplotlib"
    ],
)
