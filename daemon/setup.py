from setuptools import setup, find_packages


def readme():
    with open("README.md", 'r') as f:
        return f.read()


setup(
    name="daemon",
    description="A project",
    version="0.0.1",
    long_description=readme(),
    author="Ryan Avery",
    author_email="ravery@ucsb.edu",
    packages=find_packages(
        exclude=[
        ]
    ),
    include_package_data=True,
    url='https://github.com/rbavery/daemon',
    install_requires=[
        'flask>0',
        'flask_env',
        'flask_restful'
    ],
    tests_require=[
        'pytest'
    ],
    test_suite='tests'
)
