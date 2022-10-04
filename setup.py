from setuptools import setup
import os
import sys
from distutils.version import StrictVersion

# TODO REFACTORING

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    assert StrictVersion(
        "{}.{}".format(sys.version_info[0], sys.version_info[1])
    ) >= StrictVersion("3.8"), "Must use python 3.8 or newer"
    # with open("requirements.txt") as f:
    #     for line in f:
    #         install(line)
    with open("./requirements.txt", "r") as f:
        requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
    setup(
        name="GP_KF",
        version="0.0.1",
        author="Francesco Crocetti & Jeffrey Mao",
        author_email="fc2353@nyu.edu",
        description="A library to easy deploy trajectories using kalman Filters instead Based on Gaussian Processes",
        keywords="kalman Filters, Gaussian Processes, Machine Learning, Drone,",
        url="https://wp.nyu.edu/arpl/",
        packages=['base', 'model', 'tools', 'configs'],
        long_description=read('README.md'),
        python_requires="==3.8",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Topic :: Utilities",
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3.8',
            'Topic :: Research',

        ],
        dependency_links=[
            'https://pypi.org/',

        ],
        install_requires=requirements,

    )
