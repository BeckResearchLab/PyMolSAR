from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: Linux",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description:
description = "Small-Molecule-Design-Toolkit: predict the activity or property of small molecules"
# Long description
long_description = """
Small-Molecule-Design-Toolkit
========
Small-Molecule-Design-Toolkit aims to provide a high quality open-source toolkit
to develop Quantitative Structure-Activity Relationships and Quantitative
Structure-Property Relationships to predict the activity or property of small
molecules. To get started, please go to the repository README_.
.. _README: https://github.com/BeckResearchLab/small-molecule-design-toolkit/blob/master/README.md
License
=======
``Small-Molecule-Design-Toolkit`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2018--, Rahul Avadhoot
"""

NAME = "smdt"
MAINTAINER = "Rahul Avadhoot"
MAINTAINER_EMAIL = "rahul.avadhoot@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/BeckResearchLab/small-molecule-design-toolkit"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Rahul Avadhoot"
AUTHOR_EMAIL = "rahul.avadhoot@gmail.com"
PLATFORMS = "Linux"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'smdt': [pjoin('examples', '*')]}
REQUIRES = ["numpy"]