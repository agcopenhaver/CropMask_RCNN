# daemon [![v0.0.1](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/rbavery/daemon/releases)

[![Build Status](https://travis-ci.org/rbavery/daemon.svg?branch=master)](https://travis-ci.org/rbavery/daemon) [![Coverage Status](https://coveralls.io/repos/github/rbavery/daemon/badge.svg?branch=master)](https://coveralls.io/github/rbavery/daemon?branch=master) [![Documentation Status](https://readthedocs.org/projects/daemon/badge/?version=latest)](http://daemon.readthedocs.io/en/latest/?badge=latest)

Daemons to query database for field features as geojsons and save queries as png masks.

(This is currently not functional)
See the full documentation at https://mapper.readthedocs.io


# Debug Quickstart
1. Create and start a conda or virtual env with requirements_dev.txt.
2. Set environmental variables appropriately
3. Then
```
export FLASK_DEBUG=1
./debug.sh
```
Code can be updated while in debug mode and will be automatically reloaded.
For a list of URLs that return something, check routes.py for now.

# Docker Quickstart
Inject environmental variables appropriately at either buildtime or runtime
```
# docker build . -t mapper
# docker run -p 5000:80 mapper --name my_mapper
```

# Endpoints
## /
### GET
#### Parameters
* None
#### Returns
* JSON: {"status": "Not broken!"}

# Environmental Variables
* None

# Author
Ryan Avery <ravery@ucsb.edu>
