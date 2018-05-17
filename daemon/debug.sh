#!/bin/sh
export FLASK_DEBUG=1
FLASK_APP=daemon python -m flask run -h 0.0.0.0 -p 5000
