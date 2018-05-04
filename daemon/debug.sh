#!/bin/sh

FLASK_APP=daemon python -m flask run -h 0.0.0.0 -p 5000
