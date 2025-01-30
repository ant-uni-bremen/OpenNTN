#!/bin/bash

# Script to install package from GitHub

# Run pip install command
pip install git+https://github.com/ant-uni-bremen/OpenNTN

OpenNTN_DIR=$(pip show OpenNTN | grep Location | cut -d' ' -f2)/OpenNTN
python "$OpenNTN_DIR/OpenNTN/post_install.py"

SIONNA_DIR=$(python -c "import sionna; import os; print(os.path.dirname(sionna.__file__))")

ln -s $OpenNTN_DIR $SIONNA_DIR/channel

mv $OpenNTN_DIR/OpenNTN $OpenNTN_DIR/38811
