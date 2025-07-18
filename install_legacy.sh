#!/bin/bash

# Script to install the OpenNTN package from GitHub

# Run pip install command
pip install git+https://github.com/ant-uni-bremen/OpenNTN@legacy

# Download python file which integrates OpenNTN into Sionna
POST_INSTALL_URL="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/refs/heads/main/post_install_legacy.py"
POST_INSTALL_NAME=$(basename "$POST_INSTALL_URL")
curl -L -o "$POST_INSTALL_NAME" "$POST_INSTALL_URL"

# Integrate OpenNTN into Sionna by adapteing init files
python "$POST_INSTALL_NAME"

# Create link to OpenNTN inside Sionna
OpenNTN_DIR=$(pip show OpenNTN | grep Location | cut -d' ' -f2)/OpenNTN
SIONNA_DIR=$(pip show sionna | grep Location | cut -d' ' -f2)/sionna
ln -s $OpenNTN_DIR $SIONNA_DIR/channel/tr38811

# Remove temporary variables after installation
rm "$POST_INSTALL_NAME"
