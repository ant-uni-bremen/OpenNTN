#This file is a tutorial on creating a coverage report and generating the SVG image

#Step 1, run    coverage run -m unittest discover -s test -p '*_test.py'   in VSC

#Step 2, run    coverage report -m                                         in VSC

#Step 3,        coverage html                                              in VSC

#Step 4,        coverage-badge -o coverage.svg                             in the adequate env (sionna_debugging)

#Step 5, commit new svg