from setuptools import setup, find_packages

setup(
    name="OpenNTN",  # Change this to your package name
    version="0.1.0",
    author="Tim Due",
    author_email="duee@ant.uni-bremen.de",
    description="An extension of the Sionna framework including channel models for NTN models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ant-uni-bremen/OpenNTN",
    packages=find_packages(),
    install_requires=[
        "sionna"  # Add dependencies
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    
    include_package_data=True,
)


    