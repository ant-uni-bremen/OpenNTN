# OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations
This git provides an implementation of the channel models for dense urban, urban, and suburban scenarios according to the 3GPP TR38.811 standard. It is an extension to the existing Sionna™ framework and integrates into it as another module.

## Installation

1. Install Sionna
  <code>pip install sionna</code>
For more information on the different installation options we refer the reader to the [sionna documentation](https://nvlabs.github.io/sionna/installation.html).
2. Download the install.sh file
3. execute the install.sh file
   <code>. install.sh</code>

## Contents of OpenNTN
OpenNTN imlements the models for Non-Terrestrial Networks in the dense urban, urban, and suburban scenarios as defined in the standard 3GPP TR38.811. These are similar to the existing models defined in 3GPP TR38.901, which are already implemented in Sionna:tm:. For an optimal integration, the user interface of 38811 is kept as similar as possible to the user interface of 38901, with the addition of a few parameters, such as the satellite height, user elevation angle, and new antenna radiation patterns. For a practical demonstration, we refer the reader to the notebooks found in the examples section.
