# PLEASE NOTE!

This is OpenNTN 0.19.0, the version of OpenNTN for Sionna 0.19.2, which is built on TensorFlow. It is frozen. OpenNTN 1.0.0, for Sionna 1.x, is installed from the tag v1.0.0 of this git, and OpenNTN 2, for Sionna 2.x and PyTorch, with <code>pip install openntn</code>. You can still find the [documentation of the legacy version of Sionna here](https://jhoydis.github.io/sionna-0.19.2-doc/), including an [installation guide here](https://jhoydis.github.io/sionna-0.19.2-doc/installation.html). <br>

# OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations
This git provides an implementation of the channel models for dense urban, urban, and suburban scenarios according to the 3GPP TR38.811 standard. It is an extension to the existing Sionna™ framework and integrates into it as another module.

## Installation

As of very recently, Sionna has released version 1.0. This installation uses the legacy version 0.19.<br>

1. Install Sionna <br>
  <code>pip install sionna==0.19.2</code> <br>
For more information on the different installation options we refer the reader to the [sionna documentation](https://nvlabs.github.io/sionna/installation.html).
2. Download the install.sh file of the tag v0.19.0 of this git <br>
   <code>curl -fLO https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/v0.19.0/install.sh</code>
3. Execute the install.sh file, which installs OpenNTN 0.19.0 from that tag <br>
   <code>. install.sh</code>

## Contents of OpenNTN
OpenNTN imlements the models for Non-Terrestrial Networks in the dense urban, urban, and suburban scenarios as defined in the standard 3GPP TR38.811. These are similar to the existing models defined in 3GPP TR38.901, which are already implemented in Sionna:tm:. For an optimal integration, the user interface of 38811 is kept as similar as possible to the user interface of 38901, with the addition of a few parameters, such as the satellite height, user elevation angle, and new antenna radiation patterns. For a practical demonstration, we refer the reader to the notebooks found in the examples section.

## Citing OpenNTN
When you use OpenNTN for research, please cite us as: "An Open Source Channel Emulator for Non-Terrestrial Networks,T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben, A. Dekorsy​, Advanced Satellite Multimedia Systems Conference/Signal Processing for Space Communications Workshop (ASMS/SPSC 2025), Sitges, Spanien, 26. - 28. Februar 2025",\
or by using the BibTeX:\
@inproceedings{OpenNTN,\
  author = {T. D\"{u}e and M. Vakilifard and C. Bockelmann and D. W\"{u}bben and A. Dekorsy​},\
  year = {2025},\
  month = {Feb},\
  title = {An Open Source Channel Emulator for Non-Terrestrial Networks},\
  URL = {https://www.ant.uni-bremen.de/sixcms/media.php/102/15080/An%20Open%20Source%20Channel%20Emulator%20for%20Non-Terrestrial%20Networks.pdf}, \
  address={Sitges, Spain},\
  abstract={Non-Terrestrial Networks (NTNs) are one of the key technologies to achieve the goal of ubiquitous connectivity in 6G. However, as real world data in NTNs is expensive, there is a need for accurate simulations with appropriate channel models that can be used for the development and testing communication technologies for various NTN scenarios. In this work, we present our implementation of multiple channel models for NTNs provided by the 3rd Generation Partnership Project (3GPP) in an open source framework. The framework can be integrated into the existing Python framework Sionna™ , enabling the investigations of NTNs using link-level simulations. By keeping the framework open source, we allow users to adapt it for specific use cases without needing to implement the complex underlying mathematical framework. The framework is implemented in Python as an extension to the existing Sionna™ framework, which already provides a large number of existing 5G-compliant communications components. As the models in the framework are based on Tensorflow and Keras, they are compatible with not only Sionna™ , but also many existing software solutions implemented in Tensorflow and Keras, including a significant amount of the Machine Learning (ML) related research.},\
  booktitle={Advanced Satellite Multimedia Systems Conference/Signal Processing for Space Communications Workshop (ASMS/SPSC 2025)}\
}

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

Note: This project is intended to be used with NVIDIA’s Sionna™ framework, which is licensed under the Apache License, Version 2.0 (the "License"). Users must comply with the terms of that license when using this OpenNTN in addition with Sionna™.
