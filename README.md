# Getting Started

Welcome to OpenNTN, an Open-Source Framework for Non-Terrestrial Network Channel Simulations! This git provides an implementation of the channel models for dense urban, urban, and suburban scenarios according to the 3GPP TR38.811 standard. It is an extension to the existing Sionna™ framework and integrates into it as another module. <br>

We recommend the following workflow for new users of OpenNTN: <br>

1. **Get an overview of the capabilities of OpenNTN.** <br>
The standards on the channel models are long and complex. We have written a paper outlining the capabilities of OpenNTN and describing the simulation process, condensing the required information of the standards into a short and easy to use format. We recommend to everyone that is not yet well experienced with the standards to first read [this paper](https://www.ant.uni-bremen.de/sixcms/media.php/102/15080/An%20Open%20Source%20Channel%20Emulator%20for%20Non-Terrestrial%20Networks.pdf) and understand the underlying models.
2. **Get familiar with Sionna.** <br>
OpenNTN can be used on its own, but integrating it into Sionna™ does provide the largest benefit, as it allows for the simple setup of end-to-end systems. We recommend to everyone that has not yet worked with Sionna™ before to first do the well documented [tutorials of Sionna™](https://nvlabs.github.io/sionna/phy/tutorials.html), especially the ones using the 38.901 channel models, as the channel models from OpenNTN were designed to be as similar to them as possible to provide the best possible integration into the existing system. 
3. **Run the interactive examples.** <br>
Under the examples section we have collected three notebooks to introduce OpenNTN. In the first, the channel characteristics of NTN channels are showcased to extend the paper and improve the understanding of the channels through visualizations. In the second, a standard end-to-end link level simulation is setup to showcase the integration of OpenNTN into the Sionna™ framework. Lastly, in the machine learning example we train and end-to-end system with a neural receiver to showcase the integration and training of machine learning components using OpenNTN, which is done the same way as in existing Sionna™ implementations.
4. **Use OpenNTN however you want!** <br>
After understanding what you can do with OpenNTN and how to do it, use OpenNTN for your projects however you see fit. Use it out of the box in end-to-end simulations, reuse individual components like the link budget calculations, or integrate your specific work into it, for example in the form of novel antenna structures. 

# Different Sionna™ Versions

Sionna has released versions 1.0+ in 2025, fundamentally reshaping aspects of its internal structure and making it even more modular and powerful in many ways. To support the integration of OpenNTN with both the Sionna™ 1.0+ versions and the legacy version Sionna™ 0.19.2, which is still used in many projects from before 2025, we have seperated the branches of this repository. The branch main is compatible with the versions Sionna™ 1.0+, while branch legacy is compatible with Sionna™ 0.19.2. You can find the two respective installations for OpenNTN in the section below.  <br>
Unfortunately, maintaining both the legacy and the main version of OpenNTN rigorously and introducing new future features to both would be very resource intense. Thus, to maintain our level of quality, we plan to focus only on the main version of OpenNTN in the future and freeze the legacy version in its current state. We still plan to support your requests and update the legacy version if necessary, but new features from future standards will not be introduced to legacy OpenNTN. <br>
You can still find the [documentation of the legacy version of Sionna here](https://jhoydis.github.io/sionna-0.19.2-doc/), including an [installation guide here](https://jhoydis.github.io/sionna-0.19.2-doc/installation.html). <br>

# Installation

Sionna™ is currently in its version 1.0+, but multiple older projects still require the lagacy version 0.19.2. Based on the Sionna™ version you use, select either the main installation for Sionna™ 1.0+ or the legacy installtion for Sionna™ 0.19.2. The provided installation files install.sh and install_legacy.sh are written for Linux based systems. OpenNTN can be used on Windows, however, the convinient installation scripts are not yet provided. Adding these is planned for a future update. <br>

## Main installation for Sionna™ 1.0+

1. Install Sionna <br>
  <code>pip install sionna</code> <br>
For more information on the different installation options we refer the reader to the [sionna documentation](https://nvlabs.github.io/sionna/installation.html). 
2. Download the install.sh file found in this git 
3. Execute the install.sh file <br>
   <code>. install.sh</code>

## Legacy installation for Sionna™ 0.19.2

1. Install Sionna <br>
  <code>pip install sionna==0.19</code> <br>
For more information on the different installation options we refer the reader to the [sionna documentation](https://nvlabs.github.io/sionna/installation.html).
2. Download the install_legacy.sh file found in this git 
3. Execute the install_legacy.sh file <br>
   <code>. install_legacy.sh</code>

## Possible installation issues

This section addresses potential issues that might occur when installing OpenNTN. If you encounter a problem that is not listed here, please feel free to contact us, both to receive support for your installation and to help us work on solutions for releases. Our goal is to provide an easy-to-set-up and easy-to-use tool, and your feedback is highly appreciated as it helps us further enhance OpenNTN. The contact information is listed at the end of the section.

1. Installation on Windows <br>
As mentioned above, installer files are currently only provided for Linux. Until a Windows installer is available, please follow the installation steps manually: download a copy of OpenNTN, move it into the channels directory of your Sionna™ installation, open the __init__.py file inside the channel directory, and add the import statement for OpenNTN: from . import tr38811
2. Installing Sionna™ for the First Time <br>
When using OpenNTN with Sionna™, OpenNTN can only function correctly if Sionna™ is properly installed and working. Therefore, we strongly recommend verifying a working Sionna™ installation before installing OpenNTN. Further instructions can be found in the [sionna documentation](https://nvlabs.github.io/sionna/installation.html).
3. Conflicts With Other Packages <br>
Currently, we are not aware of any package conflicts when using OpenNTN. However, we generally recommend creating separate environments for dedicated applications. If you encounter issues that appear to be caused by other packages, please first try using a clean environment with only the required dependencies. Additionally, please report any observed conflicts so that we can investigate and provide potential solutions.
4. Installation for GPU Usage <br>
Installing Sionna™ with GPU support tends to be more challenging than installing the CPU version, as this also requires CUDA and DrJit installation which must match your device. However, if Sionna™ works correctly on the GPU, OpenNTN should work properly as well. If GPU-related issues only occur when using OpenNTN (but not with Sionna™ itself), please contact us for further support.
5. Example Notebooks Not Running / Kernel Issues <br>
The provided examples are implemented as Jupyter Notebooks. When running these, please ensure that your environment includes all required dependencies to be selectable Jupyter kernel in order to execute the examples successfully.

Contact:
- First contact: Tim Düe — duee@ant.uni-bremen.de
- Second contact: MohammadAmin Vakilifard — vakilifard@ant.uni-bremen.de

# Contents of OpenNTN
OpenNTN implements the models for Non-Terrestrial Networks in the dense urban, urban, and suburban scenarios as defined in the standard 3GPP TR38.811. These are similar to the models defined in 3GPP TR38.901 for terrestrial channels, which are already implemented in Sionna™. To make the use of OpenNTN as easy as possible and make the existing projects and tutorials as reusable as possible, the user interface of the OpenNTN 38811 channels is kept as similar as possible to the user interface of the existing 38901 channels. The user interface was kept as similar as only, only adding necessary new parameters, such as the satellite height, user elevation angle, and new antenna radiation patterns. For a practical demonstration, we refer the reader to the notebooks found in the examples section. <br>

As the standards on the channel models are very large and complex, it can be difficult for newcomers to get an overview of the capabilities of the channel models and an understanding of their process. To adress this, we have written [this paper](https://ieeexplore.ieee.org/document/11202820), in which we summarize the capabilities of the channels and how they actually work in a short and easy fashion.

# Citing OpenNTN
When you use OpenNTN for research, please cite us as: "OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations,T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben, A. Dekorsy​,  	International Workshop on Smart Antennas (WSA), Erlangen, Germany, 16. - 18. September 2025",\
or by using the BibTeX:\
@inproceedings{OpenNTNPaper\
  author = {T. D\"{u}e and M. Vakilifard and C. Bockelmann and D. W\"{u}bben and A. Dekorsy​},\
  year = {2025},\
  month = {Sep},\
  title = {OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations},\
  URL = {https://www.ant.uni-bremen.de/sixcms/media.php/102/15183/OpenNTN_An_OpenSource_Framework_For_NTN_Simulations.pdf}, \
  address={Erlangen, Germany},\
  abstract={Non-Terrestrial Networks (NTNs) are critical enablers of ubiquitous connectivity in future 6G systems, but they also face challenges such as long propagation delays, significant Doppler effects, and diverse channel conditions. Developing and testing communication technologies for NTNs requires realistic and flexible simulation tools. In this work, we present OpenNTN, an open-source software framework for simulating NTN channel models based on the 3GPP TR38.811 standard. OpenNTN is de- signed as an extension to the Python-based Sionna™ framework, providing channel models compatible with existing interfaces, enabling the use of the powerful tools found in Sionna™ . The framework offers a flexible user interface, enabling researchers to investigate diverse scenarios. As an open-source implementation, OpenNTN empowers the research community to fully use and adapt the framework, offering a solution for both fast and easy NTN research using realistic channel models, while also providing the opportunity to extend the model for custom research interests beyond the current state of the standards.},\
  booktitle={International Workshop on Smart Antennas (WSA)}\
}

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

Note: This project is intended to be used with NVIDIA’s Sionna™ framework, which is licensed under the Apache License, Version 2.0 (the "License"). Users must comply with the terms of that license when using this OpenNTN in addition with Sionna™.
