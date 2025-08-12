## Deep Neural Encoder-Decoder Model To Relate fMRI Brain Activity With Naturalistic Stimuli

**Students:** Florian David | florian.david@epfl.ch  / Rodrigo Almeida Saraiva Dos Anjos | rodrigo.anjos@epfl.ch

**Institution:** Neuro-X, EPFL  
**Last Update:** 09/07/2025  
**Host Lab:** Medical Image Processing Lab (MIP:Lab), EPFL  
**Supervisor:** Michael Chan  

---

#### Abstract

We propose an end-to-end deep neural encoder-decoder model to encode and decode brain activity in response to naturalistic stimuli using functional magnetic resonance imaging (fMRI) data. Leveraging temporally correlated input from consecutive film frames, we employ temporal convolutional layers in our architecture, which effectively allows to bridge the temporal resolution gap between natural movie stimuli and fMRI acquisitions. Our model predicts activity of voxels in and around the visual cortex and performs reconstruction of corresponding visual inputs from neural activity. Finally, we investigate brain regions contributing to visual decoding through saliency maps. We find that most contributing regions are the middle occipital area, the fusiform area, and the calcarine, respectively employed in shape perception, complex recognition in particular face perception, and basic visual features such as edges and contrasts. These functions being strongly solicited are in line with the decoder's capability to reconstruct edges, faces, and contrasts. All in all, this suggests the possibility to probe our understanding of visual processing in films using as proxy to the behaviour of deep learning models such as the one proposed in this paper.

![image](https://github.com/user-attachments/assets/c85daf23-b65d-4993-b7f7-a75fc379c799)

---

### Repository Structure
This repository contains several key scripts and notebooks essential to the project:

- **run.ipynb**: Pipelines to preprocess the data, store it into train/val/test sets, load the data, train a model, and test it.
- **dataset.py**: Definition of functions for processing, storing, and loading datasets used in the project.
- **models.py**: Definition of model architectures and functions for training and testing the models.
- **visualisation.py**: Definition of functions for visualizing the results of the models.
- **imports.py**: Contains all necessary imports.
- **mask_schaefer1000_4609.npy**: Visual mask used in the study covering 4609 voxels based on Schaefer's 1000 parcelation.
- **Project Overview.pdf**: Presentation of the project covering data preprocessing, model architectures and results.
- **Description of subfolders and files in main folder (Video_dataset).pdf**: Structure of the folder Video_dataset on the MIP server.
