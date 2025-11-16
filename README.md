<!-- # Colour Characterisation in Optical See-Through Augmented Reality
This repository contains the main codes used for the thesis project titled "Exploring Colours in Optical See-Through Augmented Reality" as a part of the Erasmus Mundus COSI master's programme  
 
"Classical Methods (Matrix and Polynomial Regression)" folder contains the MATLAB scripts for the matrix + LUT method and the polynomial regression method of characterisation.  

"DL Model for characterisation" folder contains the python script of the model that performs characterisation. This is a novel approach introduced in the thesis.  

"Psychophysical Experiments" folder contains C# code for the 2AFC experiment (ColourPatch_Compare.cs) and the Just noticeable difference experiment (JND.cs), as well as a code (AutoTester.cs) to automate the 2AFC experiment on Unity for testing purposes.
-->
# Colour Characterisation in Optical See-Through Augmented Reality  
**Exploring Colours in Optical See-Through Augmented Reality**  
*Master’s Thesis – Erasmus Mundus MS COSI*

---

## Overview  
This repository contains the code and resources used in the thesis project **“Exploring Colours in Optical See-Through Augmented Reality (OST-AR)”**.  
The project investigates how colours are perceived and reproduced in OST AR devices, comparing classical colour characterisation techniques with a deep learning based method.  
It also includes psychophysical experiments used to validate perceptual outcomes as well as data collection to create a novel dataset for the DL algorithm.

---

## Objectives  
- Characterise colour in optical see-through AR systems.  
- Implement and evaluate classical characterisation techniques (matrix, LUT, polynomial regression).  
- Develop and test a deep learning model on novel data.  
- Validate results through psychophysical experiments (2AFC, JND).  

---

## Repository Structure  
* Classical Methods (Matrix and Polynomial Regression)  
    * MATLAB scripts for matrix + LUT and polynomial regression techniques.  
* DL Model for characterisation  
    * Python implementation of the deep-learning model described in the thesis.  
* Psychophysical Experiments  
    * ColourPatch_Compare.cs  (2AFC experiment)  
    * JND.cs                  (Just Noticeable Difference experiment)  
    * AutoTester.cs           (Unity based automation for 2AFC testing)  
* README.md

---

## Components

### **Classical Methods (MATLAB)**  
Implements traditional colour-characterisation pipelines, including:  
- Matrix transforms with Look up table for non linearity
- Polynomial regression  

### **Deep Learning Model (Python)**  
A neural network trained to map between captured/displayed colours and target colour values under OST AR conditions described in the novel image dataset for about 5000 images.

### **Psychophysical Experiments (C# / Unity)**  
- **2AFC (Two-Alternative Forced Choice)**: People compare AR-rendered colour patches with reference patches.  
- **JND (Just Noticeable Difference)**: People quantify perceptual thresholds for colour differences in AR for 5 colour centers.  
- **Automation Script**: streamlines experiment development and testing.

---

## Procedure 

### **Prerequisites**
- MATLAB  
- Python 3 (with TensorFlow/PyTorch, NumPy, etc.)  
- Unity (for psychophysical experiments) with MRTK3 installed   
- HoloLens 2
- Konica Minolta CS 2000 Spectroradiometer

---

## Usage

### **Classical Methods**
1. Open MATLAB scripts in the `Classical Methods` folder.  
2. Load your calibration data collected using the spectroradiometer.  
3. Run the characterisation pipeline to generate transformations.

### **Deep-Learning Model**
1. Install the regular Python dependencies required for running deep learning models.
2. Prepare the dataset of images (for this project images were captured through the Hololens 2 screen with different backgrounds while different shapes and colours were displayed).
3. Run the code of the dataset.

### **Psychophysical Experiments**
1. Open the Unity project with the correct packages such as MRTK3 installed and import the provided C# scripts.  
2. Deploy to Hololens 2.  
3. Run the 2AFC or JND test with participants.  
4. Export and analyse the collected results.

---

## Results & Findings  
- Classical methods provide baseline accuracy for OST AR colour characterisation.   
- The deep-learning model captures non-linear optical phenomena caused by the Hololens 2 screen and outperforms classical methods on delta E 2000 metric.  

---

## Limitations & Future Work  
- Limited dataset size for training and validation.  
- Deep learning model may require device specific calibration.  

**Potential future improvements:**  
- Real time colour correction in AR.  
- Larger datasets under varied illumination conditions.  
- Cross device generalisation.  
- HDR and multispectral extensions.  
- User adaptive calibration.

---

## How to Cite  
If you use this work, please cite the associated thesis or repository:

> *Chowdhury, Dipayan*. “Exploring Colours in Optical See-Through Augmented Reality.”  
> Master’s Thesis, Erasmus Mundus MS COSI, *2025*.

