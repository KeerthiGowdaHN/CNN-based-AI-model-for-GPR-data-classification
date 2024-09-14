# CNN-based-AI-model-for-GPR-data-classification
 Ground Penetrating Radar (GPR) AI Model

This project implements a Ground Penetrating Radar (GPR) data processing and classification pipeline using a Convolutional Neural Network (CNN). The pipeline involves data preprocessing, AI-based analysis for subsurface feature classification, and 3D visualization of the results. The key stages of this project are:

1. Data Preprocessing:
   - Noise Reduction: Reduces noise in the GPR data by subtracting the mean of each signal.
   - Gain Adjustment: Amplifies weaker signals for better representation of subsurface features.
   - Time-to-Depth Conversion: Converts the time-based data to depth-based data, allowing better spatial interpretation.
   - Normalization: Ensures the data is scaled between 0 and 1 for uniform model input.

2. AI-Based Classification:
   - A Convolutional Neural Network (CNN) is trained on the preprocessed GPR data. The model is designed to classify subsurface features based on binary labels (e.g., presence/absence of a feature).
   - The CNN consists of multiple 3D convolutional layers followed by pooling layers, a flattening layer, and dense layers for classification.

3. Classification and 3D Visualization:
   - The trained model is used to predict subsurface features for new GPR data.
   - The results are visualized in a 3D space using `pyvista`, where the classified subsurface structures are represented using a color-coded grid.

Key Technologies:
- Python
- TensorFlow/Keras: For building and training the CNN model.
- Numpy: For data manipulation and preprocessing.
- PyVista: For 3D visualization of the classified GPR data.

How to Run the Project:
1. Load the GPR data (`.GPZ` files) and preprocess them using the included functions.
2. Train the CNN model using the preprocessed data and corresponding labels.
3. Use the trained model to classify new GPR data and visualize the results in 3D.
![Screenshot 2024-09-15 001300](https://github.com/user-attachments/assets/d3296ff3-61a6-4b06-b88d-cc0f7eb383c3)

