# Shape Detection using Topological Data Analysis

The goal of our project was to develop a model capable of differentiating between various 3-dimensional geometric shapes using a topological data analysis approach.

## Key Components

Here's an overview of the key components of our project:

- **Data Generation:** We generated point clouds representing different 3D shapes, introducing randomness for natural variations.

- **Data Standardization:** We standardized the data to ensure comparability by aligning shapes with a common reference frame.

- **Topological Analysis:** We used topological data analysis to extract key features from the point clouds, including persistence diagrams and persistence images.

- **Machine Learning:** We built a Support Vector Machine (SVM) model to classify shapes based on their topological features.

- **Robustness Testing:** We assessed the model's robustness by testing it on translated, rotated, and challenging shapes.

- **4D Extension:** We extended the model to recognize shapes in 4-dimensional space.

## Code Modules

Our project's code is organized into several modules:

- **m_create_pointclouds.py:** Generates initial point clouds of various shapes.
  
- **m_rotate_translate.py:** Implements shape rotation and translation for robustness testing.
  
- **standardize_ap.py:** Standardizes the data for consistent analysis.
  
- **remove_ball.py:** Removes central balls from shapes to distinguish similar topologies.
  
- **create_PD_and_PI.py:** Computes persistence diagrams and images for machine learning.
  
- **svm.ipynb:** Builds and optimizes the SVM model, includes visualizations.

Please be aware that there are also 4D versions available along with supplementary data creation files.

## Data Availability

We have provided the data used in our project. You can directly run the `svm.ipynb` notebook to perform shape classification. However, if you wish to create your own versions of the data, you can modify the previously stated files according to your requirements. To generate your data, run these files sequentially. Additionally, execute `combine3.py` to combine your generated data. Afterward, you can run `svm.ipynb` to perform shape classification on your custom data.


