# PCA-vs-Autoencoder
PCA (Principal Component Analysis) and Autoencoders for efficient feature extraction and data compression.

**Principal Component Analysis**: Principal Component Analysis (PCA) is a dimensionality reduction technique used to simplify high-dimensional data while preserving its essential structure. It works by identifying the directions, called principal components, in which the data varies the most. These components are orthogonal to each other, allowing PCA to transform data into a new coordinate system where most of the variance is captured by the first few components, making it a valuable tool for data visualization and feature selection in various fields, including machine learning and image processing.

**AutoEncoders**: Autoencoders are neural network architectures used for unsupervised learning and dimensionality reduction. They consist of an encoder, which maps input data into a lower-dimensional representation, and a decoder that attempts to reconstruct the original input from this representation. The autoencoder learns to capture essential features in the data by minimizing the difference between the input and the reconstructed output, making it a powerful tool for tasks like data compression, anomaly detection, and feature learning in deep learning applications.

In this project, we apply the PCA and the autoencoder (AE) to a collection of handwritten digit images from the USPS dataset. The python script expects the data to be in a .mat fortmat (for example "USPS.mat"). We reduce the dimensionailty of the given matrix using both the techniques and reconstruct the orginal matrix from the reduced data. We train the autoencoder for 300 epoch over the dataset. To measure the reconstruction error, we utilize the frobeniu normalization error. We evaluate PCA with the following number of principal components {32, 64, 128}. Similarly, we evaluate AutoEncoder with the following inner dimension {32, 64, 128}.

| p=d | PCA   | Autoencoder |
|-----|-------|-------------|
| 32  | 129.59| 129.85      |
| 64  | 85.87 | 86.26       |
| 128 | 45.62 | 46.39       |



