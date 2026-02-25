 Bean Leaf Disease Detection(Deep Learning)

CNN model for classifying bean leaf diseases using TensorFlow Datasets (TFDS).

Detects angular_leaf_spot, bean_rust, and healthy leaves from the TFDS "beans" dataset (~10K images). Achieves 95% training accuracy and 80% test accuracy after 55 epochs.

 Tools Used
- Python 3.x
- TensorFlow 2.x (Keras Sequential API)
- TensorFlow Datasets (TFDS)
- Matplotlib (visualizations)
- NumPy (array operations)

 How It Works

1. Data Loading  
   Downloads "beans" dataset via TFDS with train/validation/test splits.

2. Preprocessing  
   - Resizes images to 150x150 pixels  
   - Normalizes pixel values (0-1 range)  
   - Applies data augmentation & batching (32 images/batch)

3. Model Architecture  
   
   Conv2D(32) → BatchNorm → MaxPool → 
   Conv2D(64) → BatchNorm → MaxPool → 
   Conv2D(128) → BatchNorm → MaxPool → 
   Flatten → Dense(128, Dropout=0.5) → Dense(3, softmax)
   

4. Training  
   - Optimizer: Adam  
   - Loss: Sparse categorical crossentropy  
   - Epochs: 55  
   - Batch size: 32

5. Evaluation  
   - Train accuracy: 95.1%  
   - Test accuracy: 79.7%  
   - Visualizes predictions on validation samples

 Quick Start
bash
pip install tensorflow tensorflow-datasets matplotlib numpy
jupyter notebook tfds-5.ipynb


Run cells sequentially - GPU recommended for faster training. Interactive prediction on image index 0-31 available.
