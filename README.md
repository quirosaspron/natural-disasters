Overview
This project aims to automate post-disaster damage assessment using deep learning. By leveraging satellite and aerial imagery,
the model predicts segmentation masks that classify damage into different severity levels. The core of this project is a
Siamese U-Net architecture with a ResNet50 encoder, trained on the xBD dataset.


To get the data and set the train folder visit: https://xview2.org/
📂 damage-assessment-project
│── 📂 train
│   ├── images/         # Pre- and post-disaster images
│   ├── labels/         # JSON annotations for damage levels
│── best_dmg_assessment.h5    # Saved trained model
|── damage_mask.py               # Generates segmentation masks from JSON labels
│── damage_data_generator.py     # Data generator for training
│── siam_unet.py                 # Defines Siamese U-Net model
│── damage_assessment_training.py # Trains the model
│── damage_assessment_test.py     # Tests the trained model
│── demo.py                       # Runs predictions on multiple image pairs
│── display.py                    # Utility to visualize images
│── tst.py                         # Debugging script for model testing
│── 📜 README.md


Model Architecture
The Siamese U-Net model consists of:
🔹 Input: Pairs of pre- and post-disaster images (512, 512, 6).
🔹 Encoder: Shared ResNet50 extracts feature representations.
🔹 Feature Concatenation: Combines extracted features from both images.
🔹 Decoder (U-Net): Upsamples to reconstruct segmentation masks.
🔹 Output: Pixel-wise classification into four damage levels.

Key Features
✔ Automated damage segmentation from satellite images.
✔ Uses transfer learning with a pre-trained ResNet50 encoder.
✔ Custom data generator for dynamically loading training data.
✔ Class imbalance handling using focal loss.
✔ Efficient training & testing pipeline with visualization.


damage_mask.py:
This script contains a function load_mask that generates a segmentation mask from a JSON label file.
The mask represents different damage classes (no-damage, minor-damage, major-damage, destroyed) as pixel values.
The script also includes a test section to visualize the generated mask using matplotlib.

damage_data_generator.py:
This script defines a DamageDataGenerator class that inherits from tensorflow.keras.utils.Sequence. 
It is used to generate batches of pre- and post-disaster images along with their corresponding damage masks for training the model.
The generator pairs pre- and post-disaster images and their corresponding JSON labels, resizes them, and loads them into memory.

siam_unet.py:
This script defines a Siamese U-Net model for damage segmentation. The model uses a shared ResNet50 encoder for both pre- and post-disaster images,
concatenates the features, and then passes them through a decoder to produce a damage segmentation mask.
The model is designed to learn a damage map with a contractive loss layer and reconstruct the final segmentation mask.

damage_assessment_training.py:
This script is responsible for training the Siamese U-Net model. It uses the DamageDataGenerator to load data,
defines a custom focal loss function, and compiles the model with the Adam optimizer.
The script also includes a custom MeanIoU metric for evaluating the model during training.
After training, the model is saved to a file.

damage_assessment_test.py:
This script loads the trained model and tests it on a single pair of pre- and post-disaster images.
It preprocesses the images, makes predictions, and visualizes the results, including the predicted damage mask and an overlay of the mask on the post-disaster image.

demo.py:
This script is a more generalized version of the testing script. It processes multiple image pairs, predicts damage masks, and displays the results using matplotlib.
It also includes a function to overlay the predicted damage mask on the post-disaster image.

display.py:
This script is a simple utility to load and display an image from the dataset. It can be used to quickly visualize the raw images before processing.

tst.py:
Processes a single image pair, and visualizes the results. It includes a function to overlay the predicted damage mask on the post-disaster image and display the results.

Key Components:
Data Loading and Preprocessing: The scripts handle loading and preprocessing of images and JSON labels, resizing them to a consistent size, and generating segmentation masks.

Model Definition: The Siamese U-Net model is defined with a shared encoder and a decoder that produces a damage segmentation mask.

Training: The model is trained using a custom focal loss function.

Testing and Visualization: The trained model is tested on new images, and the results are visualized using matplotlib.
