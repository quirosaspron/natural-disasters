"""Just a file to quickly visualize the dataset images"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and display the raw image
image_path = "train/images/hurricane-michael_00000051_post_disaster.png"

img = cv2.imread(image_path)  # Load image
# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.resize(img, (512, 512))  # Resize to match model input
    # img = img.astype(np.float32) / 255.0  # Normalize pixel values
    return img

if img is None:
    print("Error: Image not found or failed to load!")
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct visualization
    img = preprocess_image(image_path)
    plt.imshow(img)
    plt.title("Loaded Image")
    plt.axis("off")
    plt.show()

