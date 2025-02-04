import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras import backend as K

# Configuration
TEST_IMAGE_DIR = "test_images/"  # Folder containing pre1.png, post1.png, etc.
NUM_TEST_PAIRS = 13              # Number of image pairs to process
IMG_SIZE = (512, 512)            # Input size for the model

# Focal loss function (required for loading the model)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true_onehot = K.one_hot(K.cast(y_true, 'int32'), 4)
        y_true_onehot = K.cast(y_true_onehot, y_pred.dtype)
        cross_entropy = -y_true_onehot * K.log(y_pred)
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fn

# Load the trained model
def load_model():
    return tf.keras.models.load_model(
        "best_dmg_assessment.h5",
        custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=50)}
    )

# Preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, IMG_SIZE)             # Resize to model input size
    return img

# Process a single image pair and display results
def process_and_display_pair(model, pre_path, post_path, pair_id):
    # Load and preprocess images
    pre_img = preprocess_image(pre_path)
    post_img = preprocess_image(post_path)
    
    if pre_img is None or post_img is None:
        print(f"Skipping pair {pair_id} - missing image(s)")
        return

    # Prepare model input
    pre_input = np.expand_dims(pre_img, axis=0)  # Add batch dimension
    post_input = np.expand_dims(post_img, axis=0)

    # Make prediction
    pred = model.predict([pre_input, post_input])
    damage_mask = np.argmax(pred, axis=-1)[0]  # Get predicted mask

    # Define colors for damage classes
    mask_colors = np.array([
        [0, 0, 0],      # No damage (black)
        [0, 255, 0],    # Minor damage (green)
        [255, 255, 0],  # Major damage (yellow)
        [255, 0, 0]     # Destroyed (red)
    ])

    # Convert mask to RGB for visualization
    damage_mask_rgb = mask_colors[damage_mask]

    # Overlay mask on post-disaster image
    overlay = cv2.addWeighted(post_img.astype(np.uint8), 0.7, 
                              damage_mask_rgb.astype(np.uint8), 0.3, 0)

    # Display results using matplotlib
    plt.figure(figsize=(15, 5))

    # Pre-disaster image
    plt.subplot(1, 3, 1)
    plt.title("Pre-disaster Image")
    plt.imshow(pre_img)
    plt.axis('off')

    # Post-disaster image
    plt.subplot(1, 3, 2)
    plt.title("Post-disaster Image")
    plt.imshow(post_img)
    plt.axis('off')

    # Overlay of predicted mask on post-disaster image
    plt.subplot(1, 3, 3)
    plt.title("Predicted Damage Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    # Show the plot
    plt.suptitle(f"Image Pair {pair_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Main function to run the demo
def run_demo():
    # Load the model
    model = load_model()
    print("Model loaded successfully")

    # Process each image pair
    for i in range(1, NUM_TEST_PAIRS + 1):
        pre_path = os.path.join(TEST_IMAGE_DIR, f"pre{i}.png")
        post_path = os.path.join(TEST_IMAGE_DIR, f"post{i}.png")

        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            print(f"Skipping pair {i} - files missing")
            continue

        print(f"Processing pair {i}...")
        process_and_display_pair(model, pre_path, post_path, i)

    print("Demo complete.")

if __name__ == "__main__":
    run_demo()
