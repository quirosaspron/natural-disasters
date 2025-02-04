import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras import backend as K


class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_inter = self.add_weight(name='total_inter', initializer='zeros', shape=(num_classes,))
        self.total_union = self.add_weight(name='total_union', initializer='zeros', shape=(num_classes,))
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=self.num_classes)
        inter = tf.linalg.diag_part(cm)
        union = tf.reduce_sum(cm, axis=0) + tf.reduce_sum(cm, axis=1) - inter
        self.total_inter.assign_add(inter)
        self.total_union.assign_add(union)

    def result(self):
        iou_per_class = self.total_inter / (self.total_union + self.epsilon)
        return tf.reduce_mean(iou_per_class)

    def reset_states(self):
        self.total_inter.assign(tf.zeros_like(self.total_inter))
        self.total_union.assign(tf.zeros_like(self.total_union))

    def get_config(self):
        config = super(MeanIoU, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config



alpha = [100, 1, 1, 1]

def focal_loss(gamma=2.0, alpha=alpha):
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

# Load trained model
#model = tf.keras.models.load_model("damage_assessment.h5")

model = tf.keras.models.load_model(
    "final_model.h5",
    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=alpha),
        'MeanIoU': lambda **kwargs: MeanIoU(num_classes=4, **kwargs) }
)
# Define paths for a single pre- and post-disaster image
TEST_IMAGE_DIR = "train/images/"
PRE_IMAGE_NAME = "palu-tsunami_00000166_pre_disaster.png"
POST_IMAGE_NAME = "palu-tsunami_00000166_post_disaster.png"
pre_image_path = os.path.join(TEST_IMAGE_DIR, PRE_IMAGE_NAME)
post_image_path = os.path.join(TEST_IMAGE_DIR, POST_IMAGE_NAME)

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (512, 512))  # Resize to match model input
    # img = img / 255.0  # Normalize pixel values to [0, 1] (if required by the model)
    return img

# Load and preprocess images
pre_img = preprocess_image(pre_image_path)
post_img = preprocess_image(post_image_path)

# Add batch dimension
pre_img = np.expand_dims(pre_img, axis=0)  # Shape: (1, 512, 512, 3)
post_img = np.expand_dims(post_img, axis=0)  # Shape: (1, 512, 512, 3)

# Predict damage class
pred_damage = model.predict([pre_img, post_img])

print(pred_damage.shape)  # Should be (1, 512, 512, num_classes)
print(np.unique(np.argmax(pred_damage, axis=-1)))  # Should be [0, 1, 2, 3]

# Get predicted class (assuming softmax output)
damage_mask = np.argmax(pred_damage, axis=-1)[0]  # Shape: (512, 512)

# Visualize the images and mask
plt.figure(figsize=(15, 5))

# Pre-disaster image
plt.subplot(1, 3, 1)
plt.title("Pre-disaster Image")
plt.imshow(pre_img[0])  # Remove batch dimension for display
plt.axis('off')

# Post-disaster image
plt.subplot(1, 3, 2)
plt.title("Post-disaster Image")
plt.imshow(post_img[0])  # Remove batch dimension for display
plt.axis('off')

# Damage mask
plt.subplot(1, 3, 3)
plt.title("Damage Mask")
plt.imshow(damage_mask, cmap='hot_r')  # Use a colormap like 'jet' for better visualization
plt.axis('off')
plt.colorbar(ticks=[0, 1, 2, 3], label="Damage Class")

plt.show()
