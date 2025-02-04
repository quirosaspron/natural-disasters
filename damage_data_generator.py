import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
from damage_mask import load_mask

class DamageDataGenerator(Sequence):
    def __init__(self, image_dir, label_dir, batch_size=8, img_size=(512, 512), shuffle=True):
        """
        Data generator for damage assessment task.

        Args:
            image_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing labels (JSON files).
            batch_size (int): Number of samples per batch.
            img_size (tuple): Target image size (H, W).
            shuffle (bool): Whether to shuffle data after each epoch.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        # Get all pre-disaster images
        self.image_pairs = self._load_image_pairs()

        self.indices = np.arange(len(self.image_pairs))
        self.on_epoch_end()

    def _load_image_pairs(self):
        """Finds and pairs pre- and post-disaster images with their corresponding masks."""
        image_pairs = []
        for filename in os.listdir(self.image_dir):
            if "_pre_disaster.png" in filename:
                base_name = filename.replace("_pre_disaster.png", "")
                pre_image_path = os.path.join(self.image_dir, filename)
                post_image_path = os.path.join(self.image_dir, f"{base_name}_post_disaster.png")
                mask_path = os.path.join(self.label_dir, f"{base_name}_post_disaster.json")

                if os.path.exists(post_image_path) and os.path.exists(mask_path):
                    image_pairs.append((pre_image_path, post_image_path, mask_path))

        return sorted(image_pairs)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.image_pairs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_pre, batch_post, batch_masks = self.__data_generation(batch_indices)

        return (batch_pre, batch_post), batch_masks

    def __data_generation(self, batch_indices):
        """Load batch images and masks."""
        batch_pre = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        batch_post = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((self.batch_size, *self.img_size), dtype=np.uint8)

        for i, idx in enumerate(batch_indices):
            pre_path, post_path, mask_path = self.image_pairs[idx]

            # Load pre-disaster image
            pre_image = cv2.imread(pre_path)
            pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
            pre_image = cv2.resize(pre_image, self.img_size)

            # Load post-disaster image
            post_image = cv2.imread(post_path)
            post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)
            post_image = cv2.resize(post_image, self.img_size)

            # Load mask
            mask = load_mask(mask_path, img_size=self.img_size)
            batch_pre[i] = pre_image
            batch_post[i] = post_image
            batch_masks[i] = mask
        return batch_pre, batch_post, batch_masks

    def on_epoch_end(self):
        """Shuffle data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Testing
if __name__ == "__main__":
    generator = DamageDataGenerator(image_dir="train/images", label_dir="train/labels", batch_size=2)
    (X_pre, X_post), Y = generator[0]

    print("Pre-disaster image shape:", X_pre.shape)
    print("Post-disaster image shape:", X_post.shape)
    print("Mask shape:", Y.shape)

