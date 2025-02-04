import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
from damage_data_generator import DamageDataGenerator
from siam_unet import siamese_unet
from tensorflow.keras import backend as K

# === CONFIGURATION ===
IMAGE_DIR = "train/images/"
LABEL_DIR = "train/labels/"
BATCH_SIZE = 1
INPUT_SHAPE = (512, 512, 3)
EPOCHS = 3
MODEL_SAVE_PATH = "damage_final.h5"

# === LOAD DATA ===
train_generator = DamageDataGenerator(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    batch_size=BATCH_SIZE,
    img_size=INPUT_SHAPE[:2]
)

# == CUSTOM METRIC ==
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_inter = self.add_weight(name='total_inter', initializer='zeros', shape=(num_classes,))
        self.total_union = self.add_weight(name='total_union', initializer='zeros', shape=(num_classes,))
        self.epsilon = 1e-7  # To avoid division by zero

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class labels
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute confusion matrix
        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=self.num_classes)
        
        # Calculate intersection and union for each class
        inter = tf.linalg.diag_part(cm)
        union = tf.reduce_sum(cm, axis=0) + tf.reduce_sum(cm, axis=1) - inter
        
        # Update state variables
        self.total_inter.assign_add(inter)
        self.total_union.assign_add(union)

    def result(self):
        # Calculate IoU for each class and return mean
        iou_per_class = self.total_inter / (self.total_union + self.epsilon)
        return tf.reduce_mean(iou_per_class)

    def reset_states(self):
        self.total_inter.assign(tf.zeros_like(self.total_inter))
        self.total_union.assign(tf.zeros_like(self.total_union))



# === FOCAL LOSS FUNCTION ===
alpha = [5, 1, 1, 1]
def focal_loss(gamma=2.0, alpha=alpha):
    def focal_loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert sparse labels to one-hot
        y_true_onehot = K.one_hot(K.cast(y_true, 'int32'), 4)  # 4 classes
        y_true_onehot = K.cast(y_true_onehot, y_pred.dtype)
        
        # Calculate focal loss
        cross_entropy = -y_true_onehot * K.log(y_pred)
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fn


# === BUILD MODEL ===
model = siamese_unet(input_shape=INPUT_SHAPE)

# === COMPILE MODEL ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(gamma=2.0, alpha=alpha),  # Aggressive punishment,
    metrics=['accuracy', MeanIoU(num_classes=4)])

# === TRAIN MODEL ===
model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    verbose=1
)

# === SAVE TRAINED MODEL ===
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

