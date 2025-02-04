import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

def siamese_unet(input_shape=(512, 512, 3)):
    """
    Siamese U-Net for damage segmentation.
    - Two shared encoders (ResNet50 backbone).
    - Learns a damage map with a contractive loss layer.
    - Decoder reconstructs the final segmentation mask.
    """

    # === ENCODER ===
    def build_encoder(input_layer):
        """Pretrained ResNet50 encoder (without top layers)."""
        base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=input_layer)
        encoder_output = base_model.get_layer("conv4_block6_out").output  # Feature extraction
        return base_model.input, encoder_output


    pre_input = Input(shape=input_shape, name="pre_disaster_image")
    post_input = Input(shape=input_shape, name="post_disaster_image")

    shared_encoder = tf.keras.models.Model(*build_encoder(pre_input))  # Shared encoder model

    pre_encoded = shared_encoder(pre_input)  # Extract features from pre-disaster image
    post_encoded = shared_encoder(post_input)  # Extract features from post-disaster image

    # === CONTRACTIVE LOSS LAYER ===
    damage_map = Concatenate()([pre_encoded, post_encoded])

    # === DECODER ===
    def decoder_block(input_tensor, filters):
        """Upsampling block with convolution layers."""
        x = UpSampling2D((2, 2))(input_tensor)
        x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        return x
    

    x = decoder_block(damage_map, 512)
    x = decoder_block(x, 256)
    x = decoder_block(x, 128)
    x = decoder_block(x, 64)
    x = Conv2D(4, (1, 1), activation="softmax", padding="same", name="damage_mask")(x)  # Output mask

    # === MODEL ===
    model = Model(inputs=[pre_input, post_input], outputs=x, name="Siamese_U-Net")
    return model

# testing
if __name__ == "__main__":
    model = siamese_unet()
    model.summary()

