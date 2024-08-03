
def build_backbone(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = Conv2D(256, (1, 1))(x)  # Adjust channels for transformer input
    return base_model.input, x

def transformer_encoder(inputs, num_heads=8, ff_dim=512, dropout_rate=0.1):
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization()(out1)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    out2 = Add()([out1, ff_output])
    out2 = LayerNormalization()(out2)

    return out2

def detr(input_shape=(256, 256, 3), num_classes=91, num_queries=100):
    inputs = Input(input_shape)

    # Backbone
    backbone_input, backbone_output = build_backbone(input_shape)
    
    # Flatten the backbone output to prepare for the transformer
    x = tf.keras.layers.Reshape((-1, 256))(backbone_output)  # Assuming 256 channels

    # Transformer Encoder
    for _ in range(6):  # Number of encoder layers
        x = transformer_encoder(x)

    # Detection head
    x = Dense(num_queries * (num_classes + 4), activation=None)(x)  # For bounding boxes and classes
    x = Reshape((num_queries, num_classes + 4))(x)  # Reshape to (num_queries, num_classes + 4)
    
    # Model
    model = Model(inputs=inputs, outputs=x)

    return model

