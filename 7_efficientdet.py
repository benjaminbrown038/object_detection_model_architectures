def load_efficientnet(input_shape=(224, 224, 3), num_classes=1000, include_top=True, weights='imagenet'):
    base_model = EfficientNetB0(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        classes=num_classes
    )
    return base_model

def create_custom_efficientnet(input_shape=(224, 224, 3), num_classes=10):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Freeze base model layers
    base_model.trainable = False

    # Add new classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model
