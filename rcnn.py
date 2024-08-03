# Region Based Convolutional Neural Networks

# Inefficiencies:

def build_backbone(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    return inputs, x

def roi_pooling(features, rois, output_size):
    batch_indices = tf.zeros(tf.shape(rois)[0], dtype=tf.int32)
    
    pooled_features = tf.image.crop_and_resize(
        features, rois, batch_indices, output_size
    )
    return pooled_features

def rcnn_model(input_shape=(224, 224, 3), num_classes=21, num_rois=256):
    inputs = Input(shape=input_shape)
    
    # Backbone Network
    backbone_input, base_layers = build_backbone(input_shape)
    
    # Region Proposals (Placeholder input)
    rois = Input(shape=(num_rois, 4))  # [num_rois, (y1, x1, y2, x2)]
    
    # RoI Pooling
    pooled_features = Lambda(lambda x: roi_pooling(x[0], x[1], (7, 7)))([base_layers, rois])
    
    # Fully Connected Layers
    x = Flatten()(pooled_features)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    
    # Classification and Bounding Box Regression
    class_logits = Dense(num_classes, activation='softmax')(x)
    bbox_regression = Dense(num_classes * 4)(x)  # 4 coordinates for each class
    
    model = Model(inputs=[backbone_input, rois], outputs=[class_logits, bbox_regression])
    
    return model

