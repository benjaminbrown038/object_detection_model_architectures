# Fast Region Based Convolutional Neural Networks

def roi_pooling(features, boxes, output_size):
    """
    Applies RoI pooling to the features and boxes.
    
    Parameters:
    features (tensor): The feature map from the backbone network.
    boxes (tensor): The region proposals.
    output_size (tuple): The size of the output feature map.
    
    Returns:
    tensor: The RoI-pooled feature map.
    """
    # Box indices (batch indices for each box, assumed to be 0)
    batch_indices = tf.zeros(tf.shape(boxes)[0], dtype=tf.int32)
    
    # Apply crop_and_resize
    pooled_features = tf.image.crop_and_resize(
        features, boxes, batch_indices, output_size
    )
    return pooled_features

def fast_rcnn_model(input_shape=(224, 224, 3), num_classes=21, num_rois=256):
    inputs = Input(shape=input_shape)
    
    # Backbone (e.g., a simplified VGG or ResNet)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Example of features extraction
    features = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    
    # Region proposals (e.g., bounding boxes from a separate RPN)
    rois = Input(shape=(num_rois, 4))  # [num_rois, (y1, x1, y2, x2)]
    
    # RoI Pooling
    pooled_features = Lambda(lambda x: roi_pooling(x[0], x[1], (7, 7)))([features, rois])
    
    # Fully Connected Layers
    x = Flatten()(pooled_features)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    
    # Classification and Bounding Box Regression
    class_logits = Dense(num_classes, activation='softmax')(x)
    bbox_regression = Dense(num_classes * 4)(x)  # 4 coordinates for each class
    
    model = Model(inputs=[inputs, rois], outputs=[class_logits, bbox_regression])
    
    return model

