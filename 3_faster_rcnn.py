# Requires Fine Tuning of Multiple Components: RPN, Classifier, Bounding Box Regressor

def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(base_layers)
    x = Conv2D(num_anchors * 2, (1, 1), padding='valid')(x)  # Classification (objectness) score
    x = Conv2D(num_anchors * 4, (1, 1), padding='valid')(x)  # Bounding box regression
    
    return x

def roi_pooling(features, rois, output_size):
    """
    Applies RoI pooling to the features and boxes.
    
    Parameters:
    features (tensor): The feature map from the backbone network.
    rois (tensor): The region proposals.
    output_size (tuple): The size of the output feature map.
    
    Returns:
    tensor: The RoI-pooled feature map.
    """
    batch_indices = tf.zeros(tf.shape(rois)[0], dtype=tf.int32)
    
    pooled_features = tf.image.crop_and_resize(
        features, rois, batch_indices, output_size
    )
    return pooled_features

def faster_rcnn_model(input_shape=(224, 224, 3), num_classes=21, num_anchors=9, num_rois=256):
    inputs = Input(shape=input_shape)
    
    # Backbone (e.g., VGG16)
    backbone = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    base_layers = backbone.get_layer('block5_pool').output
    
    # Region Proposal Network
    rpn_output = rpn(base_layers, num_anchors)
    
    # RoI Pooling
    rois = Input(shape=(num_rois, 4))  # [num_rois, (y1, x1, y2, x2)]
    pooled_features = Lambda(lambda x: roi_pooling(x[0], x[1], (7, 7)))([base_layers, rois])
    
    # Fully Connected Layers
    x = Flatten()(pooled_features)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    
    # Classification and Bounding Box Regression
    class_logits = Dense(num_classes, activation='softmax')(x)
    bbox_regression = Dense(num_classes * 4)(x)  # 4 coordinates for each class
    
    model = Model(inputs=[inputs, rois], outputs=[rpn_output, class_logits, bbox_regression])
    
    return model

