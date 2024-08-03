def build_backbone(input_shape=(300, 300, 3)):
    backbone = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    return backbone.input, backbone.output


def ssd_detection_heads(x, num_classes, num_anchors):
    # Detection heads for class prediction
    cls_head = Conv2D(num_anchors * num_classes, (3, 3), padding='same', activation='softmax')(x)
    
    # Detection heads for bounding box regression
    bbox_head = Conv2D(num_anchors * 4, (3, 3), padding='same')(x)
    
    return cls_head, bbox_head

def ssd_model(input_shape=(300, 300, 3), num_classes=21, num_anchors=9):
    inputs = Input(shape=input_shape)
    
    # Backbone network
    backbone_input, backbone_output = build_backbone(input_shape)
    
    # Detection heads for different feature maps
    cls_head, bbox_head = ssd_detection_heads(backbone_output, num_classes, num_anchors)
    
    # Create the SSD model
    model = Model(inputs=backbone_input, outputs=[cls_head, bbox_head])
    
    return model

def ssd_loss(num_classes):
    def loss(y_true, y_pred):
        cls_loss = tf.keras.losses.CategoricalCrossentropy()(y_true[0], y_pred[0])
        bbox_loss = tf.keras.losses.MeanSquaredError()(y_true[1], y_pred[1])
        return cls_loss + bbox_loss
    return loss

