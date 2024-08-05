def build_backbone(input_shape=(256, 256, 3)):
    backbone = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    return backbone.input, backbone.output


def yolo_head(x, num_classes, num_anchors):
    # Detection head: (grid_size, grid_size, num_anchors * (num_classes + 5))
    yolo_head = Conv2D(num_anchors * (num_classes + 5), (1, 1), padding='same')(x)
    return yolo_head

def yolo_model(input_shape=(256, 256, 3), num_classes=20, num_anchors=9):
    inputs = Input(shape=input_shape)
    
    # Backbone network
    backbone_input, backbone_output = build_backbone(input_shape)
    
    # YOLO Detection Head
    yolo_out = yolo_head(backbone_output, num_classes, num_anchors)
    
    # Create the YOLO model
    model = Model(inputs=backbone_input, outputs=yolo_out)
    
    return model

def yolo_loss(num_classes, num_anchors):
    def loss(y_true, y_pred):
        # Reshape predictions
        grid_size = tf.shape(y_pred)[1]
        y_pred = tf.reshape(y_pred, (-1, grid_size, grid_size, num_anchors, num_classes + 5))
        
        # Extract ground truth and predictions
        y_true_boxes = y_true[..., :4]  # (y_true boxes)
        y_true_conf = y_true[..., 4]    # (confidence)
        y_true_cls = y_true[..., 5:]    # (class labels)

        y_pred_boxes = y_pred[..., :4]  # (predicted boxes)
        y_pred_conf = y_pred[..., 4:5]  # (predicted confidence)
        y_pred_cls = y_pred[..., 5:]    # (predicted class scores)

        # Compute losses
        box_loss = tf.reduce_sum(tf.square(y_true_boxes - y_pred_boxes), axis=-1)
        conf_loss = tf.reduce_sum(tf.square(y_true_conf - y_pred_conf), axis=-1)
        class_loss = tf.reduce_sum(tf.square(y_true_cls - y_pred_cls), axis=-1)

        return box_loss + conf_loss + class_loss

    return loss

