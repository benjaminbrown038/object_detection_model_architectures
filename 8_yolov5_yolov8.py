def build_backbone(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    return inputs, x


def build_neck(x):
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x1 = UpSampling2D()(x1)
    
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x2 = UpSampling2D()(x2)
    
    x = Concatenate()([x1, x2])
    return x

def yolo_head(x, num_classes, num_anchors):
    # Detection head: (grid_size, grid_size, num_anchors * (num_classes + 5))
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), padding='same')(x)
    return x

def yolov5_model(input_shape=(256, 256, 3), num_classes=20, num_anchors=9):
    inputs, backbone_output = build_backbone(input_shape)
    
    # Build Neck
    neck_output = build_neck(backbone_output)
    
    # YOLOv5 Detection Head
    yolo_out = yolo_head(neck_output, num_classes, num_anchors)
    
    # Create the YOLOv5 model
    model = Model(inputs=inputs, outputs=yolo_out)
    
    return model

def yolo_loss(num_classes, num_anchors):
    def loss(y_true, y_pred):
        # Reshape predictions
        grid_size = tf.shape(y_pred)[1]
        y_pred = tf.reshape(y_pred, (-1, grid_size, grid_size, num_anchors, num_classes + 5))
        
        # Extract ground truth and predictions
        y_true_boxes = y_true[..., :4]
        y_true_conf = y_true[..., 4]
        y_true_cls = y_true[..., 5:]

        y_pred_boxes = y_pred[..., :4]
        y_pred_conf = y_pred[..., 4:5]
        y_pred_cls = y_pred[..., 5:]

        # Compute losses
        box_loss = tf.reduce_sum(tf.square(y_true_boxes - y_pred_boxes), axis=-1)
        conf_loss = tf.reduce_sum(tf.square(y_true_conf - y_pred_conf), axis=-1)
        class_loss = tf.reduce_sum(tf.square(y_true_cls - y_pred_cls), axis=-1)

        return box_loss + conf_loss + class_loss

    return loss


