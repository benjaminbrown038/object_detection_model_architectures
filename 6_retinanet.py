def build_backbone(input_shape=(256, 256, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    return base_model.input, x


def retina_net_head(x, num_classes, num_anchors):
    # Classification head
    cls_head = Conv2D(num_anchors * num_classes, (3, 3), padding='same', activation='sigmoid')(x)
    
    # Bounding box regression head
    bbox_head = Conv2D(num_anchors * 4, (3, 3), padding='same')(x)
    
    return cls_head, bbox_head

def retina_net_model(input_shape=(256, 256, 3), num_classes=80, num_anchors=9):
    inputs = Input(shape=input_shape)
    
    # Backbone
    backbone_input, backbone_output = build_backbone(input_shape)
    
    # RetinaNet Heads
    cls_head, bbox_head = retina_net_head(backbone_output, num_classes, num_anchors)
    
    # Create the model
    model = Model(inputs=backbone_input, outputs=[cls_head, bbox_head])
    
    return model

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        return K.sum(fl, axis=1)
    return focal_loss_fixed
