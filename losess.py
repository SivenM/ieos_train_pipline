import tensorflow as tf


class SSDLoss(tf.losses.Loss):

    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.num_boxes = 1
        self.alpha = alpha
        self._box_loss = tf.keras.losses.Huber() #SSDBoxLoss()
        self._cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = y_true[:, :, 4:]
        cls_predictions = y_pred[:, :, 4:]
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_labels, logits=cls_predictions
        )
        box_loss = self._box_loss(box_labels, box_predictions)
        loss = (cls_loss + self.alpha * box_loss) / self.num_boxes
        return loss