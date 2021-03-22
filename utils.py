"""
Вспомогательные функции
"""


import numpy as np
import tensorflow as tf


def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,)
    
def convert_to_xywh2(box):
    return tf.concat(
        [(box[:2] + box[2:]) / 2.0, box[2:] - box[:2]],
        axis=-1,)

def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,)

def decode_box_predictions(anchor_boxes, box_predictions):
    boxes = box_predictions
    boxes = tf.concat(
        [
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    return boxes_transformed

def get_boxes_cls(y_pred):
    anchor_boxes = create_prior_boxes2()
    box_predictions = y_pred[:, :, :4]
    cls_predictions = tf.nn.softmax(y_pred[:, :, 4:])
    boxes = decode_box_predictions(anchor_boxes[None, ...], box_predictions)
    return boxes, cls_predictions

def compute_iou(boxes1, boxes2):
    """Вычисляет попарно IOU для двух тензоров

    Arguments:
      boxes1: Тензор размером `(N, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`.
      boxes2: Тензор размером `(M, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`

    Returns:
      Попарную IOU матрицу размером`(N, M)`, где значение i-ой строки
        j-ого столбца IOU между iым боксом and jым боксом из
        boxes1 and boxes2 соответственно.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = boxes2
    lu = tf.maximum(boxes1_corners[:, :2], boxes2_corners[:2])
    rd = tf.minimum(boxes1_corners[:, 2:], boxes2_corners[2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box1_area = boxes1[:, 2] * boxes1[:, 3]
    box2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    union_area = np.maximum(box1_area + box2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def compute_iou_metric(boxes, gt_true):
    """
    Считает IOU между gt_true И предсказанными боксами
    """
    lu = np.maximum(boxes[:2], gt_true[:2])
    rd = np.minimum(boxes[2:], gt_true[2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[ 0] * intersection[1]
    box1_area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    box2_area = (gt_true[2] - gt_true[0]) * (gt_true[3] - gt_true[1])
    union_area = np.maximum(box1_area + box2_area - intersection_area, 1e-8)
    return np.clip(intersection_area / union_area, 0.0, 1.0)

def compute_iou1(boxes1, boxes2):
    """Вычисляет попарно IOU для двух тензоров

    Arguments:
      boxes1: Тензор размером `(N, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`.
      boxes2: Тензор размером `(M, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`

    Returns:
      Попарную IOU матрицу размером`(N, M)`, где значение i-ой строки
        j-ого столбца IOU между iым боксом and jым боксом из
        boxes1 and boxes2 соответственно.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = boxes2
    lu = tf.maximum(boxes1_corners[:,None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    box1_area = boxes1[:, 2] * boxes1[:, 3]
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = np.maximum(box1_area[:, None] + box2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def compute_iou2(boxes1, boxes2):
    """Вычисляет попарно IOU для двух тензоров

    Arguments:
      boxes1: Тензор размером `(N, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`.
      boxes2: Тензор размером `(M, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`

    Returns:
      Попарную IOU матрицу размером`(N, M)`, где значение i-ой строки
        j-ого столбца IOU между iым боксом and jым боксом из
        boxes1 and boxes2 соответственно.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = boxes2
    lu = tf.maximum(boxes1_corners[:,None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    box1_area = boxes1[:, 2] * boxes1[:, 3]
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = np.maximum(box1_area[:, None] + box2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)