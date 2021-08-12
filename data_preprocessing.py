import enum
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import utils


class AncorBoxCreator:
    """
    Генерирует анкербоксы
    """
    def __init__(self, fmap_dims={'det_2': 1}, obj_scales={'det_2': 0.9}, aspect_ratios={'det_2': [1.]}, img_size=64):
        self.img_size = img_size
        self.fmap_dims = fmap_dims             
        self.obj_scales = obj_scales 
        self.aspect_ratios = aspect_ratios
        self.fmaps = list(self.fmap_dims.keys())

    def create_boxes(self):
        prior_boxes = []    
        for k, fmap in enumerate(self.fmaps):
          for i in range(self.fmap_dims[fmap]):
            for j in range(self.fmap_dims[fmap]):
              cx = (j + 0.5) / self.fmap_dims[fmap]
              cy = (i + 0.5) / self.fmap_dims[fmap]  
              for ratio in self.aspect_ratios[fmap]:
                prior_boxes.append([cx, cy, self.obj_scales[fmap] * np.sqrt(ratio), self.obj_scales[fmap] / np.sqrt(ratio)])
        prior_boxes = tf.convert_to_tensor(prior_boxes) 
        return prior_boxes * self.img_size


class LabelEncoder:
    """Трансформирует аннотации в таргеты для обучения.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.
    Этот класс имеет операции для создания таргетов для пакета сэмплов, 
    которые состоят из входных изображений, ограничивающих рамок для 
    присутствующих объектов и их идентификаторов классов.

    Attributes:
      anchor_box: генератор анкер боксов.
      box_variance: Коэффициенты масштабирования, используемые для масштабирования
        целевых объектов ограничивающей рамки
    """

    def __init__(self, fmap_dims, obj_scales, aspect_ratios, img_size=64):
        self._anchor_box = AncorBoxCreator(fmap_dims, obj_scales, aspect_ratios, img_size).create_boxes()
        self.num_boxes = self._anchor_box.shape[0]

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.4, ignore_iou=0.4
    ):
        """Сопоставляет gt боксы с анкербоксами через IOU.

        1. вычисляет попарную IOU для M `anchor_boxes` and N `gt_boxes`
           и выдает `(M, N)` размером матрицу.
        2. gt боксы с максимальным IOU в каждой строке назначается анкер бокс при 
           при условии что IOU больше чем `match_iou`.
        3. Если максимальная IOU в строке меньше чем `ignore_iou`, анкер боксу 
           назначается фоновый класс
        4. Остальные блоки, которым не назначен класс игнорируются

        Arguments:
          anchor_boxes: Тензор размором `(total_anchors, 4)`
            представляет все анкербоксыrepresenting для данной формы входного 
            изображения, где каждый анкер бокс форматом `[x, y, width, height]`.
          gt_boxes: Ттензор размером `(num_objects, 4)` представляющие gt боксы,
           где формат бокса`[x, y, width, height]`.
          match_iou: Значение представляющее минимальный порог IOU для определения
           того, может ли gt бокс назначен анкер боксу
          ignore_iou: Значение представляющее максимальный порог IOU для определения
            анкер боксу класс фона

        Returns:
          matched_gt_idx: Индес найденного объекта
          positive_mask: маска анкер боксов, которым назначены gt боксы.
          ignore_mask: маска анкер боксов, которая игнорируется во времяобучения
        """
        iou_matrix = utils.compute_iou1(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(iou_matrix, match_iou)
        negative_mask = tf.less(iou_matrix, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Трансформирует gt боксы в таргеты для обучения"""
        xy = tf.math.divide_no_nan((matched_gt_boxes[:, :2] - anchor_boxes[:, :2]), anchor_boxes[:, 2:])
        wh = tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])
        box_target = tf.concat(
            [
                xy,
                tf.where(tf.equal(wh, -np.inf), -100., wh),
            ],
            axis=-1,
        )
        #box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, gt_boxes, cls_ids):
        """Создает боксы и классифициет таргеты для одиночного сэмпла"""
        anchor_boxes = self._anchor_box
        gt_boxes = tf.cast(gt_boxes, dtype=tf.float32)
        gt_boxes = tf.reshape(gt_boxes, ((1,) + gt_boxes.shape))
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        gt_boxes = utils.convert_to_xywh(gt_boxes)
        box_target = self._compute_box_target(anchor_boxes, gt_boxes)
        cls_gt = tf.ones((self.num_boxes, 1), dtype=tf.float32)
        cls_bg = tf.cast(tf.equal(cls_gt, 0.), dtype=tf.float32)
        label = tf.concat([box_target, cls_gt], axis=1)
        label = tf.concat([label, cls_bg], axis=1)
        return label

    def encode_batch(self, gt_boxes, cls_ids):
        """Создает боксы и классифицирует таргеты для батча"""
        images_shape = tf.shape(gt_boxes)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        return labels.stack()

    def encode_bg(self, gt_boxes):
        images_shape = tf.shape(gt_boxes)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = tf.zeros([self.num_boxes, 4], dtype=tf.float32)
            human_cls = tf.zeros([self.num_boxes, 1], dtype=tf.float32)
            bg_cls = tf.ones([self.num_boxes, 1], dtype=tf.float32)
            label = tf.concat([label, human_cls], axis=-1)
            label = tf.concat([label, bg_cls], axis=-1)
            labels = labels.write(i, label)
        return labels.stack()


class DataLoader:
    """
    Создает датасет
    """
    def __init__(self, image_path, labels_path, num_data=None, label_encoder_data=None, img_size=(64, 64), label_encoder=False):
        self.image_path = image_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.image_name_list = self._get_image_names(num_data)
        self.labels_df = self._read_labels()

        if label_encoder:
            self.ln = True
            self.label_encoder = LabelEncoder(label_encoder_data['fmap_dims'], label_encoder_data['obj_scales'], label_encoder_data['aspect_ratios'], img_size[0])
        else:
            self.ln = False

    def _get_image_names(self, num_data):
        if num_data != None:
            return os.listdir(self.image_path)[:num_data]
        else:
            return os.listdir(self.image_path)

    def _read_labels(self):
        return pd.read_csv(self.labels_path)

    def _encode(self):
        x = np.zeros((len(self.image_name_list),) + self.img_size + (1,), dtype="float32")
        y = np.zeros((len(self.image_name_list),) + (4,), dtype="float32")
        names_df = self.labels_df['filename']
        for i, image_name in enumerate(self.image_name_list):
            image_path = self.image_path + '/' + image_name
            image = tf.keras.preprocessing.image.load_img(image_path,
                                                          color_mode = "grayscale",
                                                          target_size=self.img_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            index_bbox = names_df.index[names_df == image_name]
            bbox_coords = self.labels_df.iloc[index_bbox[0], 4:]
            bbox_coords = np.array([bbox_coords['xmin'],              
                           bbox_coords['ymin'],
                           bbox_coords['xmax'],
                           bbox_coords['ymax']] )
            x[i] = image
            y[i] = bbox_coords
        return x, y

    def create_human_data(self):
        X, Y = self._encode() 
        X_train = X / 255      
        human_cls = tf.ones((Y.shape[0], 1), dtype=tf.float32)
        if self.ln:
            Y_train = self.label_encoder.encode_batch(Y, human_cls)
            return X_train, Y_train, Y
        else:
            return X_train, Y, human_cls

    def create_bg_data(self):
        X, Y = self._encode() 
        X_train = X / 255
        if self.ln:
            Y_train = self.label_encoder.encode_bg(Y)
            return X_train, Y_train, Y
        else:
            bg_cls = tf.zeros((Y.shape[0], 1), dtype=tf.float32)
            return X_train, Y, bg_cls



class HumanDataset(tf.keras.utils.Sequence):

    def __init__(self, image_dir, labels_path, batch_size, img_size=(64, 64), normalise_bbox=True):
        self.img_dir = image_dir
        self.labels_df = self._read_labels(labels_path)
        self.batch_size = batch_size
        self.img_size = img_size
        self.imgs_data = os.listdir(self.img_dir)
        np.random.shuffle(self.imgs_data)
        self.names_df = self.labels_df['filename']
        self.normalise_bbox = normalise_bbox

    def _read_labels(self, labels_path):
        return pd.read_csv(labels_path)

    def __len__(self):
        return len(self.imgs_data) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_imgs_data = self.imgs_data[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y_box = np.zeros((self.batch_size,) + (4,), dtype="float32")
        y_cls = np.zeros((self.batch_size,) + (1,), dtype="float32")
        for j, img_name in enumerate(batch_imgs_data):
            path = os.path.join(self.img_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(path,
                                                        color_mode = "grayscale",
                                                        target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            index_bbox = self.names_df.index[self.names_df == img_name]
            bbox_coords = self.labels_df.iloc[index_bbox[0], 4:]
            bbox_coords = np.array([bbox_coords['xmin'],              
                           bbox_coords['ymin'],
                           bbox_coords['xmax'],
                           bbox_coords['ymax']], dtype="float32")
            if self.normalise_bbox:
                bbox_coords[0] /= self.img_size[1] 
                bbox_coords[1] /= self.img_size[0]
                bbox_coords[2] /= self.img_size[1]
                bbox_coords[3] /= self.img_size[0]
            x[j] = img
            y_box[j] = bbox_coords
            if self.labels_df.iloc[index_bbox[0], 3] == 'person':
                y_cls[j] = 1.
        return x, (y_box, y_cls)

class DatasetCreator:
    """
    Создает датасет для обучения в зависимости от соотношения
    """
    def __init__(self, human_data, bg_data, test_type=None):
        """
        human_data: кортеж имеет нобор изображений, таргетов и гт боксов человека
        bg_data: кортеж имеет нобор изображений, таргетов и гт боксов фона
        """
        self.test_type = test_type
        self.human_data = human_data
        self.bg_data = bg_data
        self.num_human_data = self.human_data[0].shape[0]

    def shuffle_data(self, dataset):
        seed = 42
        for data in dataset:
            if type(data) != np.ndarray:
                data = data.numpy()
            np.random.seed(seed)
            np.random.shuffle(data)
        return dataset

    def train_val(self, data):
        fold = 5
        k=0
        num_val_samples = data[0].shape[0] // fold
        img_val = data[0][k * num_val_samples: (k+1) * num_val_samples]
        target_val= data[1][k * num_val_samples: (k+1) * num_val_samples]
        gt_val = data[2][k * num_val_samples: (k+1) * num_val_samples]

        img_train = np.concatenate([data[0][:k * num_val_samples], 
                                        data[0][(k + 1) * num_val_samples:]],
                                        axis=0)
        target_train = np.concatenate([data[1][:k * num_val_samples], 
                                        data[1][(k + 1) * num_val_samples:]],
                                        axis=0)
        gt_train = np.concatenate([data[2][:k *num_val_samples], 
                                         data[2][(k + 1) * num_val_samples:]],
                                            axis=0)
        return (img_train, target_train, gt_train), (img_val, target_val, gt_val)

    def create_dataset(self, k_folds=True, val=False):
        if k_folds:
            return [self.human_data, self.bg_data]
        else:
            human_data_shuffled = self.shuffle_data(self.human_data)
            bg_data_shuffled = self.shuffle_data(self.bg_data)
            if val:
                train_human, val_human = self.train_val(human_data_shuffled)
                train_bg, val_bg = self.train_val(bg_data_shuffled)

                train_data = (
                    np.concatenate([train_human[0], train_bg[0]], axis=0),
                    np.concatenate([train_human[1], train_bg[1]], axis=0),
                    np.concatenate([train_human[2], train_bg[2]], axis=0),
                    )
                val_data = (
                    np.concatenate([val_human[0], val_bg[0]], axis=0),
                    np.concatenate([val_human[1], val_bg[1]], axis=0),
                    np.concatenate([val_human[2], val_bg[2]], axis=0),
                    )
                #train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size=5000, seed=42).batch(32).prefetch(1)
                #val_dataset = tf.data.Dataset.from_tensor_slices(val_data).shuffle(buffer_size=50, seed=42).batch(16).prefetch(1)
                return self.shuffle_data(train_data), (val_data)
            else:
                return (
                    np.concatenate([self.human_data[0], self.bg_data[0]], axis=0),
                    np.concatenate([self.human_data[1], self.bg_data[1]], axis=0),
                    np.concatenate([self.human_data[2], self.bg_data[2]], axis=0),
                    )