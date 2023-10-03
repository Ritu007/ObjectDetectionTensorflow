import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data_preparation import format_image, list_files

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging

# uncomment the following line if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.__version__)

# input_size = 224
#
# test_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/test'
#
# train_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train'
#
# validation_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid'


def data_load(files, data_path="data/obj/", input_size=224, image_ext=".jpg"):
    X = []
    Y = []

    image_path = data_path + "/images"
    label_path = data_path + "/labels"
    for file in files:
        img = cv2.imread(os.path.join(image_path, file + image_ext), cv2.IMREAD_GRAYSCALE)

        k = 0

        with open(label_path + "/" + file + ".txt", 'r') as fp:
            line = fp.readlines()[0].strip()
            print(line)
            values = line.split()
            k = int(values[0])
            print(k)

            box = np.array(values[1:], dtype=float)

        img, box = format_image(img, box)
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / input_size
        label = np.append(box, k)

        X.append(img)
        Y.append(label)

    X = np.array(X)

    X = np.expand_dims(X, axis=3)

    with tf.device("CPU"):

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result

# training_files = list_files(train_data_path)
# validation_files = list_files(validation_data_path)
# test_files = list_files(test_data_path)
#
# raw_train_ds = data_load(training_files, train_data_path)
# raw_validation_ds = data_load(validation_files, validation_data_path)
# raw_test_ds = data_load(test_files, test_data_path)

# CLASSES = 2


def format_instance(image, label, CLASSES = 11):
    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])


# BATCH_SIZE = 32

# see https://www.tensorflow.org/guide/data_performance

def tune_training_ds(dataset, BATCH_SIZE = 2):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    # dataset = dataset.repeat() # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# train_ds = tune_training_ds(raw_train_ds)
def tune_validation_ds(dataset, valid_length):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(valid_length // 4)
    # dataset = dataset.repeat()
    return dataset
# validation_ds = tune_validation_ds(raw_validation_ds)


def tune_test_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    # dataset = dataset.repeat()
    return dataset


# plt.figure(figsize=(20, 10))
# for images, labels in train_ds.take(1):
#     for i in range(BATCH_SIZE):
#         ax = plt.subplot(4, BATCH_SIZE//4, i + 1)
#         label = labels[0][i]
#         box = (labels[1][i] * input_size)
#         box = tf.cast(box, tf.int32)
#
#         image = images[i].numpy().astype("float") * 255.0
#         image = image.astype(np.uint8)
#         image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#
#         color = (0, 0, 255)
#         if label[0] > 0:
#             color = (0, 255, 0)
#
#         cv2.rectangle(image_color, box.numpy(), color, 2)
#
#         plt.imshow(image_color)
#         plt.axis("off")

