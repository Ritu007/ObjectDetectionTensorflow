import os
from tensor_format import tune_test_ds, data_load
from data_preparation import list_files, format_image
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from base_models import build_model
import time


test_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/test'
new_data_path = 'E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/test'
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
input_size = 64

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


test_files = list_files(new_data_path)
raw_test_ds = data_load(test_files, new_data_path, input_size)
test_ds = tune_test_ds(raw_test_ds)


plt.figure(figsize=(12, 10))

test_list = list(test_ds.take(15).as_numpy_iterator())

# print((test_list))

image, labels = test_list[0]

# print(image[0], labels[0])

# model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))
# model = model.load_weights(checkpoint_path)

model = tf.keras.models.load_model('saved_model/my_model2.h5')
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

for i in range(len(test_list)):

    ax = plt.subplot(4, 5, i + 1)
    image, labels = test_list[i]

    predictions = model(image)

    predicted_box = predictions[1][0] * input_size
    predicted_box = tf.cast(predicted_box, tf.int32)

    predicted_label = predictions[0][0]

    image = image[0]

    actual_label = labels[0][0]
    actual_box = labels[1][0] * input_size
    actual_box = tf.cast(actual_box, tf.int32)

    image = image.astype("float") * 255.0
    image = image.astype(np.uint8)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    print("Predicted Label: ",predicted_label, "Predicted Box: ", predicted_box)

    print("Actual Label: ",actual_label, "Actual Box: ", actual_box.numpy())

    color = (255, 0, 0)
    # print box red if predicted and actual label do not match
    # if (predicted_label[0] > 0.5 and actual_label[0] > 0) or (predicted_label[0] < 0.5 and actual_label[0] == 0):
    #     color = (0, 255, 0)
    #
    # img_label = "dog"
    # if predicted_label[0] > 0.5:
    #     img_label = "cat"
    img_label = str(predicted_label)

    predicted_box_n = predicted_box.numpy()
    actual_box_n = actual_box.numpy()
    cv2.rectangle(image_color, predicted_box_n, color, 2)
    # cv2.rectangle(image_color, actual_box_n, (0, 0, 255), 2)
    # cv2.rectangle(image_color, (predicted_box_n[0], predicted_box_n[1] + predicted_box_n[3] - 20), (predicted_box_n[0] + predicted_box_n[2], predicted_box_n[1] + predicted_box_n[3]), color, -1)
    # cv2.putText(image_color, img_label, (predicted_box_n[0] + 5, predicted_box_n[1] + predicted_box_n[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())

    stretch_near = cv2.resize(image_color, (512, 512),
                              interpolation=cv2.INTER_LINEAR)

    # plt.title("IoU:" + format(IoU, '.4f'))
    cv2.imshow("prediction",stretch_near)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    time.sleep(5)


    # plt.axis("off")