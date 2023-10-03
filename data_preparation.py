import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt


# test_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/test'
#
# train_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train'
#
# validation_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid'

input_size = 64


def format_image(img, box, input_size = 64):
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = box[0], box[1], box[2], box[3]
    new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r), int(h * height / r)]

    return new_image, new_box


def list_files_one(full_data_path="data/obj/", image_ext='.jpg', split_percentage=[70, 20]):
    files = []

    discarded = 0
    # masked_instance = 0
    image_path = full_data_path + '/images'
    label_path = full_data_path + '/labels'
    for r, d, f in os.walk(full_data_path):
        for file in f:
            if file.endswith(".txt"):

                # first, let's check if there is only one object
                with open(label_path + "/" + file, 'r') as fp:
                    lines = fp.readlines()
                    if len(lines) < 1:
                        discarded += 1
                        continue

                strip = file[0:len(file) - len(".txt")]
                # secondly, check if the paired image actually exist
                image_file = image_path + "/" + strip + image_ext
                if os.path.isfile(image_file):
                    # checking the class. '0' means masked, '1' for unmasked
                    # if lines[0][0] == '0':
                    #     masked_instance += 1
                    files.append(strip)

    size = len(files)
    print(str(discarded) + " file(s) discarded")
    print(str(size) + " valid case(s)")
    # print(str(masked_instance) + " are masked cases")

    random.shuffle(files)

    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)

    return files[0:split_training], files[split_training:split_validation], files[split_validation:]


def list_files(data_path="data/obj/", image_ext='.jpg'):
    files = []

    discarded = 0
    masked_instance = 0
    cats = 0
    dogs = 0

    labels_path = data_path+"/labels"

    for r, d, f in os.walk(labels_path):
        # print(f)
        for file in f:
            # file = file
            if file.endswith(".txt"):

                # first, let's check if there is only one object
                with open(labels_path + "/" + file, 'r') as fp:
                    line = fp.readlines()[0].strip()
                    if len(line) < 1:
                        discarded += 1
                        continue

                strip = file[0:len(file) - len(".txt")]
                # secondly, check if the paired image actually exist
                image_path = data_path + "/images/" + strip + image_ext
                if os.path.isfile(image_path):
                    # checking the class. '0' means masked, '1' for unmasked
                    # if line[0] == '0':
                    #     dogs += 1
                    # else:
                    #     cats += 1

                    files.append(strip)

    size = len(files)
    print(str(discarded) + " file(s) discarded")
    print(str(size) + " valid case(s)")
    # print(str(cats) + " are cats")
    # print(str(dogs) + " are dogs")

    # random.shuffle(files)
    return files
    # split_training = int(split_percentage[0] * size / 100)
    # split_validation = split_training + int(split_percentage[1] * size / 100)
    #
    # return files[0:split_training], files[split_training:split_validation], files[split_validation:]


# train_files = list_files(train_data_path)
# print(train_files[0])
#
# validation_files = list_files(validation_data_path)
# print(validation_files[0])
#
# test_files = list_files(test_data_path)
# print(test_files[0])

# temp_image = cv2.imread(train_data_path + "/images/" + train_files[0] + ".jpg", cv2.IMREAD_GRAYSCALE)
# temp_box = []
# temp_label = 0
# with open(train_data_path + "/labels/" + train_files[0] + ".txt", 'r') as fp:
#     lines = fp.readlines()
#     print(lines[0])
#     new_list = lines[0].split(" ")
#     temp_label = new_list[0]
#     for i in range(1,5):
#         # print(list[i])
#         temp_box.append(float(new_list[i]))
#
#
# print(temp_box, temp_label)
# cv2.imshow("New Window", temp_image)
# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()


# temp_img_formated, box = format_image(temp_image, temp_box)
#
# temp_color_img = cv2.cvtColor(temp_img_formated, cv2.COLOR_GRAY2RGB)
#
# if temp_label == '0':
#     cv2.rectangle(temp_color_img, box, (0, 255, 0), 2)
#
# else:
#     cv2.rectangle(temp_color_img, box, (0, 0, 255), 2)
#
# plt.imshow(temp_color_img)
# plt.axis("off")
# plt.show()
