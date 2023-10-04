import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard
from keras.models import Sequential
from data_preparation import format_image, list_files, list_files_one
from tensor_format import data_load, tune_training_ds, tune_validation_ds, format_instance
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

train_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train'
validation_data_path = 'E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid'
full_data_path = 'E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/export'

DROPOUT_FACTOR = 0.5
input_size = 64
CLASSES = 11
EPOCHS = 100
BATCH_SIZE = 16


def build_feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_FACTOR)(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    return x


def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x


def build_classifier_head(inputs):
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name='classifier_head')(inputs)


def build_regressor_head(inputs):
    return tf.keras.layers.Dense(units='4', name='regressor_head')(inputs)


def build_model(inputs):
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'classifier_head': 'categorical_crossentropy', 'regressor_head': 'mse'},
                  metrics={'classifier_head': 'accuracy', 'regressor_head': 'mse'})

    return model


def get_training_data(one_path):

    if(not one_path):
        training_files = list_files(train_data_path)
        validation_files = list_files(validation_data_path)
        # test_files = list_files(test_data_path)

        raw_train_ds = data_load(training_files, train_data_path)
        raw_validation_ds = data_load(validation_files, validation_data_path)
        # raw_test_ds = data_load(test_files, test_data_path)

        train_ds = tune_training_ds(raw_train_ds)
        validation_ds = tune_validation_ds(raw_validation_ds, len(validation_files))

    else:
        training_files, validation_files, test_files = list_files_one(full_data_path)
        # validation_files = list_files_one(ful)
        # # test_files = list_files(test_data_path)

        raw_train_ds = data_load(training_files, full_data_path)
        raw_validation_ds = data_load(validation_files, full_data_path)
        # raw_test_ds = data_load(test_files, test_data_path)

        train_ds = tune_training_ds(raw_train_ds)
        validation_ds = tune_validation_ds(raw_validation_ds, len(validation_files))

    return train_ds, validation_ds, len(training_files)

def train_model():
    model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))
    model.summary()
    train_ds, validation_ds, training_length = get_training_data(True)
    # from keras.utils import plot_model
    #
    # plot_model(model, show_shapes=True, show_layer_names=True)

    # training_files = list_files(train_data_path)
    # validation_files = list_files(validation_data_path)
    # # test_files = list_files(test_data_path)
    #
    # raw_train_ds = data_load(training_files, train_data_path)
    # raw_validation_ds = data_load(validation_files, validation_data_path)
    # # raw_test_ds = data_load(test_files, test_data_path)
    #
    # train_ds = tune_training_ds(raw_train_ds)
    # validation_ds = tune_validation_ds(raw_validation_ds, len(validation_files))
    #
    # checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    #
    history = model.fit(train_ds,
                        steps_per_epoch=(training_length // BATCH_SIZE),
                        validation_data=validation_ds, validation_steps=1,
                        epochs=EPOCHS)

    model.save('saved_model/my_model2.h5')

# train_model()
