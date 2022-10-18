#!/usr/bin/env python3
import pathlib

import tensorflow as tf
from tensorflow import keras


class OcrModel:
    def __init__(
        self,
        training_data: str,
        testing_data: str,
        batch_size: int = 3,
        image_width: int = 32,
        image_heigth: int = 32
    ) -> None:
        self.__training_data = pathlib.Path(training_data)
        self.__testing_data = pathlib.Path(testing_data)
        self.__batch_size = batch_size
        self.__img_w = image_width
        self.__img_h = image_heigth
        self.__model = None
        self.__training = tf.keras.preprocessing.image_dataset_from_directory(
            self.__training_data,
            validation_split=.3,
            subset="training",
            seed=36,
            image_size=(self.__img_h, self.__img_w),
            batch_size=self.__batch_size
        )

        self.__validation = tf.keras.preprocessing.image_dataset_from_directory(
            self.__testing_data,
            validation_split=.3,
            subset="validation",
            seed=36,
            image_size=(self.__img_h, self.__img_w),
            batch_size=self.__batch_size
        )

    def get_classes(self) -> list:
        return self.__training.class_names

    def save_classes(self, path: str) -> None:
        with open(path, 'w') as classes_file:
            for classes in self.get_classes():
                classes_file.write(f"{classes}\n")

    def build(self, optimizer: str = "adam") -> None:
        self.__model = tf.keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(1. / 255),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(16, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(36, activation='softmax')
        ])

        self.__model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def train_and_save(
        self,
        path: str,
        epochs: int = 3,
        save_format: str = "h5",
        log_dir: str = "training_logs"
    ) -> None:
        self.build()
        logdir = log_dir
        # write log
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1, write_images=logdir,
        )

        self.__model.fit(
            self.__training,
            validation_data=self.__validation,
            epochs=epochs,
            callbacks=[tensorboard_callback]
        )
        self.__model.summary()
        self.__model.save(path, save_format=save_format)
