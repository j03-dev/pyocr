import cv2
import numpy as np
import os
import tensorflow as tf

from .tools.pretraitement import pretraitement
from .tools.segmentation import segmentation


class Pyocr:
    def __init__(self, model_path: str, classes_path: str) -> None:
        self.DIRECTORY = os.path.abspath(".")
        self.MODEL = tf.keras.models.load_model(
            os.path.join(self.DIRECTORY, model_path))

        with open(classes_path, 'r') as classes_file:
            self.CLASSES = [classe.strip("\n")
                            for classe in classes_file.readlines()]

    def predict_classes(self, x: np.ndarray) -> str:
        predict_x = self.MODEL.predict(x)
        classes_x = np.argmax(predict_x, axis=1)[0]
        return self.CLASSES[classes_x]

    def print_rect(self, list_chars: list, brect_list: list, path_image: str) -> np.ndarray:
        image = cv2.imread(path_image)
        for char, brect in zip(list_chars, brect_list):
            x, y, w, h = brect
            classe = self.predict_classes(char)
            cv2.rectangle(image, brect, (0, 255, 0), 2)
            cv2.putText(image, str(classe), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 123, 68), 2)
        return image

    def sort_chars_brect(self, prediction_list_and_brect: tuple) -> list:
        list_sorted = []
        backup = []
        current_y = prediction_list_and_brect[0][2]

        # divise la list
        for i in range(len(prediction_list_and_brect)):
            char, x, y, w, h = prediction_list_and_brect[i]
            if abs(current_y - y) > 5:
                list_sorted.append(backup)
                backup = []  # reset backup list
                current_y = y  # set new line

            backup.append((char, x, y, w, h))
        list_sorted.append(backup)

        # sort list_sorted en fonction de x
        for list_ in list_sorted:
            list_.sort(key=lambda x: x[1])
        return list_sorted

    def make_list_chars_reable(self, list_chars: list, brect_list) -> str:
        prediction_list_and_brect = []
        for chars, brect in zip(list_chars, brect_list):
            x, y, w, h = brect
            ch = self.predict_classes(chars)
            prediction_list_and_brect.append(
                (ch, x, y, w, h))

        prediction_list_and_brect = self.sort_chars_brect(
            prediction_list_and_brect)

        r = ""
        for line in prediction_list_and_brect:
            r = r + "".join(ch for ch, x, y, w, h in line) + " "

        return r
    
    def reconnaissance_and_print_rect(self, path_image: str) -> None:
        image = cv2.imread(path_image)
        image_d, image = pretraitement(image)
        list_chars, brect = segmentation(image, image_d)
        self.print_rect(list_chars, brect)

    def reconnaissance(self, path_image: str) -> str:
        image = cv2.imread(path_image)
        image_d, image = pretraitement(image)
        list_chars, brect = segmentation(image, image_d)
        return self.make_list_chars_reable(list_chars, brect)
