import cv2
import numpy as np
import os
import tensorflow as tf

from .tools.pretraitement import pretraitement
from .tools.segmentation import segmentation


class Ocr:
    def __init__(self, model_path: str, classes_path: str) -> None:
        """init model and load model and classes on started"""
        self._DIRECTORY = os.path.abspath(".")

        model_path = os.path.join(self._DIRECTORY, model_path)
        if os.path.exists(model_path):
            self._MODEL = tf.keras.models.load_model(model_path)
        else:
            raise Exception(f"this {model_path=} does't exists")

        classes_path = os.path.join(self._DIRECTORY, classes_path)
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as classes_file:
                self._CLASSES = [classe.strip("\n")
                                for classe in classes_file.readlines()]
        else:
            raise Exception(f"This {classes_path=} does't exists\n directory not found")

    def set_model(self, model_path: str) -> None:
        """function to load new model or change model"""
        model_path = os.path.join(self._DIRECTORY, model_path)
        if os.path.exists(model_path):
            self._MODEL = tf.keras.models.load_model(model_path)
        else:
            raise Exception(f"this {model_path=} does't exists")

    def set_classes(self, classes_path: str) -> None:
        """function to change classes"""
        classes_path = os.path.join(self._DIRECTORY, classes_path)
        if os.path.exists(path):
            with open(classes_path, 'r') as classes_file:
                self._CLASSES = [classe.strip("\n")
                                for classe in classes_file.readlines()]
        else:
            raise Exception(f"This {classes_path=} does't exists\n directory not found")


    def predict_classes(self, x: np.ndarray) -> str:
        """function to predict the classs  on image with softmax value"""
        predict_x = self._MODEL.predict(x)
        classes_x = np.argmax(predict_x, axis=1)[0]
        return self._CLASSES[classes_x]

    def add_rect(self, list_chars: list, brect_list: list, image: np.ndarray) -> np.ndarray:
        """add rectangle countoure on image"""
        for char, brect in zip(list_chars, brect_list):
            x, y, w, h = brect
            classe = self.predict_classes(char)
            cv2.rectangle(image, brect, (0, 255, 0), 2)
            cv2.putText(image, str(classe), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 123, 68), 2)
        return image

    def sort_chars_brect(self, prediction_list_and_brect: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """sort brect list, and return brect list sorted"""
        list_sorted: list[tuple[str, int]] = []
        backup: tuple[str, int] = []
        current_y: int = prediction_list_and_brect[0][2]

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

    def make_readable(self, list_chars: list[str], brect_list: tuple[int]) -> str:
        """return string of predition sorted"""
        prediction_list_and_brect: list[tuple[str, int]] = []
        for chars, brect in zip(list_chars, brect_list):
            x, y, w, h = brect
            # use model for recognition classe, return prediction classes
            ch: str = self.predict_classes(chars)
            prediction_list_and_brect.append(
                (ch, x, y, w, h)
            )

        prediction_list_and_brect: list[tuple[str, int]] = self.sort_chars_brect(
            prediction_list_and_brect
        )

        r = ""
        for line in prediction_list_and_brect:
            r = r + "".join(ch for ch, x, y, w, h in line) + " "

        return r

    def reconnaissance_and_add_rect(self, path_image: str) -> None:
        """'reconnaissance  d'image' and show image with rectangle"""
        image = cv2.imread(path_image)
        image_d, image = pretraitement(image)
        list_chars, brect = segmentation(image, image_d)
        result: np.ndarray = self.add_rect(list_chars, brect, image) # return image with rect
        cv2.imshow("result", result)

    def reconnaissance(self, path_image: str) -> str:
        """'reconnaissance d'image' and return string readable"""
        image = cv2.imread(path_image)
        image = cv2.imread(path_image)
        image_d, image = pretraitement(image)
        list_chars, brect = segmentation(image, image_d)
        return self.make_readable(list_chars, brect)
