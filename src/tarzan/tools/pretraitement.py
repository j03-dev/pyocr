import cv2
import numpy as np


def pretraitement(image: np.ndarray) -> np.ndarray:
    # image auxiliarire representer en niveau de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # passer d´une image en niveau de gris en noire et blanc
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                           cv2.THRESH_BINARY_INV)[1]

    # suppression de bruit
    thresh_s = cv2.medianBlur(thresh, 3)

    kernel = np.ones((2, 2), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))

    # lissage
    image_dilate = cv2.dilate(thresh_s, kernel, iterations=2)

    # new kernel
    new_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im = cv2.filter2D(image, -1, new_kernel)

    return image_dilate, im
