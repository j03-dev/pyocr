import cv2
import numpy as np
from imutils.contours import sort_contours


def segmentation(image_rgb: np.ndarray, image_d: np.ndarray) -> tuple:
    result: list = []
    brect_list: list = []
    contours = cv2.findContours(
        image_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method='top-to-bottom')[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (5 <= w <= 150) and (15 <= h <= 120):
            print(w, h)
            roi = image_rgb[y:y + h, x:x + w]
            try:
                roi = cv2.resize(roi, (32, 32))
                roi = roi.reshape((-1, 32, 32, 3))
                result.append(roi)
                brect_list.append((x, y, w, h))
            except Exception as e:
                print(f"Exception => {e}")
                continue
    return result, brect_list
