import os
import cv2
import dlib
import numpy as np
from wide_resnet import WideResNet
from contextlib import contextmanager
from keras.utils.data_utils import get_file


def main():

    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


    @contextmanager
    def video_capture(*args, **kwargs):
        cap = cv2.VideoCapture('Video.mp4')
        try:
            yield cap
        finally:
            cap.release()


    def yield_images():
        with video_capture(0) as cap:
            while True:
                ret, img = cap.read()

                if not ret:
                    raise RuntimeError("Failed to capture image")

                yield img


    WEIGHT = 'weights/weights.hdf5'
    DEPTH = 16
    WIDTH = 8
    MARGIN = 0.4

    if not WEIGHT:
        WEIGHT = get_file("weights.hdf5",
                               origin='https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5',
                               cache_subdir="./",
                               cache_dir=os.path.dirname(os.path.abspath(__file__)))

    detector = dlib.get_frontal_face_detector()

    img_size = 64
    model = WideResNet(img_size, depth=DEPTH, k=WIDTH)()
    model.load_weights(WEIGHT)

    for img in yield_images():
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - MARGIN * w), 0)
                yw1 = max(int(y1 - MARGIN * h), 0)
                xw2 = min(int(x2 + MARGIN * w), img_w)
                yw2 = min(int(y2 + MARGIN * h), img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(30)

        if key == 27:
            break

main()