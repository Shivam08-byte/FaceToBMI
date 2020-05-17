
import cv2
import sys
import dlib
import numpy as np
from contextlib import contextmanager
from model.model import get_model
from config import cfg
import torch
from data.data import data_transforms
from PIL import Image


def get_trained_model():
    weights_file = cfg.best_trained_model_file
    model = get_model()
    model.load_state_dict(torch.load(cfg.best_trained_model_file))
    return model


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y-size[1]),
                  (x+size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images_from_camera():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture image")
            yield img


def run_demo():
    args = sys.argv[1:]
    multiple_targets = '--multiple' in args
    single_or_multiple = 'multiple faces' if multiple_targets else 'single face'
    model = get_trained_model()
    print(f"Loading model to detect BMI of {single_or_multiple}")

    last_seen_bmis = []
    detector = dlib.get_frontal_face_detector()

    for img in yield_images_from_camera():
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), cfg.default_img_width,
                          cfg.default_img_width, 3))
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                    1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - cfg.margin * w), 0)
                yw1 = max(int(y1 - cfg.margin * h), 0)
                xw2 = min(int(x2 + cfg.margin * w), img_w - 1)
                yw2 = min(int(y2 + cfg.margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(
                    img[yw1:yw2 + 1, xw1:xw2 + 1, :], (cfg.default_img_width, cfg.default_img_width))

            faces = faces.astype(np.uint8).reshape(224, 224, 3)
            faces = Image.fromarray(faces)
            faces = data_transforms['test'](faces)
            faces = faces.unsqueeze(0)
            predictions = model(faces).tolist()

            if multiple_targets:
                for i, d in enumerate(detected):
                    label = str(predictions[i][0])
                    draw_label(img, (d.left(), d.top()), label)
            else:
                last_seen_bmis.append(predictions[0][0])
                if len(last_seen_bmis) > cfg.num_of_frames:
                    last_seen_bmis.pop(0)
                elif len(last_seen_bmis) < cfg.num_of_frames:
                    continue
                avg_bmi = sum(last_seen_bmis) / float(cfg.num_of_frames)
                label = "BMI:" + "{: .2f}".format(avg_bmi)
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow('result', img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    run_demo()
