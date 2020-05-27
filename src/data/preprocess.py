import cv2
from os.path import join, exists
from os import makedirs, listdir
from config import cfg
import dlib
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()


def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def crop_faces(plot_images=False, max_images_to_plot=5):
    bad_crop_count = 0
    cropped_annotation = join(cfg.test_data_path, "annotation.csv")
    if not exists(cfg.cropped_data_path):
        makedirs(cfg.cropped_data_path)
    good_cropped_images = []
    good_cropped_img_file_names = []
    detected_cropped_images = []
    original_images_detected = []
    for images_name in listdir(cfg.raw_test_data_path):
        try:
            np_img = cv2.imread(
                join(cfg.raw_test_data_path, images_name), cv2.IMREAD_UNCHANGED)
            assert np_img.ndim == 3
        except Exception as e:
            print(join(cfg.raw_test_data_path, images_name))
            print("Oops!", e.__class__, "occurred.")
        else:
            if np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]
            detected = detector(np_img, 1)
            img_h, img_w, _ = np.shape(np_img)
            original_images_detected.append(np_img)

            if len(detected) != 1:
                bad_crop_count += 1
                continue

            d = detected[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            # xw1 = int(x1 - cfg.margin * w)
            # yw1 = int(y1 - cfg.margin * h)
            # xw2 = int(x2 + cfg.margin * w)
            # yw2 = int(y2 + cfg.margin * h)
            # cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2)
            cropped_img = crop_image(np_img, x1, y1, x2, y2)
            cropped_path_for_file = join(cfg.cropped_data_path, images_name)
            cv2.imwrite(cropped_path_for_file, cropped_img)

            good_cropped_img_file_names.append(
                join(cfg.cropped_data_path, images_name))

    # save info of good cropped images
    with open(join(cfg.intermediate_data_path, 'combined_annotation.csv'), 'r') as f:
        info = f.read().splitlines()
        column_headers = info[0]
        all_imgs_info = info[1:]

    cropped_imgs_info = [l for l in all_imgs_info if l.split(
        ',')[0] in good_cropped_img_file_names]

    with open(cropped_annotation, 'w+') as f:
        f.write('%s\n' % column_headers)
        for l in cropped_imgs_info:
            f.write('%s\n' % l)

    print(f"Cropped {len(original_images_detected)} images and saved in {cfg.cropped_data_path} - info in {cropped_annotation}")
    print(f"Error detecting face in {bad_crop_count} images")

    if plot_images:
        print('Plotting images...')
        img_index = 0
        plot_index = 1
        plot_n_cols = 3
        plot_n_rows = len(original_images_detected) if len(
            original_images_detected) < max_images_to_plot else max_images_to_plot
        for row in range(plot_n_rows):
            plt.subplot(plot_n_rows, plot_n_cols, plot_index)
            plt.imshow(original_images_detected[img_index].astype('uint8'))
            plot_index += 1

            plt.subplot(plot_n_rows, plot_n_cols, plot_index)
            plt.imshow(detected_cropped_images[img_index])
            plot_index += 1

            plt.subplot(plot_n_rows, plot_n_cols, plot_index)
            plt.imshow(good_cropped_images[img_index])
            plot_index += 1

            img_index += 1
    plt.show()
    return good_cropped_images
