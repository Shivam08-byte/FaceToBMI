from config import cfg
from glob import glob
from os.path import join, exists
from os import makedirs
import pandas as pd
from preprocess import crop_faces
import urllib.request as request


def read_data():
    dfs = []
    files = glob(join(cfg.external_data_path, "*.csv"))
    for file in files:
        df = pd.read_csv(file)
        df.dropna(inplace=True)
        df.drop_duplicates(subset="image-src", keep=False, inplace=True)
        df["height"] = pd.to_numeric(df["height"], downcast="float")
        df["weight"] = pd.to_numeric(df["weight"], downcast="float")
        df.reset_index(drop=True, inplace=True)
        data = df.drop(
            [column for column in df.columns if column not in cfg.useful_columns], 1)
        dfs.append(data)

    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.drop_duplicates(subset="image-src", keep=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    frame.to_csv(join(cfg.intermediate_data_path,
                      "unclean_combined_annotation.csv"), index=False)
    _ = frame.info()
    return frame


def check_url(dataframe):
    dataframe['image-src'] = cfg.web+dataframe['image-src']
    index_of_bad_urls = []
    for index, url in enumerate(dataframe['image-src']):
        try:
            request.urlopen(url)
        except:
            print(f"{url} is not good!")
            index_of_bad_urls.append(index)
            print(index_of_bad_urls)
    # try:
    #     request.urlopen(dataframe['image-src'][408])
    # except:
    #     index_of_bad_urls.append(408)
    #     print(index_of_bad_urls)
    dataframe.drop(index_of_bad_urls, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.to_csv(join(cfg.intermediate_data_path,
                          "combined_annotation.csv"), index=False)
    _ = dataframe.info()
    return dataframe


def crawl_data_from_frame(dataframe=None):
    if not dataframe:
        filename = join(cfg.intermediate_data_path, "combined_annotation.csv")
        dataframe = pd.read_csv(filename)
    if not exists(cfg.raw_test_data_path):
        makedirs(cfg.raw_test_data_path)
    bmi = dataframe['weight'] / \
        ((dataframe['height']/100)*(dataframe['height']/100))
    dataframe['bmi'] = bmi
    height = dataframe['height']/100
    dataframe['height'] = height
    for index, url in enumerate(dataframe['image-src']):
        images_name = str(index).zfill(4) + ".jpg"
        raw_path_for_file = join(cfg.raw_test_data_path, images_name)
        cropped_path_for_file = join(cfg.cropped_data_path, images_name)
        request.urlretrieve(url, raw_path_for_file)
        dataframe.iloc[index, 2] = cropped_path_for_file

    cols = ['image-src', 'height', 'weight', 'bmi']
    dataframe = dataframe[cols]
    dataframe.rename(columns={'image-src': 'Path',
                              'bmi': "BMI"}, inplace=True)
    dataframe.to_csv(join(cfg.intermediate_data_path,
                          "combined_annotation.csv"), index=False)
    _ = dataframe.info()


def crawl_data():
    # dataframe = read_data()
    # clean_dataframe = check_url(dataframe=dataframe)
    # crawl_data_from_frame(dataframe=clean_dataframe)
    # crawl_data_from_frame()
    # crop_faces()


if __name__ == "__main__":
    crawl_data()
