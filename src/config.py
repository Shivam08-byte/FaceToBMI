from os.path import join, dirname
from os import getcwd


class Config:
    # Set up Path
    root = dirname(getcwd())
    # Path for data
    data_path = join(root, "data")
    src_path = join(root, "src")
    raw_data_path = join(data_path, "raw")
    output_path = join(root, "output")

    intermediate_data_path = join(data_path, "interim")
    full_annotation_file = join(intermediate_data_path, "full_annotation.csv")
    female_annotation_file = join(
        intermediate_data_path, "female_annotation.csv")
    male_annotation_file = join(intermediate_data_path, "male_annotation.csv")

    image_path = join(data_path, "images")

    processed_path = join(data_path, "processed")
    total_data_processed_file = join(processed_path, "total_dataset.pt")

    external_data_path = join(data_path, "external")
    test_data_path = join(data_path, "test_data")
    raw_test_data_path = join(intermediate_data_path, "images")
    cropped_data_path = join(test_data_path, "images")
    test_data_annotation_file = join(test_data_path, "annotation.csv")

    asia_data_path = join(data_path, "asia")
    asia_image_path = join(asia_data_path, "images")
    asia_annotation_file = join(asia_data_path, "annotation.csv")

    visualization_path = join(src_path, "visualization")

    # Path for model
    model_path = join(root, "models")
    pretrained_model_path = join(model_path, "pretrained")
    trained_model_path = join(model_path, "trained")

    # Setup variables
    batch_size = 16
    base_learning_rate = 1e-5
    epochs = 500  # should be 500 or higher
    weight_decay = 0.0005
    gamma = 0.001
    power = 0.75
    num_of_tries = 20  # should be 10 or more
    momentum = 0.9
    web = "https://wiki.d-addicts.com"
    useful_columns = ['height', 'weight', 'image-src']
    margin = 0.1


cfg = Config()
