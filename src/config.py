from os.path import join
from os import getcwd


class Config:
    # Set up Path
    src_path = getcwd()
    root = join("..", src_path)
    # Path for data
    data_path = join(root, "data")
    raw_data_path = join(data_path, "raw")
    intermediate_data_path = join(data_path, "interim")
    full_annotation_file = join(intermediate_data_path, "full_annotation.csv")
    female_annotation_file = join(
        intermediate_data_path, "female_annotation.csv")
    male_annotation_file = join(intermediate_data_path, "male_annotation.csv")
    image_path = join(data_path, "images")
    processed_path = join(data_path, "processed")
    total_data_processed_file = join(processed_path, "total_dataset.pt")

    # Path for model
    model_path = join(root, "models")
    pretrained_model_path = join(model_path, "pretrained")
    init_model_file = join(pretrained_model_path, "init_model.pt")

    # Setup variables
    batch_size = 16
    base_learning_rate = 1e-5
    epochs = 1  # should be 500 or higher
    weight_decay = 0.0005
    gamma = 0.001
    power = 0.75
    num_of_tries = 1  # should be 10 or more
    momentum = 0.9


cfg = Config()
