from model.model import get_model
from model.train import train_top_layer, train_all_layers
from data.data import train_val_test_split


if __name__ == '__main__':
    model = get_model()
    print(model)
    # train_top_layer(model)
    # train_all_layers(model)
    # train_loader, valid_loader, test_loader = train_val_test_split()
    # print("Train size", len(train_loader))
    # print("Validation size", len(valid_loader))
    # print("Test size", len(test_loader))
