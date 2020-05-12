from model.model import get_model
from model.train import train_model, test_model
from data.data import train_val_test_split


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = train_val_test_split()
    train_model(train_loader, valid_loader)
    test_model(test_loader)
