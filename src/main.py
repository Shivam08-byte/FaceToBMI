from model.model import get_model
from model.train import train_model, test_model
from data.data import train_val_test_split
from data.scrape import crawl_data

if __name__ == '__main__':
    # crawl_data()
    train_loader, valid_loader, test_loader = train_val_test_split()
    model = get_model()
    target = "bmi"
    train_model(train_loader=train_loader, valid_loader=valid_loader,
                model=model, target=target)
    _, _ = test_model(test_loader=test_loader, model=model, target=target)
