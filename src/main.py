from model.model import get_model
from model.train import train_model, test_model
from data.data import train_val_test_split
from data.scrape import crawl_data

if __name__ == '__main__':
    # train_loader, valid_loader, test_loader = train_val_test_split()
    # model = get_model()
    # train_model(train_loader=train_loader,
    #             valid_loader=valid_loader, model=model)
    # test_model(test_loader=test_loader, model=model)
    crawl_data()
