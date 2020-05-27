from model.model import get_model
from model.train import train_model, test_model
from data.data import train_val_test_split
from data.scrape import crawl_data
from visualization.visualize import statistical_plot
from data.preprocess import crop_faces

if __name__ == '__main__':
    crawl_data()
    # train_loader, valid_loader, test_loader = train_val_test_split()
    # model = get_model()
    # target = "bmi"
    # train_model(train_loader=train_loader, valid_loader=valid_loader,
    #             model=model, target=target)
    # test_model(test_loader=test_loader, model=model, target=target)
    # data = [1.2, 1.3, 1.5, 1.9, 1.1, 1.1, 1.3, 1.8, 1.6, 1.9]
    # statistical_plot(data=data, target=target)
