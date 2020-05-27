from model.model import get_model
from model.train import train_model, test_model
from data.data import train_val_test_split
from data.scrape import crawl_data
from visualization.visualize import statistical_plot
from data.preprocess import crop_faces

if __name__ == '__main__':
    target = "bmi"
    # crawl_data()
    # crop_faces()
    model = get_model()
    _, _, western_test_loader = train_val_test_split(type="west")
    # train_model(train_loader=train_loader, valid_loader=valid_loader, model=model, target=target)
    test_model(test_loader=western_test_loader,
               model=model, target=target, type="west")
    _, _, asian_test_loader = train_val_test_split(type="test")
    test_model(test_loader=asian_test_loader,
               model=model, target=target, type="west")
