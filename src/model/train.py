import torch
import numpy as np
from config import cfg
from torch import nn, optim
import matplotlib.pyplot as plt
from os.path import join
from loss import MAELoss

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device('cpu')


def get_criterion():
    # Setting criterion and loggin
    criterion = nn.SmoothL1Loss()
    return criterion


def get_optimizer(epoch, model):
    learning_rate = cfg.base_learning_rate*(1+cfg.gamma*epoch)**(-cfg.power)
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate,
                          weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    return optimizer


def get_test_criterion():
    test_criterion = nn.L1Loss()
    return test_criterion


def train_model(train_loader, valid_loader, model, epochs, target, type):
    criterion = get_criterion()

    model = model.to(device)
    criterion = criterion.to(device)

    valid_loss_min = np.Inf
    train_losses, valid_losses = [], []
    print(f"\tTraining model for {epochs} epochs with target is {target}")
    for epoch in range(epochs):
        optimizer = get_optimizer(epoch, model)
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for images, height, weight, bmi in train_loader:
            images, bmi, height, weight = images.to(device), bmi.to(
                device), height.to(device), weight.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            if target == "bmi":
                loss = criterion(predictions, bmi)
            elif target == "height":
                loss = criterion(predictions, height)
            elif target == "weight":
                loss = criterion(predictions, weight)
            else:
                print("Unknown target")
                return
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        ######################
        # validate the model #
        ######################
        else:
            with torch.no_grad():
                model.eval()
                for images, height, weight, bmi in valid_loader:
                    images, bmi, height, weight = images.to(device), bmi.to(
                        device), height.to(device), weight.to(device)
                    predictions = model(images)
                    if target == "bmi":
                        loss = criterion(predictions, bmi)
                    elif target == "height":
                        loss = criterion(predictions, height)
                    elif target == "weight":
                        loss = criterion(predictions, weight)
                    else:
                        print("Unknown target")
                        return
                    valid_loss += loss.item()*images.size(0)

            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if valid_loss <= valid_loss_min:
                print('\t\tEpoch: {}/{}\tValidation loss decreased ({:.6f} --> {:.6f})\tSaving model'.format(
                    epoch+1, epochs, valid_loss_min, valid_loss))
                file_name = target + '_' + type + '_best_model.pt'
                file_address = join(cfg.trained_model_path, file_name)
                torch.save(model.state_dict(), file_address)
                valid_loss_min = valid_loss

    fig = plt.figure(figsize=(12, epochs))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    _ = plt.legend(frameon=False)
    file_name = target + "_train_valid.png"
    figure_path = join(cfg.visualization_path, file_name)
    fig.savefig(figure_path)
    plt.close(fig)


def test_model(test_loader, model, target, type, plot_sample=True):
    file_name = target + '_' + type + '_best_model.pt'
    file_address = join(cfg.trained_model_path, file_name)
    model.load_state_dict(torch.load(file_address, map_location=device))
    test_criterion = get_test_criterion()
    test_criterion = test_criterion.to(device)
    model = model.to(device)

    test_loss = 0.0

    print(f"\tTesting model with target {target}")
    with torch.no_grad():
        model.eval()
        for images, height, weight, bmi in test_loader:
            images, bmi, height, weight = images.to(device), bmi.to(
                device), height.to(device), weight.to(device)

            predictions = model(images)
            if target == "bmi":
                loss = test_criterion(predictions, bmi)
            elif target == "height":
                loss = test_criterion(predictions, height)
            elif target == "weight":
                loss = test_criterion(predictions, weight)
            else:
                print("Unknown target")
                return
            test_loss += loss.item()*images.size(0)

    # average test loss
    test_loss = test_loss/len(test_loader.sampler)
    print(f"\tTesting loss with {type} data is: {test_loss:.3f}")

    if plot_sample:
        print("\tExport sampling images\n")
        images, height, weight, bmi = next(iter(test_loader))
        predictions = model(images.to(device))
        images = images.numpy()
        fig = plt.figure(figsize=(20, cfg.batch_size))
        for idx in np.arange(cfg.batch_size):
            ax = fig.add_subplot(4, cfg.batch_size/4,
                                 idx+1, xticks=[], yticks=[])
            plt.imshow(np.transpose(images[idx, :], (1, 2, 0)))
            if target == "bmi":
                ax.set_title("Predicted:{:.2f}/ Actual: {:.2f}".format(predictions[idx, :].item(), bmi[idx, :].item(
                )), color=("green" if predictions[idx, :].item() == bmi[idx, :].item() else "red"))
            elif target == "height":
                ax.set_title("Predicted:{:.2f}/ Actual: {:.2f}".format(predictions[idx, :].item(), height[idx, :].item(
                )), color=("green" if predictions[idx, :].item() == height[idx, :].item() else "red"))
            elif target == "weight":
                ax.set_title("Predicted:{:.2f}/ Actual: {:.2f}".format(predictions[idx, :].item(), weight[idx, :].item(
                )), color=("green" if predictions[idx, :].item() == weight[idx, :].item() else "red"))
        file_name = target + "_" + type + "_test_sample.png"
        figure_path = join(cfg.visualization_path, file_name)
        fig.savefig(figure_path)
        plt.close(fig)

    return test_loss
