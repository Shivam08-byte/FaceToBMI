import torch
import numpy as np
from config import cfg
from torch import nn, optim
import matplotlib.pyplot as plt
from os.path import join

train_on_gpu = torch.cuda.is_available()
# train_on_gpu = False  # for laptop


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


def train_model(train_loader, valid_loader, model):
    criterion = get_criterion()

    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    valid_loss_min = np.Inf
    train_losses, valid_losses = [], []
    print("\tTraining models")
    for epoch in range(cfg.epochs):
        optimizer = get_optimizer(epoch, model)
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for images, _, _, bmi in train_loader:
            if train_on_gpu:
                images, bmi = images.cuda(), bmi.cuda()
            # print(images.is_cuda, bmi.is_cuda, next(model.parameters()).is_cuda) check if all varibale is on GPU
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, bmi)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        ######################
        # validate the model #
        ######################
        else:
            with torch.no_grad():
                model.eval()
                for images, _, _, bmi in valid_loader:
                    if train_on_gpu:
                        images, bmi = images.cuda(), bmi.cuda()
                    predictions = model(images)
                    loss = criterion(predictions, bmi)
                    valid_loss += loss.item()*images.size(0)

            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            # print(
            #     "Epoch: {}/{}..".format(epoch+1, cfg.epochs),
            #     "Training loss: {:.3f}".format(train_loss),
            #     "Validating loss: {:.3f}".format(valid_loss)
            # )
            if valid_loss <= valid_loss_min:
                print('\t\tEpoch: {}/{}\tValidation loss decreased ({:.6f} --> {:.6f})\tSaving model'.format(
                    epoch+1, cfg.epochs, valid_loss_min, valid_loss))
                torch.save(model.state_dict(), cfg.best_trained_model_file)
                valid_loss_min = valid_loss

    fig = plt.figure(figsize=(25, cfg.epochs))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    _ = plt.legend(frameon=False)
    figure_path = join(cfg.visualization_path, "train_valid.png")
    fig.savefig(figure_path)
    plt.close(fig)


def test_model(test_loader, model, plot_sample=True):
    model.load_state_dict(torch.load('../models/trained/best_model.pt'))
    test_criterion = get_test_criterion()

    if train_on_gpu:
        test_criterion = test_criterion.cuda()
        model = model.cuda()

    test_loss = 0.0

    print("\tTesting model...")
    with torch.no_grad():
        model.eval()
        for images, _, _, bmi in test_loader:
            if train_on_gpu:
                images, bmi = images.cuda(), bmi.cuda()
            predictions = model(images)
            loss = test_criterion(predictions, bmi)
            test_loss += loss.item()*images.size(0)

    # average test loss
    test_loss = test_loss/len(test_loader.sampler)
    print(f"\tTesting loss: {test_loss:.3f}")
    return test_loss

    if plot_sample:
        images, height, weight, bmi = next(iter(test_loader))
        predictions = model(images)
        images = images.numpy()
        fig = plt.figure(figsize=(25, cfg.batch_size))
        for idx in np.arange(cfg.batch_size):
            ax = fig.add_subplot(4, cfg.batch_size/4,
                                 idx+1, xticks=[], yticks=[])
            plt.imshow(np.transpose(images[idx, :], (1, 2, 0)))
            ax.set_title(
                "Predicted:{:.2f}/ Actual: {:.2f}".format(
                    predictions[idx, :].item(), bmi[idx, :].item()),
                color=("green" if predictions[idx, :].item() == bmi[idx, :].item() else "red"))
        figure_path = join(cfg.visualization_path, "test_sample.png")
        fig.savefig(figure_path)
        plt.close(fig)


def plot_sample(data_loader, model):
    model.load_state_dict(torch.load('../models/trained/best_model.pt'))

    images, height, weight, bmi = next(iter(data_loader))
    predictions = model(images)
    images = images.numpy()
    fig = plt.figure(figsize=(25, cfg.batch_size))
    for idx in np.arange(cfg.batch_size):
        ax = fig.add_subplot(4, cfg.batch_size/4, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx, :], (1, 2, 0)))
        ax.set_title("Predicted:{:.2f}/ Actual: {:.2f}".format(predictions[idx, :].item(), bmi[idx, :].item(
        )), color=("green" if predictions[idx, :].item() == bmi[idx, :].item() else "red"))
