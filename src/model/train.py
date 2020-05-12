import cv2
import torch
import numpy as np
from config import cfg
from torch import nn, optim
from data.data import train_val_test_split
from model.model import get_model

train_on_gpu = torch.cuda.is_available()


def get_criterion():
    # Setting criterion and loggin
    criterion = nn.SmoothL1Loss()
    if train_on_gpu:
        criterion.cuda()
    return criterion


def get_optimizer(epoch, model):
    learning_rate = cfg.base_learning_rate*(1+cfg.gamma*epoch)**(-cfg.power)
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate,
                          weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    return optimizer


def get_test_criterion():
    test_criterion = nn.L1Loss()
    if train_on_gpu:
        test_criterion = test_criterion.cuda()
    return test_criterion


def train_model(train_loader, valid_loader):
    print('Training top layer only')

    model = get_model()

    if train_on_gpu:
        model = model.cuda()

    criterion = get_criterion()

    valid_loss_min = np.Inf
    train_losses, valid_losses = [], []

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
                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        images, bmi = images.cuda(), bmi.cuda()
                    # Run model
                    predictions = model(images)
                    loss = criterion(predictions, bmi)
                    valid_loss += loss.item()*images.size(0)

            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            # print(
            #     "Epoch: {}/{}..".format(epoch+1, epochs),
            #     "Training loss: {:.3f}".format(train_loss),
            #     "Validating loss: {:.3f}".format(valid_loss)
            # )
            if valid_loss <= valid_loss_min:  # save model if validation loss has decreased
                print('Epoch: {}/{}.. Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    epoch+1, cfg.epochs, valid_loss_min, valid_loss))
                torch.save(model.state_dict(),
                           '../models/trained/best_model.pt')  # for testing
                valid_loss_min = valid_loss


def test_model(test_loader, plot_sample=False):
    model.load_state_dict(torch.load('../models/trained/best_model.pt'))
    test_loss = 0.0
    test_criterion = get_test_criterion()
    print("Testing...")
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

    if plot_sample:
        model.load_state_dict(torch.load('../models/trained/best_model.pt'))

        # obtain one batch of test images
        images, height, weight, bmi = next(iter(test_loader))

        # get sample predictions
        predictions = model(images)

        # prep images for display
        images = images.numpy()

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(25, cfg.batch_size))

        for idx in np.arange(cfg.batch_size):
            ax = fig.add_subplot(4, cfg.batch_size/4,
                                 idx+1, xticks=[], yticks=[])
            plt.imshow(np.transpose(images[idx, :], (1, 2, 0)))
            ax.set_title(
                "Predicted:{:.2f}/ Actual: {:.2f}".format(
                    predictions[idx, :].item(), bmi[idx, :].item()),
                color=("green" if predictions[idx, :].item() == bmi[idx, :].item() else "red"))
