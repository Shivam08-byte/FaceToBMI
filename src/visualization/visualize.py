import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import cfg
from os.path import join


def statistical_plot(data=[], target="bmi"):
    t = list(range(1, cfg.num_of_tries+1))
    df = pd.DataFrame(list(zip(t, data)),
                      columns=['Try-th', 'Test loss'])
    df['Test loss'].describe()
    print(df['Test loss'].describe())
    print("Export statistical graph.")

    plt.figure(figsize=(10, 10))
    sns.set_color_codes("pastel")
    ax = sns.distplot(df['Test loss'], bins=cfg.num_of_tries)
    fig = ax.get_figure()
    file_name = target + "_test_distribution.png"
    file_address = join(cfg.visualization_path, file_name)
    fig.savefig(file_address)

    plt.figure(figsize=(10, 3))
    sns.set_color_codes("pastel")
    ax = sns.boxplot(x="Test loss", data=df)
    fig = ax.get_figure()
    file_name = target + "_test_box_plot.png"
    file_address = join(cfg.visualization_path, file_name)
    fig.savefig(file_address)


def plot_sample(data_loader, model, target="bmi"):
    file_name = target + '_best_model.pt'
    file_address = join(cfg.trained_model_path, file_name)
    model.load_state_dict(torch.load(file_address))
    images, height, weight, bmi = next(iter(data_loader))
    predictions = model(images)
    images = images.numpy()
    fig = plt.figure(figsize=(12, cfg.batch_size))
    for idx in np.arange(cfg.batch_size):
        ax = fig.add_subplot(4, cfg.batch_size/4, idx+1, xticks=[], yticks=[])
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


if __name__ == "__main__":
    statistical_plot()
