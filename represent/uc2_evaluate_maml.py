from represent.datamodules.uc2_landcover_datamodule import load_samples, CLASSNAMES, COARSE_CLASSES
from represent.datamodules.uc2_landcover_datamodule import UC2LandCoverDataModule, UC2LandCoverDataset, CLASSES, sample_task
import pandas as pd
import numpy as np
from meteor import METEOR
from meteor import models
import torch
import os
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def split_support_query(ds, shots, at_least_n_queries=1, random_state=0, classcolumn="majorityclass"):
    classes, counts = np.unique(ds.index[classcolumn], return_counts=True)
    classes = classes[
        counts > 2]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    classes, counts = np.unique(ds.index[classcolumn], return_counts=True)

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index[classcolumn] == c].reset_index()

        N_samples = shots if len(samples) > shots else len(samples) - 1  # keep at least one sample for query
        support = samples.sample(N_samples, random_state=random_state, replace=False)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries

def get_data(shots, keep_classes, limit_test_samples=None):
    train_ds = UC2LandCoverDataset(root="/data/RepreSent/UC2", partition="Train",
                                   segmentation=False, keep_classes=keep_classes, simplified_classes=False,
                                   image_size=610)

    supports, queries = split_support_query(train_ds, shots)
    X_train, y_train = load_samples(train_ds, supports)
    y_train = keep_classes[y_train]

    test_ds = UC2LandCoverDataset(root="/data/RepreSent/UC2", partition="Test", image_size=610,
                                  segmentation=False, keep_classes=keep_classes, simplified_classes=False)

    test_ds_index = test_ds.index.reset_index()

    if limit_test_samples is not None:
        test_ds_index = test_ds_index.sample(limit_test_samples)

    # test set from the test area
    X_test, y_test = load_samples(test_ds, test_ds_index)

    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred):
    present_classes = np.unique(y_true)
    present_classnames = [CLASSNAMES[c] for c in present_classes]

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true", labels=present_classes)
    cm_all = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None, labels=present_classes)

    fig, ax = plt.subplots(figsize=(10, 10))

    sc = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid("off")
    ax.set_xticks(np.arange(len(present_classes)))
    ax.set_yticks(np.arange(len(present_classes)))

    ax.set_xticklabels(present_classnames, rotation="30", ha="right")
    ax.set_yticklabels(present_classnames)

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            v = f"{cm_all[c, r]}"
            v = f"{v[:-3]}k" if len(v) > 3 else v
            color = "black" if cm[c, r] < 0.5 else "white"
            ax.text(r, c, v, ha="center", va="center", color=color)

    ax.grid(False)

    ax.set_xlabel("predicted", labelpad=-600)
    ax.set_ylabel("true", loc="center", labelpad=-690)

    cbar = fig.colorbar(sc, ax=ax, location='right', anchor=(0, 0.3), shrink=0.9)

    cbar.ax.set_title('recall')

    return fig

from math import log

def H(n):
    """Returns an approximate value of n-th harmonic number.

       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def num_draws(n, k=0):
    return n*H(n-k)

def calculate_gradient_steps(N, batchsize, captured_percentile=0.9):
    k = N-int(N*captured_percentile)
    draws = num_draws(N,k)
    return int(draws/batchsize)


def main():
    from tqdm.auto import tqdm
    keep_classes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 23, 24])

    test_ds = UC2LandCoverDataset(root="/data/RepreSent/UC2", partition="Test",
                                  segmentation=False, keep_classes=keep_classes, simplified_classes=False,
                                  image_size=610)

    for shots in [500, 1000, 5000]:  #
        print(shots)

        X_train, y_train, _, _ = get_data(shots, keep_classes, limit_test_samples=1)

        classes, counts = np.unique(y_train, return_counts=True)

        subset_bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10",
                        "S2B11", "S2B12"]

        # initialize an RGB model
        basemodel = models.get_model("maml_resnet12", subset_bands=subset_bands)

        # model_state_dict = torch.load("../weights/uc2_maml_resnet12.pth")
        # basemodel.load_state_dict(model_state_dict)

        # basemodel = model
        
        batch_size = 256
        
        # scale gradient steps with training size
        gradient_steps = calculate_gradient_steps(X_train.shape[0], batch_size, captured_percentile=0.8)
        # use at least 200 gradient steps
        gradient_steps = np.max([gradient_steps, 200])
        print(gradient_steps)
        
        taskmodel = METEOR(basemodel, verbose=False, device="cuda", gradient_steps=gradient_steps, mode="one_vs_one", batch_size=batch_size)

        output_folder = f"/data/RepreSent/UC2/results/{shots}-shots-{gradient_steps}-gradientsteps"
        os.makedirs(output_folder, exist_ok=True)
        weights_path = os.path.join(output_folder, f"uc2_maml_finetuned_shots{shots}_{gradient_steps}.pth")
        
        if not os.path.exists(weights_path):
            taskmodel.fit(X_train, y_train)
            torch.save({"params": taskmodel.params, "labels": taskmodel.labels}, weights_path)
        else:
            taskmodel.params = torch.load(weights_path)["params"]
            taskmodel.labels = torch.load(weights_path)["labels"]

        # evaluate
        indices = np.random.RandomState(0).randint(len(test_ds), size=1024)
        test_ds_ = torch.utils.data.Subset(test_ds, indices)

        dl = torch.utils.data.DataLoader(test_ds, batch_size=1000)
        taskmodel.verbose = False
        y_scores, y_trues = [], []

        for X_test, y_true in tqdm(dl, total=len(dl)):
            y_pred, y_score = taskmodel.predict(X_test)
            y_scores.append(y_score.cpu().numpy())
            y_trues.append(keep_classes[y_true.cpu().numpy()])

        y_test = np.hstack(y_trues)
        y_score = np.hstack(y_scores).T
        y_pred_index = y_score.argmax(1)
        y_pred = taskmodel.labels[y_pred_index]

        with open(os.path.join(output_folder, "results.txt"), "w") as f:
            print(classification_report(y_true=y_test, y_pred=y_pred, labels=keep_classes,
                                        target_names=[CLASSNAMES[int(c)] for c in keep_classes],
                                        zero_division=0), file=f)

        print(classification_report(y_true=y_test, y_pred=y_pred, labels=keep_classes,
                                    target_names=[CLASSNAMES[int(c)] for c in keep_classes],
                                    zero_division=0))

        # print(top_k_accuracy_score(y_test, y_score.T, labels=keep_classes, k=3))

        fig = plot_confusion_matrix(y_test, y_pred)
        fig.savefig(os.path.join(output_folder, "confusion.png"), bbox_inches="tight")

        torch.save({"y_pred": y_pred,
                    "y_score": y_score,
                    "y_test": y_test}, os.path.join(output_folder, "predictions.pt"))

if __name__ == '__main__':
    main()
