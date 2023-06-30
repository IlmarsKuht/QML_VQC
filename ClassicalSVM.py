import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from data import load_datasets, normalize_dataset
from sklearn.decomposition import PCA
import numpy as np
import time
import random
import os

class Options:
    def __init__(self, num_classes_=None, feature_size_=None, batch_size_=None, max_iterations_=None, 
                 margin_=None, lr_=None, test_split_=None, qubits_=None, FM_layers_=None, ansatz_layers_=None,
                 feature_map_=None, ansatz_=None, loss_=None, dataset_=None, optimizer_=None,
                 params_=None, param_dim_=None, lr_factor_=None, patience_=None, stop_train_=None, ansatz_gates_=None,
                 fmap_gates_=None, iterations_=None, trial_=None, q_time_=None, c_time_=None, t_time_=None,
                 highest_acc_=None, last_acc_=None):
        self.num_classes = num_classes_
        self.feature_size = feature_size_
        self.batch_size = batch_size_
        self.max_iterations = max_iterations_
        self.margin = margin_
        self.lr = lr_
        self.test_split = test_split_
        self.qubits = qubits_
        self.FM_layers = FM_layers_
        self.ansatz_layers = ansatz_layers_
        self.feature_map = feature_map_
        self.ansatz = ansatz_
        self.loss = loss_
        self.dataset = dataset_
        self.optimizer = optimizer_
        self.params = params_
        self.param_dim = param_dim_
        self.lr_factor = lr_factor_
        self.patience = patience_
        self.stop_train = stop_train_
        self.ansatz_gates = ansatz_gates_
        self.fmap_gates = fmap_gates_
        self.iterations = iterations_
        self.trial = trial_
        self.q_time = q_time_
        self.c_time = c_time_
        self.t_time = t_time_
        self.highest_acc = highest_acc_
        self.last_acc = last_acc_


class SVM(nn.Module):
    def __init__(self, input_size):
        super(SVM, self).__init__()  
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out


def SVM_Prediction(features, labels, options):
    start_time = time.time()
    costs, test_acc, train_acc, output_list = [], [], [], []
    models = []
    
    for i in range(len(np.unique(labels))):
        models.append(SVM(len(features[0])))
        models[i].float()

    optimizer = AdamW(params=[{'params': model.parameters()} for model in models],
                       lr=options.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=options.lr_factor, patience=options.patience, verbose=True)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=options.test_split)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    for it in range(options.max_iterations):
        batch_index = np.random.randint(0, len(x_train), options.batch_size)
        x_train_batch = x_train[batch_index]
        y_train_batch = y_train[batch_index]

        optimizer.zero_grad()
        curr_cost = multiclass_svm_loss(models, x_train_batch, y_train_batch, options.margin)
        curr_cost.backward()
        optimizer.step()
        scheduler.step(curr_cost)

        if it % 10 == 0:

            train_idx = random.sample(range(len(x_train)), min(100, len(x_train)))
            test_idx = random.sample(range(len(x_test)), min(100, len(x_test)))

            predictions_train = classify(models, x_train[train_idx])
            predictions_test = classify(models, x_test[test_idx])
            acc_train = accuracy(y_train[train_idx], predictions_train)
            acc_test = accuracy(y_test[test_idx], predictions_test)

            print("Trial:{} Classical | Iter: {} | Cost: {:0.4f} | LR: {:0.4f} \n"
                "".format(options.dataset, it, curr_cost.item(), optimizer.param_groups[0]['lr']))

            output_list.append("Trial:{} Classical | Iter: {} | Cost: {:0.4f} | LR: {:0.4f}  \n"
                "".format(options.dataset, it, curr_cost.item(), optimizer.param_groups[0]['lr']))

            costs.append(curr_cost.item())
            train_acc.append(acc_train)
            test_acc.append(acc_test)
        else:
            output_list.append(f"Cost: {curr_cost.item():.4f} | Iter: {it}")
            print(f"Trial: {options.dataset} Classical | Iter: {it} |  Cost: {curr_cost.item():.4f}")


    end_time = time.time()
    duration = end_time-start_time


    return costs, test_acc, train_acc, duration

def multiclass_svm_loss(models, x_batch, y_batch, margin):
    loss = 0
    num_samples = len(y_batch)
    for i, x in enumerate(x_batch):
        s_true = models[int(y_batch[i])](x)
        s_true = s_true.float()

        li = 0

        # Get the scores computed for this sample by the other classifiers
        for j in range(len(models)):
            if j != int(y_batch[i]):
                s_j = models[j](x)
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li

    return loss / num_samples


def classify(models, x_batch):
    predicted_labels = []
    for x in x_batch:
        scores = np.zeros(len(models))
        for c in range(len(models)):
            score = models[c](x)
            scores[c] = float(score)
        pred_class = np.argmax(scores)
        predicted_labels.append(pred_class)
    return predicted_labels


def accuracy(labels, hard_predictions):
    loss = 0
    for l, p in zip(labels, hard_predictions):
        if l==p:
            loss = loss + 1
    loss = loss / labels.shape[0]
    return loss

def save_metrics_to_file(path, file_name, costs):
    os.makedirs(path, exist_ok=True)
    file_name = f"{file_name}.txt"
    file_path = os.path.join(path, file_name)
    with open(file_path, "w") as file:
        file.write(" ".join(map(str, costs)))


def classicalSVM(options):
    directory = './datasets'
    datasets = load_datasets(directory)

    for dataset_tuple in datasets:
        dataset, dataset_name = dataset_tuple

        data = [sample[:-1] for sample in dataset]
        Y = [sample[-1] for sample in dataset]

        if options.reduction is not None and len(data[0]) > options.reduction:
            pca = PCA(n_components=options.reduction)
            data = pca.fit_transform(data)

        features = normalize_dataset(data)
        options.num_classes = len(np.unique(Y))
        options.feature_size = len(features[0])
        options.dataset = dataset_name

        costs, test_acc, train_acc, duration = SVM_Prediction(features, Y, options)

        plot_folders = [("costs", costs), ("test_acc", test_acc), ("train_acc", train_acc)]

        for x in plot_folders:
            save_metrics_to_file(f"./Trials/{dataset_name}/{x[0]}", f'ClassicalSVM_time:_{duration}', x[1])