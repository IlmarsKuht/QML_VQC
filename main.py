import pennylane as qml
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import multiprocessing
import time
import random
from sklearn.decomposition import PCA
from copy import copy
from data.data import load_datasets, normalize_dataset
from ClassicalSVM import classicalSVM

np.random.seed(0)
torch.manual_seed(0)


class Options:
    def __init__(self, num_classes_=None, feature_size_=None, batch_size_=None, max_iterations_=None, 
                 margin_=None, lr_=None, test_split_=None, qubits_=None, FM_layers_=None, ansatz_layers_=None,
                 feature_map_=None, ansatz_=None, loss_=None, dataset_=None, optimizer_=None,
                 params_=None, param_dim_=None, lr_factor_=None, patience_=None, stop_train_=None, ansatz_gates_=None,
                 fmap_gates_=None, iterations_=None, trial_=None, q_time_=None, c_time_=None, t_time_=None,
                 highest_acc_=None, last_acc_=None, reduction=None):
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
        self.reduction = reduction

def save_metrics_to_file(path, file_name, costs):
    os.makedirs(path, exist_ok=True)
    file_name = f"{file_name}.txt"
    file_path = os.path.join(path, file_name)
    with open(file_path, "w") as file:
        file.write(" ".join(map(str, costs)))

def create_plot_from_files(directory, name, ylim=None):
    file_names = os.listdir(directory)

    # Increase figure size and resolution
    plt.figure(figsize=(10, 6), dpi=350)

    for file_name in file_names:
        if not file_name.endswith(".txt"):
            continue  # Skip non-text files
        file_path = os.path.join(directory, file_name)
        combination = file_name.split(".")[0]
        with open(file_path, "r") as file:
            costs = list(map(float, file.read().strip().split()))
        feature_size = len(costs)
        plt.plot(range(feature_size), costs, label=combination)

    plt.title(f"{name}")
    plt.xlabel("Iterations in tens")
    plt.ylabel(f"{name}")
    plt.legend()

    if ylim is not None:
        plt.ylim(*ylim)

    # Save the plot in the same directory with the name "Cost Plot.png"
    save_path = os.path.join(directory, f"{name}.png")
    plt.savefig(save_path, dpi=350)
    plt.close()



def circuit(weights, feat, fmap, ansatz, qubits):
    if fmap.__name__ != "AmplitudeEmbedding":
        fmap(feat, range(qubits))
    else:
        fmap(feat, range(qubits), pad_with=0)
    ansatz(weights, wires=range(qubits))

    return qml.expval(qml.PauliZ(0))


def variational_classifier(q_circuit, params, feat, fmap, ansatz, qubits):
    weights = params[0]
    bias = params[1]
    return q_circuit(weights, feat, fmap, ansatz, qubits) + bias

def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels, fmap, ansatz, num_classes, margin, qubits):
    loss = 0
    num_samples = len(true_labels)
    for i, feature_vec in enumerate(feature_vecs):
        s_true = variational_classifier(
            q_circuits[int(true_labels[i])],
            (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
            feature_vec,
            fmap, ansatz, qubits
        )
        s_true = s_true.float()

        li = 0
        for j in range(num_classes):
            if j != int(true_labels[i]):
                s_j = variational_classifier(
                    q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec, fmap, ansatz, qubits
                )
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li

    return loss / num_samples

def classify(q_circuits, all_params, feature_vecs, fmap, ansatz, num_classes, qubits):
    predicted_labels = []
    for _, feature_vec in enumerate(feature_vecs):
        scores = np.zeros(num_classes)
        for c in range(num_classes):
            score = variational_classifier(
                q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec, fmap, ansatz, qubits
            )
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


def training(features, Y, optim, fmap, ansatz, options, qnodes):
    feat_vecs_train, feat_vecs_test, Y_train, Y_test = train_test_split(features, Y, test_size=options.test_split)
    feat_vecs_train = torch.tensor(feat_vecs_train)
    feat_vecs_test = torch.tensor(feat_vecs_test)
    Y_train = torch.tensor(Y_train)
    Y_test = torch.tensor(Y_test)
    num_train = Y_train.shape[0]
    q_circuits = qnodes
    output_list = []

    # Initialize the parameters
    all_weights = torch.stack([np.pi * torch.randn(options.param_dim) for _ in range(options.num_classes)], dim=0)
    all_weights.requires_grad = True
    all_bias = torch.stack([np.pi * torch.randn(1) for _ in range(options.num_classes)], dim=0)
    all_bias.requires_grad = True

    optimizer = optim([
        {'params': all_weights},
        {'params': all_bias}
    ],
    lr=options.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=options.lr_factor, patience=options.patience, verbose=True)

    params = (all_weights, all_bias)

    costs, train_acc, test_acc = [], [], []

    q_time, c_time = 0, 0

    # train the variational classifier
    for it in range(options.max_iterations):
        batch_index = np.random.randint(0, num_train, (options.batch_size,))
        feat_vecs_train_batch = feat_vecs_train[batch_index]
        Y_train_batch = Y_train[batch_index]


        #Track how long the quantum part takes
        start_time = time.time()
        optimizer.zero_grad()
        curr_cost = multiclass_svm_loss(q_circuits, params, feat_vecs_train_batch, Y_train_batch, fmap, ansatz, options.num_classes, options.margin, options.qubits)
        end_time = time.time()
        q_time += end_time - start_time

        #track how long the classical part takes
        start_time = time.time()
        curr_cost.backward()
        optimizer.step()
        scheduler.step(curr_cost)
        end_time = time.time()
        c_time += end_time - start_time

        if it % 10 == 0:

            train_idx = random.sample(range(len(feat_vecs_train)), min(100, len(feat_vecs_train)))
            test_idx = random.sample(range(len(feat_vecs_test)), min(100, len(feat_vecs_test)))

            predictions_train = classify(q_circuits, params, feat_vecs_train[train_idx], fmap, ansatz, options.num_classes, options.qubits)
            predictions_test = classify(q_circuits, params, feat_vecs_test[test_idx], fmap, ansatz, options.num_classes, options.qubits)
            acc_train = accuracy(Y_train[train_idx], predictions_train)
            acc_test = accuracy(Y_test[test_idx], predictions_test)

            print("Trial:{} {} | Iter: {} | Cost: {:0.4f} | LR: {:0.4f} \n"
                "".format(options.dataset, options.trial, it, curr_cost.item(), optimizer.param_groups[0]['lr']))

            output_list.append("Trial:{} {} | Iter: {} | Cost: {:0.4f} | LR: {:0.4f}  \n"
                "".format(options.dataset, options.trial, it, curr_cost.item(), optimizer.param_groups[0]['lr']))

            costs.append(curr_cost.item())
            train_acc.append(acc_train)
            test_acc.append(acc_test)
        else:
            output_list.append(f"Cost: {curr_cost.item():.4f} | Iter: {it}")
            print(f"Trial:{options.dataset} {options.trial} | Iter: {it} |  Cost: {curr_cost.item():.4f}")

        options.iterations = it

        #if cost is not improving, finish the training
        if len(costs) > options.stop_train+1:
            if min(costs[-(options.stop_train):]) > min(costs[:-(options.stop_train)]):
                output_list.append(f"Model hasn't improved for {options.stop_train} iterations, stopping training \n")
                break

    # Display the time in the desired format
    options.q_time = q_time
    options.c_time = c_time

    options.highest_acc = {'train': max(train_acc), 'test': max(test_acc)}
    options.last_acc = {'train': train_acc[-1], 'test': test_acc[-1]}

    return costs, train_acc, test_acc, output_list

def plot_and_train(optim, fmap, ansatz, features, Y, options):
    start = time.time()
    
    print(f"Trial:{options.dataset} {options.trial} | Fmap: {options.feature_map} | ansatz: {options.ansatz}")
    
    dev = qml.device("default.qubit", wires=options.qubits)

    qnodes = []
    for _ in range(options.num_classes):
        qnode = qml.QNode(circuit, dev, interface="torch")
        qnodes.append(qnode)


    ansatz_gates = ansatz[0](torch.zeros(options.param_dim), range(options.qubits)).decomposition()
    for gate in ansatz_gates:
        gate_name = str(gate).split('(')[0]
        options.ansatz_gates[gate_name] = options.ansatz_gates.get(gate_name, 0) + 1

    temp_feat = torch.randn(options.feature_size)
    if fmap[0].__name__ != "AmplitudeEmbedding":
        fmap_gates = fmap[0](temp_feat / torch.norm(temp_feat), range(options.qubits)).decomposition()
    else:
        fmap_gates = fmap[0](temp_feat / torch.norm(temp_feat), range(options.qubits), pad_with=0).decomposition()
    
    for gate in fmap_gates:
        gate_name = str(gate).split('(')[0]
        options.fmap_gates[gate_name] = options.fmap_gates.get(gate_name, 0) + 1
    
    #TRAINING STARTS
    costs, train_acc, test_acc, output_list = training(features, Y, optim, fmap[0], ansatz[0], options, qnodes)
    end = time.time()
    options.t_time = end-start

    directory = f"./Trials/{options.dataset}"
    file_name = f"{options.feature_map}_{options.ansatz}"
    directory_cost = os.path.join(directory, "costs")
    directory_train = os.path.join(directory, "train_acc")
    directory_test = os.path.join(directory, "test_acc")
    save_metrics_to_file(directory_cost, file_name, costs)
    save_metrics_to_file(directory_train, file_name, train_acc)
    save_metrics_to_file(directory_test, file_name, test_acc)

    directory = os.path.join(directory, file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'info.txt')
    with open(file_path, 'w') as file:
        for attr_name, attr_value in vars(options).items():
            file.write(f"{attr_name} {attr_value}\n")

    file_path = os.path.join(directory, 'output.txt')
    with open(file_path, "w") as file:
        for line in output_list:
            file.write(line + "\n")
    

directory = './datasets'
datasets = load_datasets(directory)

optimizers = [optim.AdamW]

def amplitude_qubits(feature_size):
    return int(np.ceil(np.log2(feature_size)))
def angle_qubits(feature_size):
    return feature_size

#stores Embedding methods and the amount of qubits they require as a function
FMaps = [(qml.AmplitudeEmbedding, amplitude_qubits),
         (qml.AngleEmbedding, angle_qubits)]

def strong_ent_size(ansatz_layers, qubits):
    return (ansatz_layers, qubits, 3)

def basic_ent_size(ansatz_layers, qubits):
    return (ansatz_layers, qubits)

start_t = time.time()
#stores the ansatz and the dimensions of the weights required
ansatzes = [(qml.StronglyEntanglingLayers, strong_ent_size),
            (qml.BasicEntanglerLayers, basic_ent_size)]

trial = 0

parameters = []
dataset_names = []

options = Options(batch_size_=10, max_iterations_=301,
                margin_=0.3, lr_=0.2, test_split_=0.2, FM_layers_=1,
                ansatz_layers_=1, loss_="Multiclass_SVM", lr_factor_=0.4,
                patience_=30, ansatz_gates_={}, fmap_gates_={},
                iterations_=1, trial_=trial, reduction=4)

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
    dataset_names.append(dataset_name)

    for optim in optimizers:
        for fmap in FMaps:
            for ansatz in ansatzes:
                options.stop_train = 4*options.patience
                options.optimizer = optim.__name__
                options.qubits = fmap[1](options.feature_size)
                options.feature_map = fmap[0].__name__
                options.ansatz = ansatz[0].__name__
                options.params = math.prod(ansatz[1](options.ansatz_layers, options.qubits))
                options.param_dim = ansatz[1](options.ansatz_layers, options.qubits)
                trial += 1
                options.trial=trial

                options_copy = copy(options)
                parameters.append((optim, fmap, ansatz, features, Y, options_copy))
                


pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

pool.starmap(plot_and_train, parameters)

pool.close()
pool.join()

#The only difference between classical and quantum is that classical is an SVM and has better weight initialization
classicalSVM(options)

plot_folders= ["costs", "test_acc", "train_acc"]

for name in dataset_names:
    for folder in plot_folders:
        create_plot_from_files(f"./Trials/{name}/{folder}", folder, (0,2) if folder == 'costs' else None)

end_t = time.time()
print(f"TIME TAKEN FOR ENTIRE ALGORITHM: {end_t-start_t}")