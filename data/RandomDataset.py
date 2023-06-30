import random
import csv

def generate_dataset(num_instances, num_attributes, num_labels):
    dataset = []
    for _ in range(num_instances):
        attributes = [str(random.randint(0,1)) for _ in range(num_attributes)]
        label = random.randint(0, num_labels-1)
        dataset.append(attributes + [str(label)])
    return dataset

def save_dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(dataset)

# Parameters for the dataset
num_instances = 50  # Number of instances (rows)
num_attributes = 2  # Number of attributes (columns)
num_labels = 2  # Number of labels

# Generate the dataset
dataset = generate_dataset(num_instances, num_attributes, num_labels)

# Save the dataset to a CSV file
filename = 'Random5.csv'
save_dataset_to_csv(dataset, filename)
