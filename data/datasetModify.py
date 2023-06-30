def convert_labels(dataset):
    converted_dataset = []
    for line in dataset:
        values = line.strip().split(',')
        labels = values[:-1]
        #if not integer labels remove the int
        label = int(values[-1])
        #Change labels to whatever you want
        if label == 1:
            label = 0
        elif label == 2:
            label = 1
        elif label == 3:
            label = 2
        labels.append(str(label))
        converted_line = ','.join(labels)
        converted_dataset.append(converted_line)
    return converted_dataset

# Load dataset from file
filename = './datasets/hayes-roth.csv'
with open(filename, 'r') as file:
    dataset = file.readlines()

# Convert labels
converted_dataset = convert_labels(dataset)

# Save converted dataset to the same file
with open(filename, 'w') as file:
    file.write('\n'.join(converted_dataset))






