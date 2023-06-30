import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def skSVM(features, labels, num_epochs):
    # Convert the features and labels to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Lists to store training accuracy and test accuracy during training
    train_accuracy = []
    test_accuracy = []

    # Train the model and collect the metrics
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=1)

    # Get the training and test accuracies from the history object
    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']

    # Plot the training accuracy and test accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(range(num_epochs), train_accuracy, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('plot.png')
