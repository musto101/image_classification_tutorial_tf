import tensorflow as tf
import kerastuner as kt
import numpy as np
from tensorflow import keras

print(f"TensorFlow Version: {tf.__version__}")
print(f"KerasTuner Version: {kt.__version__}")

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

train_images = np.concatenate(list(train_ds.map(lambda x, y: x)))
train_labels = np.concatenate(list(train_ds.map(lambda x, y: y)))

val_images = np.concatenate(list(val_ds.map(lambda x, y: x)))
val_labels = np.concatenate(list(val_ds.map(lambda x, y: y)))

inputs = np.concatenate((train_images, val_images), axis=0)
targets = np.concatenate((train_labels, val_labels), axis=0)

X_train = train_images.astype('float32') / 255.0
X_test = val_images.astype('float32') / 255.0

# baseline model
b_model = keras.Sequential()
b_model.add(keras.layers.Flatten(input_shape=(180, 180, 3)))
b_model.add(keras.layers.Dense(units=512, activation='relu', name='dense_1'))
b_model.add(keras.layers.Dropout(0.2))
b_model.add(keras.layers.Dense(10, activation='softmax'))

b_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

NUM_EPOCHS = 20

# Early stopping set after 5 epochs
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train model
b_model.fit(X_train, train_labels, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=2)

import pandas as pd


def evaluate_model(model, X_test, y_test):
    """
    evaluate model on test set and show results in dataframe.

    Parameters
    ----------
    model : keras model
        trained keras model.
    X_test : numpy array
        Features of holdout set.
    y_test : numpy array
        Labels of holdout set.

    Returns
    -------
    display_df : DataFrame
        Pandas dataframe containing evaluation results.
    """
    eval_dict = model.evaluate(X_test, y_test, return_dict=True)

    display_df = pd.DataFrame([eval_dict.values()], columns=[list(eval_dict.keys())])

    return display_df


# Evaluate model on test set and add results to dataframe
results = evaluate_model(b_model, X_test, val_labels)

# Set index to 'Baseline'
results.index = ['Baseline']

# Display results
results.head()


# tuning
def build_model(hp):
    """
    Builds model and sets up hyperparameter space to search.

    Parameters
    ----------
    hp : HyperParameter object
        Configures hyperparameters to tune.

    Returns
    -------
    model : keras model
        Compiled model with hyperparameters to tune.
    """
    # Initialize sequential API and start building model.
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(180, 180, 3)))

    # Tune the number of hidden layers and units in each.
    # Number of hidden layers: 1 - 5
    # Number of Units: 32 - 512 with stepsize of 32
    for i in range(1, hp.Int("num_layers", 2, 6)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu")
        )

        # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
        model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))

    # Add output layer.
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Define optimizer, loss, and metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model


tuner = kt.Hyperband(build_model,
                     objective="val_accuracy",
                     max_epochs=20,
                     factor=3,
                     hyperband_iterations=10,
                     directory="kt_dir",
                     project_name="kt_hyperband", )

tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, train_labels, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=2)

best_hps=tuner.get_best_hyperparameters()[0]

# Build model
h_model = tuner.hypermodel.build(best_hps)

# Train the hypertuned model
h_model.fit(X_train, train_labels, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=2)

hyper_df = evaluate_model(h_model, X_test, val_labels)

# Set index to hypertuned
hyper_df.index = ["Hypertuned"]

# Append results in dataframe
results.append(hyper_df)

hypermodel = kt.applications.HyperResNet(input_shape=(180, 180, 3), classes=5)

# Instantiate tuner with bayesian optimzation search algorithm and our hypermodel
tuner = kt.tuners.BayesianOptimization(
    hypermodel,
    objective='val_accuracy',
    max_trials=3,
    directory="kt_dir",
    project_name="kt_bayes_resnet")
