#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing libraries needed for Operating System Manipulation in Python
import os

# Import DL livraries (APIs) bulding up DL pipelines and AutoDL livraries (APIs) for tuning DL pipelines
import tensorflow as tf
import keras_tuner as kt
#-------------------------------------------------------

# Creating a tuner called DeepTuner by extending the base tuner class
class DeepTuner(kt.Tuner):
    def run_trial(self, trial, X, y, validation_data, **fit_kwargs):
        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(X, y,
                  batch_size=trial.hyperparameters.Choice("batch_size", [32, 64]),
                  **fit_kwargs  # Trains model with a tunable batch size
                  )
        # get the validation data
        X_val, y_val = validation_data
        eval_scores = model.evaluate(X_val, y_val)

        # save the model to disk
        self.save_model(trial.trial_id, model)

        # inform the oracle of the eval result, the result is a dictionary with the metric names as the keys.
        return {
            name: value for name, value in zip(model.metrics_names, eval_scores)
        }
    '''
    Since TensorFlow Keras provides methods to save and load the models, 
    we can adopt these methods to implement the save_model() and load_model() functions.
    '''
    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "models")
        model.save(fname)
    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "models")
        model = tf.keras.models.load_model(fname)

        return model
#-------------------------------------------------------

# Creates a search space for tuning MLPs
def build_regressor(hp):

    # Define the model structure sequential
    model = tf.keras.models.Sequential()

    # Defines the input dimension of the network
    model.add(tf.keras.Input(shape=(5,)))

    # Set the number of layers as a hyperparameter
    for i in range(hp.Int("num_layers",1,4)):
        model.add(
            tf.keras.layers.Dense(
                # Dynamically generates a new hyperparameter for each layer, ensuring the hyperparameter names are not the same
                hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                hp.Choice("activation",["relu","selu"]),
            )
        )
    # Adds a dense classification layer with sigmoid activation
    # model.add(tf.keras.layers.Dense(1,"sigmoid"))
    model.add(tf.keras.layers.Dense(1))  # no activation function by default

    # Instantiates the hp.Choice() method to select the optimization method and method hp.Float for select interval for learning rate
    # Uniformly random sample at logarithmic magnitude
    optimizer_name = hp.Choice("optimizer", ["adam","nadam","rmsprop"])
    learning_rate  = hp.Float("learning_rate", min_value=1e-4, max_value=0.1, sampling="log")

    # Instantiates the hp.Choice() method to select the optimization method and learning rate
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name =="nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer,  # Defines the search space for the optimizer
                  loss=tf.keras.losses.Huber(),  # Compiles the model set the loss
                  metrics=["mse","mae","mape"])  # the metric we're using is MAE, MAPE and MSE.

    # Return the model is keras
    return model