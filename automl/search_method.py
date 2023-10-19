#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Import DL livraries (APIs) bulding up DL pipelines and AutoDL livraries (APIs) for tuning DL pipelines
import tensorflow as tf
import keras_tuner as kt

from log.tensorboard import name_logdir
from automl.search_algorithm import BayesianOptimizationOracle
from automl.tuner import DeepTuner, build_regressor
#-------------------------------------------------------

# Use build-in random search algorithm to tune models
def random_tuner(X_train,y_train,X_valid,y_valid,
                 max_trials, exec_per_trial,
                 directory,proj_name,logdir,
                 patience=None, epochs=None):

    # Use build-in random search algorithm to tune models
    rt = DeepTuner( # Function call DeepTuner
        # Provides the customized random search oracle to the tuner
        oracle=kt.oracles.RandomSearch(  # Initializes the custom tuner
            objective=kt.Objective("mae", "min"), # Uses the MAE as the objective for model comparison and oracle update
            max_trials=max_trials, # The total number of  different hyperparameter value sets to try
            seed=42,
        ),
        hypermodel=build_regressor, # Function call build_regressor, passes the search space to the tuner
        overwrite=True, # Overwrites the previous project if one existed
        executions_per_trial=exec_per_trial, # The number of runs for one  hyperparameter value set
        directory=directory,# The directory in which to save the results
        project_name=proj_name, # The name of the project
    )
    # Create an list of callbacks
    # We'll use a callback to stop training when our performance metric reaches a specified level.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="mae", mode="auto",patience=patience,restore_best_weights=True,)

    # To define a List of callbacks to use
    callbacks = [early_stopping_cb, name_logdir(logdir)]

    # Execute the search process
    rt.search(X_train, y_train,validation_data=(X_valid,y_valid),callbacks=callbacks,epochs=epochs,verbose=2)
    # Return tuner Random Search
    return rt
#-------------------------------------------------------

# Use customized Bayesian Optimization search algorithm to tune models
def bo_tuner(X_train, y_train, X_valid, y_valid,
             acq_type,
             max_trials, exec_per_trial,
             directory, proj_name, logdir,
             patience=None, epochs=None
             ):
    bt = DeepTuner(
        # Uses the customized Bayesian optimization search oracle
        oracle=BayesianOptimizationOracle(
            objective=kt.Objective("mae", "min"),
            max_trials=max_trials,
            acq_type=acq_type,  # you can switch between different acquisition functions
            seed=42,
        ),
        hypermodel=build_regressor,  # Function call build_model, passes the search space to the tuner
        overwrite=True,
        executions_per_trial=exec_per_trial,
        directory=directory,
        project_name=proj_name,
    )

    # We'll use a callback to stop training when our performance metric reaches a specified level.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="mae", mode="auto", patience=patience, restore_best_weights=True)

    # To define a List of callbacks to use
    callbacks = [early_stopping_cb, name_logdir(logdir)]

    # Execute the search process
    bt.search(X_train, y_train,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks,
              epochs=epochs, verbose=2
              )
    # Return tuner Bayesian optimization
    return bt