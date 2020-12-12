import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import load


# ------------------- Define model class -------------------------------------#

class NN_model(tf.keras.Model):
    """Placeholder Docstring."""

    # All hyperparams are inputs in __init__ function after 'self'
    def __init__(self, regu=1e-6, num_neurons=16):
        # Network layers etc.
        pass

    def call(self, x):
        """Call x.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        """
        # Forward propagation
        return x


# ------------------- Load training and validation set -----------------------#

x_train = ...
y_train = ...

x_val = ...
y_val = ...


# ------------------- Define hyperparameters in skopt ------------------------#

dim_learning_rate = Real(low=1e-5, high=1e-3, prior='log-uniform',
                         name='learning_rate')
dim_num_neurons = Integer(low=2, high=4, name='num_neurons')

dim_regu = Real(low=1e-12, high=1e-2, name="regu")

dimensions = [dim_learning_rate,
              dim_num_neurons,
              dim_regu
              ]
default_parameters = [1e-3, 16,  1e-6]


# ------------------- Define fitness function --------------------------------#

# Input: Hyperparameters
# Output: Validation MSE

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, filter_exponent, regu):
    """

    Parameters
    ----------
    learning_rate : TYPE
        DESCRIPTION.
    filter_exponent : TYPE
        DESCRIPTION.
    regu : TYPE
        DESCRIPTION.

    Returns
    -------
    mse : TYPE
        DESCRIPTION.

    """

    # Define loss function
    MSE = tf.keras.losses.MeanSquaredError()

    # Define NN
    model = NN_model(regu=regu, num_neurons=num_neurons)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Set up early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    # Compile model
    model.compile(optimizer=optimizer, loss="mse")

    # Train network
    training_history = model.fit(x_train, y_train,validation_data=(x_val, y_val),epochs=1000,verbose=0,callbacks=[callback])

    # Compute valistaion error
    mse = MSE(model(x_val, ), y_val).numpy()

    # Print MSE
    print()
    print("MSE: {0:.4}".format(mse))
    print()

    # Reset TF computational graph, clear session, and delete model
    del model
    K.clear_session()
    tf.compat.v1.reset_default_graph()

    return mse

####################################################


# ------------------- Run Bayesian Optimization ------------------------------#

# If you have run Bayesian optization earlier and want to continue,
# then load previous history
# bayesiab_history = load('bayesian_history.pkl')
# x0 = bayesiab_history.x_iters
# y0 = bayesiab_history.func_vals


# Save optimization history so you can continue later
checkpoint_saver = CheckpointSaver("directory_for_saving_optimization_history",
                                   compress=9)

# Use this if you are starting a new optimization:
# Optimize using Gaussian Process minimizer (you can try other minimizers)
gp_result = gp_minimize(func=fitness,
                        x0=default_parameters,
                        dimensions=dimensions,
                        n_calls=50,
                        callback=[checkpoint_saver])

# Use this if you are continuing from previous optimization:
# gp_result = gp_minimize(func=fitness,
#                        x0=x0,
#                        y0=y0,
#                        dimensions=dimensions,
#                        n_calls=50,
#                        callback=[checkpoint_saver])
