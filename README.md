# DeepNumpy

Deep learning library written from scratch in Numpy. Why? Because it's fun! 🤷🏻‍♂️


## Stuff implemented

- **Layers:**
    - Dense
    - Relu
    - Sigmoid
    - Softmax
    - Dropout
    - Add => Pending...
    - Reshape => Pending...
    - BN => Pending...
    - Conv => Pending...
    - MaxPool => Pending...
    - AvgPool => Pending...
    - RNN => Pending...
    - LSTM => Pending...
   
- **Optimizers:**
    - SGD
    - Momentum
    - Adam => Pending...

- **Losses:**
    - MSE
    - RMSE
    - MAE
    - CrossEntropy
    - BinaryCrossEntropy
    - Hinge => Review

- **Metrics:**
    - MSE
    - RMSE
    - MAE
    - CategoricalAccuracy
    - BinaryAccuracy
    
- **Initializers:**
    - Random normal
    - Random uniform
    - Ones
    - Zeros
    - GlorotNormal => Pending...
    - GlorotUniform => Pending...
    
- **Regularizers:**
    - L1
    - L2
    - L1L2  
    
- **Others:**
    - Multi-input
    - Multi-loss support
    - Gradient checking => Pending...
    - Set modes
    
    
## Example

```python
from sklearn import datasets

from dnpy.layers import *
from dnpy.net import *
from dnpy.optimizers import *
from dnpy.regularizers import *
from dnpy import metrics, losses
    

# Get data
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

# Preprocessing
# Normalize
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# Classes to categorical
num_classes = 3
tmp = np.zeros((len(X), num_classes))
tmp[np.arange(Y.size), Y] = 1.0
Y = tmp

# Shuffle dataset
idxs = np.arange(len(X))
np.random.shuffle(idxs)
X, Y = X[idxs], Y[idxs]

x_train, y_train = X, Y
x_test, y_test = X, Y  # or whatever

# Params *********************************
batch_size = len(x_train)
epochs = 1000

# Define architecture
l_in = Input(shape=(len(x_train[0]),))
l = Dense(l_in, 20, kernel_regularizer=L2(lmda=0.01), bias_regularizer=L1(lmda=0.01))
l = Relu(l)
l = Dropout(l, 0.1)
l = Dense(l, 15)
l = Relu(l)
l = Dense(l, 3)
l_out = Softmax(l)

# Build network
mymodel = Net()
mymodel.build(
    l_in=[l_in],
    l_out=[l_out],
    opt=SGD(lr=0.1),
    losses=[losses.CrossEntropy()],
    metrics=[metrics.CategoricalAccuracy()],
    debug=False
)

# Print model
mymodel.summary(batch_size=batch_size)

# Train
mymodel.fit([x_train], [y_train],
            x_test=[x_test], y_test=[y_test],
            batch_size=batch_size, epochs=epochs,
            evaluate_epoch=True,
            print_rate=10)

# Evaluate
m = mymodel.evaluate([x_test], [y_test], batch_size=batch_size)
```