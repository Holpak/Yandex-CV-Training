# %% [markdown]
# Credits: this notebook belongs to [Practical DL](https://docs.google.com/forms/d/e/1FAIpQLScvrVtuwrHSlxWqHnLt1V-_7h2eON_mlRR6MUb3xEe5x9LuoA/viewform?usp=sf_link) course by Yandex School of Data Analysis.

# %%
import numpy as np

# %% [markdown]
# **Module** is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.

# %%


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"

# %% [markdown]
# # Sequential container

# %% [markdown]
# **Define** a forward and backward pass procedures.

# %%


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """
        current_input = input
        for module in self.modules:
            current_input = module.forward(current_input)
        self.output = current_input
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To each module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i-1]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        current_gradOutput = gradOutput
        for i in range(len(self.modules) - 1, 0, -1):
            module_input = self.modules[i-1].output
            current_gradOutput = self.modules[i].backward(
                module_input, current_gradOutput)

        self.gradInput = self.modules[0].backward(input, current_gradOutput)
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list of lists.
        """
        return [module.getParameters() for module in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list of lists.
        """
        # <<< FIX: Return list of lists >>>
        return [module.getGradParameters() for module in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()

# %% [markdown]
# # Layers

# %% [markdown]
# ## 1. Linear transform layer
# Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**

# %%


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = np.dot(input, self.W.T) + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradW = np.dot(gradOutput.T, input)
        self.gradb = np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q

# %% [markdown]
# ## 2. SoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
#
# $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
#
# Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.

# %%


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()

    def updateOutput(self, input):
        input_stable = np.subtract(input, input.max(axis=1, keepdims=True))
        exps = np.exp(input_stable)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        self.output = exps / sum_exps
        return self.output

    def updateGradInput(self, input, gradOutput):
        dot_prod = np.sum(gradOutput * self.output, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - dot_prod)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"

# %% [markdown]
# ## 3. LogSoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
#
# $\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$
#
# The main goal of this layer is to be used in computation of log-likelihood loss.

# %%


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        input_stable = np.subtract(input, input.max(axis=1, keepdims=True))
        log_sum_exp = np.log(
            np.sum(np.exp(input_stable), axis=1, keepdims=True))
        self.output = input_stable - log_sum_exp
        return self.output

    def updateGradInput(self, input, gradOutput):
        softmax_output = np.exp(self.output) 
        sum_gradOutput = np.sum(gradOutput, axis=1, keepdims=True)
        self.gradInput = gradOutput - softmax_output * sum_gradOutput
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"

# %% [markdown]
# ## 4. Batch normalization
# One of the most significant recent ideas that impacted NNs a lot is [**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. **You are** to implement the first part of the layer: features normalization. The second part (`ChannelwiseScaling` layer) is implemented below.
#
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
#
# The layer should work as follows. While training (`self.training == True`) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
# where $\mu$ and $\sigma$ - mean and variance of feature values in **batch** and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance:
# ```
#     self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
#     self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
# ```
# During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance.
#
# Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common **implementation** choice. In general "batch normalization" always assumes normalization + scaling.

# %%


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.9):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None
        
        self.x_minus_mean = None
        self.inv_std = None
        self.x_hat = None
        self.batch_size = None

    def updateOutput(self, input):
        if self.training:
            batch_mean = np.mean(input, axis=0)
            batch_variance = np.var(input, axis=0)

            self.batch_size = input.shape[0]
            self.x_minus_mean = input - batch_mean
            self.inv_std = 1. / np.sqrt(batch_variance + self.EPS)
            self.x_hat = self.x_minus_mean * self.inv_std
            self.output = self.x_hat

            if self.moving_mean is None:
                self.moving_mean = batch_mean
                self.moving_variance = batch_variance
            else:
                self.moving_mean = self.alpha * \
                    self.moving_mean + (1 - self.alpha) * batch_mean
                self.moving_variance = self.alpha * \
                    self.moving_variance + (1 - self.alpha) * batch_variance
        else:
            if self.moving_mean is None or self.moving_variance is None:
                raise RuntimeError(
                    "BatchNormalization must be trained before evaluation or moving averages must be initialized.")
            self.output = (input - self.moving_mean) / \
                           np.sqrt(self.moving_variance + self.EPS)
            self.x_minus_mean = None
            self.inv_std = None
            self.x_hat = None
            self.batch_size = None

        return self.output

    def updateGradInput(self, input, gradOutput): 
        if self.training:
            if self.x_hat is None or self.inv_std is None or self.batch_size is None:
                 raise RuntimeError("Run forward pass in training mode before backward pass. Missing cached variables.")

            m = self.batch_size
            gy = gradOutput 

            
            sum_gy = np.sum(gy, axis=0)
            sum_gy_xhat = np.sum(gy * self.x_hat, axis=0) 

            
            term1 = m * gy
            term2 = sum_gy
            term3 = self.x_hat * sum_gy_xhat 

            
            self.gradInput = (1. / m) * self.inv_std * (term1 - term2 - term3) 

        else:

            if self.moving_mean is None or self.moving_variance is None:
                 raise RuntimeError("Moving averages not available for eval mode backward pass.")
            inv_std_eval = 1. / np.sqrt(self.moving_variance + self.EPS)
            self.gradInput = gradOutput * inv_std_eval

        return self.gradInput

    def __repr__(self):
        return f"BatchNormalization(alpha={self.alpha})"

# %%
class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()


        self.gamma = np.ones(n_out)
        self.beta = np.zeros(n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradGamma = np.sum(gradOutput*input, axis=0)
        self.gradBeta = np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"

# %% [markdown]
# ## 5. Dropout
# Implement [**dropout**](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple: just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.
#
# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
#
# While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `self.output = input`.
#
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**

# %%
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if not (0.0 <= p < 1.0):
             raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training:
            
            keep_prob = 1.0 - self.p
            self.mask = (np.random.rand(*input.shape) < keep_prob) / keep_prob
            self.output = input * self.mask
        else:
            
            self.output = input
            self.mask = None 
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            if self.mask is None:
                 raise RuntimeError("Mask is None during training backward pass. Forward pass must run first in training mode.")
            self.gradInput = gradOutput * self.mask
        else:
            
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return f"Dropout(p={self.p})"

# %% [markdown]
# # Activation functions

# %%
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , (input > 0).astype(float))
        return self.gradInput

    def __repr__(self):
        return "ReLU"

# %%
class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return  self.output

    def updateGradInput(self, input, gradOutput):
        grad = np.where(input > 0, 1.0, self.slope)
        self.gradInput = gradOutput * grad
        return self.gradInput

    def __repr__(self):
        return f"LeakyReLU(slope={self.slope})"

# %%
class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def updateOutput(self, input):
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        return  self.output

    def updateGradInput(self, input, gradOutput):
        # grad = 1 if x > 0 else alpha * exp(x)
        # alpha * exp(x) is also equal to ELU(x) + alpha for x <= 0
        grad = np.where(input > 0, 1.0, self.output + self.alpha)
        self.gradInput = gradOutput * grad
        return self.gradInput

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"

# %%
class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        
        self.output = np.logaddexp(0, input)
        return  self.output

    def updateGradInput(self, input, gradOutput):
        sigmoid = 1.0 / (1.0 + np.exp(-input))
        self.gradInput = gradOutput * sigmoid
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"

# %% [markdown]
# # Criterions

# %%
class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        return self.updateOutput(input, target)

    def backward(self, input, target):
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        return self.output

    def updateGradInput(self, input, target):
        return self.gradInput

    def __repr__(self):
        return "Criterion"

# %%
class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        self.output = np.sum(np.power(input - target,2)) / input.shape[0] # Average over batch dimension
        return self.output

    def updateGradInput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        self.gradInput  = (input - target) * 2 / input.shape[0] # Scale gradient by 2/N
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

# %%
class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        batch_size = input.shape[0]
        # Assumes target is one-hot
        self.output = -np.sum(target * np.log(input_clamp)) / batch_size
        return self.output

    def updateGradInput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        batch_size = input.shape[0]
        # Assumes target is one-hot
        self.gradInput = - (target / input_clamp) / batch_size
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"

# %%
class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        batch_size = input.shape[0]
        self.output = -np.sum(target * input) / batch_size
        return self.output

    def updateGradInput(self, input, target):
        assert input.shape == target.shape, f"Input shape {input.shape} and target shape {target.shape} do not match"
        batch_size = input.shape[0]
        self.gradInput = - target / batch_size
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"

# %% [markdown]
# # Optimizers

# %% [markdown]
# ### SGD optimizer with momentum
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate` and `momentum`)
# - `state` - dict with optimizator state (used to save accumulated gradients)

# %%
def sgd_momentum(variables, gradients, config, state):
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        if not current_layer_vars:
            continue
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            if current_grad is None: 
                 var_index += 1
                 continue

            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.multiply(old_grad, config['momentum'], out=old_grad)
            np.add(old_grad, config['learning_rate'] * current_grad, out=old_grad) 

            current_var -= old_grad 

            var_index += 1

# %% [markdown]
# ## 11. [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate`, `beta1`, `beta2`, `epsilon`)
# - `state` - dict with optimizator state (used to save 1st and 2nd moment for vars)
#
# Formulas for optimizer:
#
# Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
# First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$
# Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
# New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{m_t}{\sqrt{v_t} + \epsilon}$$

# %%
def adam_optimizer(variables, gradients, config, state):
    state.setdefault('m', {})
    state.setdefault('v', {})
    state.setdefault('t', 0) 
    state['t'] += 1
    t = state['t']
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    lr = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']

    lr_t = lr * np.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        if not current_layer_vars:
            continue

        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            if current_grad is None:
                var_index += 1
                continue

            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            np.multiply(var_first_moment, beta1, out=var_first_moment)
            np.add(var_first_moment, (1.0 - beta1) * current_grad, out=var_first_moment)

            np.multiply(var_second_moment, beta2, out=var_second_moment)
            np.add(var_second_moment, (1.0 - beta2) * np.square(current_grad), out=var_second_moment)

            update_step = lr_t * var_first_moment / (np.sqrt(var_second_moment) + eps)
            current_var -= update_step

            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)

            var_index += 1


# %% [markdown]
# # Layers for advanced track homework

# %%
import scipy as sp
import scipy.signal

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding with stride 1"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        self.input_padded = None 

    def updateOutput(self, input):
        batch_size, _, H, W = input.shape
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        self.input_padded = np.pad(input, pad_width=pad_width, mode='constant', constant_values=0)
        H_out, W_out = H, W 

        self.output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for n in range(batch_size):
            for c_out in range(self.out_channels):
                res_corr = np.zeros((H_out, W_out))
                for c_in in range(self.in_channels):
                    res_corr += scipy.signal.correlate2d(
                        self.input_padded[n, c_in, :, :],
                        self.W[c_out, c_in, :, :],
                        mode='valid' 
                    )
                self.output[n, c_out, :, :] = res_corr + self.b[c_out]

        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size, _, H_in, W_in = input.shape
        pad_full = self.kernel_size - 1
        gradOutput_padded = np.pad(
            gradOutput,
            pad_width=((0, 0), (0, 0), (pad_full, pad_full), (pad_full, pad_full)),
            mode='constant',
            constant_values=0
        )
        W_rot180 = np.flip(self.W, axis=(2, 3)) 
        gradInput_padded = np.zeros_like(self.input_padded)

        
        for n in range(batch_size):
            for c_in in range(self.in_channels):
                res_conv = np.zeros(gradInput_padded.shape[2:])
                for c_out in range(self.out_channels):
                    res_conv += scipy.signal.correlate2d(
                        gradOutput_padded[n, c_out, :, :],
                        W_rot180[c_out, c_in, :, :],
                        mode='valid'
                    )
                gradInput_padded[n, c_in, :, :] = res_conv

        
        self.gradInput = gradInput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        assert self.gradInput.shape == input.shape, f"GradInput shape {self.gradInput.shape} mismatch with input shape {input.shape}"
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        batch_size, _, H_out, W_out = gradOutput.shape

        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                gradW_accum = np.zeros((self.kernel_size, self.kernel_size))
                for n in range(batch_size):

                    gradW_accum += scipy.signal.correlate2d(
                        self.input_padded[n, c_in, :, :], 
                        gradOutput[n, c_out, :, :],       
                        mode='valid' 
                    )
                self.gradW[c_out, c_in, :, :] += gradW_accum
        self.gradb += np.sum(gradOutput, axis=(0, 2, 3)) 

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = f'Conv2d({s[1]}, {s[0]}, kernel_size={s[2]}, padding={self.padding})'
        return q

# %%
class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.max_indices_mask = None

    def updateOutput(self, input):
        N, C, H, W = input.shape
        K = self.kernel_size
        assert H % K == 0, f'Input height {H} not divisible by kernel size {K}'
        assert W % K == 0, f'Input width {W} not divisible by kernel size {K}'
        H_out, W_out = H // K, W // K

        
        reshaped = input.reshape(N, C, H_out, K, W_out, K)
        self.output = reshaped.max(axis=(3, 5))

        output_expanded = self.output[:, :, :, np.newaxis, :, np.newaxis]
        self.max_indices_mask = (reshaped == output_expanded)

        self.max_indices_mask = self.max_indices_mask.reshape(N, C, H, W)

        assert self.output.shape == (N, C, H_out, W_out), f"Output shape is {self.output.shape}"
        return self.output

    def updateGradInput(self, input, gradOutput):
        N, C, H, W = input.shape
        N_out, C_out, H_out, W_out = gradOutput.shape
        K = self.kernel_size

        gradOutput_upsampled = gradOutput.repeat(K, axis=2).repeat(K, axis=3)

        if self.max_indices_mask is None:
             raise RuntimeError("Max indices mask is None. Forward pass must run first.")
        if self.max_indices_mask.shape != gradOutput_upsampled.shape:
             raise RuntimeError(f"Mask shape {self.max_indices_mask.shape} doesn't match upsampled gradOutput shape {gradOutput_upsampled.shape}")

        mask_reshaped = self.max_indices_mask.reshape(N, C, H_out, K, W_out, K)
        mask_sums = mask_reshaped.sum(axis=(3, 5), keepdims=True)
        mask_sums[mask_sums == 0] = 1
        normalized_mask_reshaped = mask_reshaped / mask_sums
        normalized_mask = normalized_mask_reshaped.reshape(self.max_indices_mask.shape)

        self.gradInput = gradOutput_upsampled * normalized_mask

        assert self.gradInput.shape == input.shape, f"GradInput shape {self.gradInput.shape} mismatch with input shape {input.shape}"
        return self.gradInput

    def __repr__(self):
        q = f'MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})'
        return q

# %%
class Flatten(Module):
    def __init__(self):
         super(Flatten, self).__init__()
         self.original_shape = None

    def updateOutput(self, input):
        self.original_shape = input.shape
        self.output = input.reshape(self.original_shape[0], -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.original_shape is None:
            raise RuntimeError("Original shape not stored. Forward pass must run first.")
        self.gradInput = gradOutput.reshape(self.original_shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"

# %%
