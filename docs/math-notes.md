# Math Notes

## CS231n Deep Learning for Computer Vision

Three key components of image classification task:
1. Score Function: Mapping the raw image pixels to class scores (e.g. a linear function)
2. Loss function: measures the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data. there are many ways and versions of this (e.g. Softmax/SVM)
3. Optimization

The linear function has the form $f(x_i, W) = Wx_i$ and the SVM is formulated as 
$$
L = \frac{1}{N}\sum_i \sum_{i \neq y_i} [max(0, f(x_i; W)_j - f(x_i; W)_{yi} + 1)] + \alpha R(W)
$$
A setting of the parameters $W$ that produces predictions for exampels $x_i$ consistent with their ground truth labels $y_i$ would also have a very low loss $L$.

## Visualizing the loss function
The loss functions we'll look at are usually defined over very high-dimensional spaces (e.g. in CIFAR-10 a linear classifier weight matrix is of size [10x3073] for a total of 30730 parameters), making them difficult to visualize. However we can still gain some intuitions about one by slicing through the high-dimensional space along rays(1 dimension), or along planes(2 dimensions). For example, we can generate a random weight matrix $W$ (which corresponds to a single point in the space), then march along a ray and record the loss function value along the way. That is, we can generate a random direction $W_1$ and compute the loss along this direction by evaluating $L(W + \alpha W_1)$ for the different values $\alpha$. This porcess generates a simple plot with the value of $\alpha$ as the x-axis and the value of the loss function as the y-axis. We can also carry out the same procedure with two dimensions by evaluating the loss $L(W + \alpha W_1 + \beta W_2)$ as we vary $\alpha$, $\beta$. In a plot, $\alpha$, $\beta$ could then correspond to the x-axis and the y-axis, and the value of the loss function can be visualized with a color:
![visualization](/home/aron/projects/transformer-from-scratch/docs/figures/ebon.png)  

We can explain the piecewise-linear structure of the loss function by examining the math. For a single example we have: 
$$
L_i = \sum_{j \neq y_i} [max(0, w_j^T x_i - w_{yi}^T x_i + 1)]
$$
It is clear from the equation that the data loss for each example is a sum of (zero-threshold due to the $max(0, -)$ function) linear functions of $W$. Moreover, each row of $W$ (i.e. $w_j$) sometimes has a positive sign in front of it (when it corresponds to the correct class for that example). To make this more explicit, consider a simple dataset that contains three 1-dimensional points and three classes. The full SVM loss (without regularization) becomes:
$$
L_0 = max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + max(0, w_2^Tx_0 - w_0^Tx_0 + 1)\\
L_1 = max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + max(0, w_2^Tx_1 - w_1^Tx_1 + 1)\\
L_2 = max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + max(0, w_1^Tx_2 - w_2^Tx_2 + 1)\\
L = \frac{(L_0 + L_1 + L_2)}{3}
$$

Since these examples are 1-dimensional, the data $x_i$ and weights $w_j$ are numbers. Looking at, for instance, $w_0$, some terms above are linear functions of $w_0$ and each is clamped at zero.  

Non-differentiable loss functions: the kinks in the loss function (due to the max operation) technically make the loss function non-differentiable because at these kinks the gradient is not defined. However, the subgradient still exists and is commonly used instead.  

## Optimization
The loss function lets us quantify the quality of any particular set of weights $W$. The goal of the optimization is to find $W$ that minimizes the loss function.  

### Following the gradient
We can compute the best direction along which we should change out weight vector that is mathematically guaranteed to be the direction of the steepest descent (at least in the limit as the step size goes towards zero). This direction will be related to the gradient of the loss function.  
In one-dimensional functions, the slope is the instantaneous rate of change of the function at any point you might be interested in. The gradient is a generalization of slope for functions that dont take a single number but a vector of numbers. Additionally, the gradient is just a vector of slopes (more commonly referred to as derivatives) for each dimension in the input space. The mathematical expression for the derivative of a 1-D functin with respect to its input is:
$$
\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}
$$
When the functions of interest take a vector of numbers instead of a single number, we call the derivatives partial derivatives, and the gradient is simply the vector of partial derivatives in each dimension.

### Computing the gradient
There are two ways:
1. Numerical gradinet (slow; approximate; easy)
2. Analytical gradient (fast; exact; error-prone)

#### Numerical Gradient
The formula given above allows us to compute the gradient numerically. Here is a generic function that takes a function `f`, a vector `x` to evaluate the gradient on, and returns the gradient of `f` at `x`:
```python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```
The above code does the following:
- Sets `grad` as an array of `zeroes` of size `x.shape`
- Starts an iterator that iterates through each dimension of `x`
- In each iteration, the current value of `x` at a specific index is stored, then is incremented by `h` and replaced in `x`
- The newly incremented value is used to calculate the $f(x + h)$ 

# Softmax

$$
Softmax(x_i) = \frac{e^{x_i}}{\sum_j{e^{x_j}}}
$$