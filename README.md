# Programming I Semester project: Multi-layer perceptron neural network

A semi-general purpose, modular, MLP model (see long comments below). Interective python env encouraged for instantiation and serialization (dill). Minor changes allow for use as a module, when loading network for practical application. 

MLPs are, in general, fairly primitive, but they lay the groundwork for understanding more sophisticated and different models. What this means for this implementation is that it's going to be kind of slow, relatively innacurate, and more prone to overfitting and such. Nevertheless, it can achieve ~95-96% trained on MNIST, ~97-98% trained on EMNIST "Digits" (b/c much larger) and mid-to-high 80% for
letters, where the limitations of this model really hurt.

Long comments (referenced in source file):
1: Loading "Digits" EMNIST dataset by default, can be trivially modified for MNIST or letters, which numpy_datasets also provides (be careful when instantiating network, appropriate number of input/outpu nodes...). Please refer to their website for more info: <https://numpy-datasets.readthedocs.io/en/latest/modules/images.html>

<-Below comments are informative in nature or explain certain choices->

2: Defining a parameterized Network class that will allow for control over almost all aspects of the neural network for any similar application (this is essentially a general structure for an MLP with the goal of binary classification, think for example classifying which pictures are dogs or cats, letters etc; anything you can find a solid DB for)

3: Note for all the initializations, np.random.random((x,y)) creates an x by y matrix with random values in the range [0.0,1.0), I am making a dense neural network so every node from previous layer is connected to every other node in the next, afterwards multiplying matrix to desired values (*2-1 gets values from [-1.0, 1.0) ...)

4: Creating space for activations; input layer set with input examples, o/w set by calculation (see 'sample' function again), very convenient to have activation set as an object parameter, mainly in backpropagation step.

5: This is a layer-general partial derivative that updates weights and biases from the back to the front as it computes (hence backpropagation). The fact that the whole layer at once can be computed is thanks to numpy's array operations that implement element-wise  calculation, and that partial derivatives of shallower layers build on the already calculated deeper ones, makes this a very efficient algorithm and the heart and soul of neural networks.

6: The following line is the first two partial derivatives and is in such a form because the derivative of the binary cross entropy cost function used, and the sigmoid activation function, cancel out, and are, as mentioned, common to all the calculations.

7: For the deepest layer, only one final partial derivative is calculated, and that one is different between biases and weights (even though it is taking the derivative of the same value, just w.r.t different things, namely bias or weight vector/matrix...). Note also that we are actually doing nothing more to the already calculated partial derivative when updating biases, or rather we are multiplying by one because the final derivative is d(sum before sigmoid)/d(biases), and as we can see in the sample function, and when taking the derivatives of all the sums individually, everything will be 0 except d/d(biases)(self.biases) which would, with numpy operations, be a vector of all ones; think d/dx(x) = 1...

8: After the first two common derivatives, after the deepest layer, every other one requires the computation of two more than the layer after (or rather before) it. This follows from the derivation of the partial derivatives, and they are as such:

9: This 'final' derivative is in a sense common to all the layers, just taking values from a different layer's activation. Here we see that every layer requires 2*n+1 derivatives to be computed, where n is the number of the layer counting from 1 and backwards. However, the performance is not 2*(1+2+3...n)+n = O(n^2) for n layers but rather really only 2n+1 = O(n) because they don't have to be computed for every layer again, and are just remembered, because they use the same values (dynamic :D !!)

10: Effectively the top-level code for doing backpropagation over and over again over the set of labeled examples, for my implementation I got caught up with trying to make the network super general purpose, and certainly it can be done by for example adding a parameter to 'train' which would be forwarded to 'backpropagation' telling it to do SGD with the whole batch, or in mini-batches etc. however it would be less so challenging, and moreso annoying, so I decided to keep it simple. The hyperparameters of the network that can therefore be customized is the number of hidden layers, the number of nodes in hidden layers, and the learning rate.
