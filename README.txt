#Feed forward neural net implementation
A mixture of a personal and computer architecture final project

With help from:

https://www.coursera.org/course/neuralnets

https://www.udacity.com/course/viewer#!/c-cs344/l-55120467

##What I did:
I created A feed forward neural net capable of training on a GPU (using Cuda) and CPU (using Armadillo for fast matrix math).
It allows an arbitrary number of layers with the restriction that only a single activation function may be used across all layers.
It trains using mini batch gradient decent with momentum.
This network can be used to classify MNIST, (hand written digits) with > 95% accuracy.

##Why I did it:
Neural networks are currently a very popular and powerful technique in machine learning.
They are being used by Google for speech recognition as well as for image processing.
Previously, I have been working with a great implementation written by another Olin student, Alec Radford, but sadly, it was not quite fast enough and I didn't understand its inner workings.
To solve both of these issues, I turned to Cuda and C++.

##Build:
Required software:
CMake, Cuda, Armadillo, google-test(gtest), Clang++ (with support for C++11)
(tested on Arch Linux, paths will need to be changed depending on OS)
`
cd ... (this directory)
cmake .
make -j4
./bin/tests
`
If everything passes, you should be good to go!

####WARNING:
This library and samples use an incredibly large amount of computation.
Save all work, as may cause system to crash due to temperature (such as on my laptop).

##Sample Usage:
For more examples see samples/.
`
//Create a network with 4 layers using the Logistic activation function and Squared_Error error function.
FeedForward_Network<Logistic, Squared_Error> network({10, 20, 30, 2});
//Randomize the initial weights in the network
randomize(network);

//train on CPU
train_batch(network, features, targets, batch_size, learning_rate, momentum);
//train on GPU
GPU_train_batch(network, features, targets, batch_size, learning_rate, momentum);

//train on CPU
arma::Mat<float> outputs = predict(network, features);
//train on GPU
arma::Mat<float> outputs_GPU = GPU_predict(network, features);
`

##Results:
On an Olin laptop, performance on the CPU is better.
On a desktop computer with a medium high grade GPU, GPU outperforms by a factor of 4 to 5.

##files
###src/
####net.hpp
  Contains implementation of FeedForward_Network
####net_raw.hpp
  Contains Raw_FeedForward_Network class

####net_CPU.hpp
  Contains implementation for CPU training and prediction
####net_GPU.hpp
  Contains highlevel implementation for GPU training and prediction

####net_GPU_impl.{cu, hpp}
  Contains Cuda implementation of training and prediction. Called by net_GPU. Should not be interacted with by end user.

####net_raw_utils.hpp
  Contains helper functions for converting between arma::Mat<float> to Raw_Matrix as well as Raw_FeedForward_Network to FeedForward_Network

###test/
Unit tests for the various pieces. Filenames should match the above.

###samples/
####{CPU, GPU}_mnist.cpp
Contains a CPU and GPU implementation to solve MNIST.
####mnist.arma
Armadillo binary of mnist data. Uploaded for convenience.

###tools/
####convert_to_arma.cpp
Helper executable to convert between csv's downloaded from kaggle to arma binary files.

##Design decisions:
One of the main goals was to keep the data (neural net weights and activations), and the training and predictive functions separate.
This would allow the user to be free to choose implementations that work the best as well as add more.
I choose to make these implementations top level functions as opposed to methods to further increase the modularity of the system.
Users are free to add their own implementations as well as mix and match CPU and GPU implementations depending on performance.

The neural net can be in the form of two different representations. The first is the standard FeedForward_Network.
This version uses arma::Mat<float> matrices for all of its internal storage.
While great for CPU development and memory safety, this implementation cannot be transfered to the GPU.
To do that, the network is converted to a Raw_FeedForward_Network. This network uses raw buffers, float *, to store all of the data.
This split in representation was probably a poor idea. It requires a large number of data copying functions, as well as adds overhead converting.
Ideally these should have been the same.

Another key feature of this implementation is the use of templates for layer configuration.
These template-d types passed in are containers that contain the requested function and function derivatives.
Because these are templates as opposed to dynamic via function pointers, the functions can be inlined by the compiler and keep good code locality by avoiding extra function calls.
While this may not seem like much, these functions are as small as 1-2 cycles, so there could be a large gain.
That being said, this has not been tested and could just be an example of premature optimization.

Parallelizing GPU implementation:
Much of the GPU implementation can be parallelized a number of different ways.
When designing a parallel algorithm, it's important to split the problem up into as many small chunks as possible. This can be quite challenging.
I choose to split up each stage of the GPU implementation into 2 dimensions.
Much of the operations resemble matrix multiplications. I chose to have each entry in the output matrix have its own thread.
Each thread then performs a gather operation on the input matrices to take the dot product of the needed row of the first matrix and column of the second.
While there are better ways to multiply matrices, this version was done in serial resulting in less parallelization and horrible cache performance and is this way for simplicity.

When running a parallel program with Cuda, also known as a Kernel, one needs to choose how to map the data to the threads and cores.
A GPU has some number of streaming multiprocessors, SM for short. Each SM has some number of cores, and each core can run some number of threads in parallel.
Its up to the programmer to determine how to map the problem into these devices.
When launching a Kernel, Cuda allows the user to run with a given number of blocks, and some number of threads per block.
In my application, I chose to make the block size a constant and scale the number of threads accordingly.
This should allow the user to tune the program to suite their hardware. The large negative to this approach is it looses the connection to the problem and thus can suffer in cache performance.

##Improvements / TO DO list
###Cache optimization
Much of the GPU implementation is done with no thought towards cache efficiency. It is my estimation that this is the main bottle neck.
Simple things, such as reordering how certain matrices are stored in memory, to more complex matters, should be employed to improve performance.

###Profiling GPU implementation:
I have a feeling the GPU implementation is running at an order of magnitude (or 2) slower than it should.
Profiling and comparing against max stats would be enlightening but sadly not done at this time.

###Python bindings:
C++ is not a very good language for scientific programming and there are not nearly as many tools available for it as say Python.
The library was designed with an intent to be used with sklearn style models so users could leverage the power of existing tools.
Sadly, no work has gone into Python bindings.

###Algorithms:
Currently, this network is fairly restrictive and will not perform very well for certain (most) problems.
It has no support for regularization techniques such as dropout or drop connect.
These techniques help allow the network to generalize more effectively instead of just over fitting to the training data.
In addition to this, the training algorithms used are a little dated. There are faster methods that work off of second derivatives of the changing weight matrices.
The current implementation only works off of the first and thus descends the gradient slower.

###Memory copies:
The current GPU implementation runs in two steps. The first step copies all needed data to the GPU, and the next step runs the GPU programs on that data.
This sequential nature is not good.
Future implementations should use streams. This would allow the program to copy data to the GPU as well as execute on data that has already been copied at the same time, thus reducing these two steps down to the largest of the two.

In addition to this, there are far to many memory copies to and from the GPU to the CPU. These should be limited.

###Memory leaks:
The GPU implementation has a few memory leaks that prevent it from running over an extended period of time.

###Clean up:
Move implementation in hpp files to separate implementation files to leave .hpp files pure. Standardize the api. Rename files. Use more descriptive variable names.

###Further experiments:
One goal I had for this project was to compare a hand-written implementation of a neural net that is very problem specific to one that is built on top of a matrix library.
For speed, the CPU implementation is already built on one such library. I would like to compare my GPU implementation to such an implementation that uses an existing GPU matrix library.

##Gotchas:
###kernel sizes
Due to the nature of my kernel sizing, I had a very difficult to spot error where depending on layer sizes; certain entries in the output matrix were not calculated.
This caused a huge issue because all of my tests appeared to be passing yet the system would not work on a real problem.

Invalid reads and writes in Cuda also gave me a big difficulty.
Originally, my Raw_Matrix class used non overflow checking reads. This caused a great deal of interesting non deterministic results.
Adding in bounds checking saved a large amount of time.
