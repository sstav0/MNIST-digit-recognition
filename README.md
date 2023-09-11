# MNIST-digit-recognition

- `Stavito` is my attempt at coding (from scratch) a neural network designed for data classification into binary or multiple categories. To assess its performance, it must undergo training using a training dataset, followed by accuracy evaluation using a separate test dataset.

- `Perceptron` project represents my attempt at creating (from scratch) a fundamental neural network from the ground up. Through this project, I gained a deeper understanding of the mathematical concepts underlying neural networks, which helped me to code "Stavito".

## Project Requirements 

- To run `Stavito`, you must have those library installed : 
	- `numpy` => for matrix calculations
	- `matplotlib` => to plot sets, log loss graphics and accuracy graphics
	- `tqdm` => to show live progression when executing the code
	- `time` => to display execution time 
	- `idx2numpy` => to convert photos from MNIST set to usable data for "Stavito"

- To run `Perceptron`, you must have those library installed : 
	- `numpy`
	- `matplotlib`
	- `tqdm`
	- `time`
	- `h5py` => to convert photos from set to usable datas for "Perceptron"

## Computing method used 

### `Stavito` 

### For $c$ layers : 
- **Input** : $X$ is a matrix with dimensions $(n^{[0]}\times m)$ where $n^{[0]}$ is the number of variables per data and $m$ is the number of data in the datatset.
**$y$** is a matrix with dimensions $(l \times m)$ where $l$ is the number of classifications possible per data. For the MNIST dataset, $l=10$.

- **Initialisation** : 
$$W^{[c]} \in \mathbb{R}^{n^{[c]}\times n^{[c-1]}}, c \in \mathbb{N}$$
$$b^{[c]} \in \mathbb{R}^{n^{[c]}\times1}, c \in \mathbb{N}$$
- **Forward Propagation**
$$A^{[0]}=X$$
$$Z^{[c]} = W^{[c]} \cdot A^{[c-1]} + b^{[c]}, c\in \mathbb{N}$$
$$A^{[c]}= {1\over 1+e^{-Z^{[c]}} }$$
- **Back-Propagation** 
$$dZ^{[c_f]}=A{[c_f]}-y$$
where $c_f$ is the final layer
$$dW^{[c]}={1 \over m} \times dZ^{[c]} \cdot A^{[c-1]^T}$$
$$db^{[c]}={1 \over m}\sum_{axe 1}dZ^{c]}$$
$$dZ^{[c-1]}=W^{[c]^T} \cdot dZ^{[c]} \cdot A^{[c-1]}(1-A^{[c-1]})$$
- **Update**
$$W^{[c]}=W^{[c]} - \alpha \cdot dW^{[c]}$$
$$b^{[c]}=b^{[c]} - \alpha \cdot db^{[c]}$$

### For the MNIST dataset : 

To convert the pictures of each handwritten number into useful data for the algorithm, I used the `idx2numpy` library. This allowed me to have to obtain the first input matrix $X$ with dimensions $(784 \times 60000)$ which represent $784$ variables with values between $0$ and $255$ (before standardization to get values between $0$ and $1$) that represent each of the pixel on the black and white photo $(28x28)$ for the $60 000$ photos in the MNIST dataset. 
The second input matrix $y$ with dimensions $(10 \times 60000)$ represent the correct number for each of the $60 000$ photos of the dataset under the form of a linear matrix with, for each of the $10$ last neuron a value of $0$ or $1$. For example, if the correct number for the first photo is 8, the first row of the first column of the $y$ matrix will be $(0 0 0 0 0 0 0 0 1 0 0)$. 

## How to use it 

To run the program for the MNIST dataset, just download the project requirements and run the code. If you'd like to use it on another dataset, just read the comments on the code. Everything is explained.Same for `Perceptron`.





