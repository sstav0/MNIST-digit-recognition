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

### For $C$ layers : 
- **Input** : 
	- $X$ : This is a matrix with dimensions of $(n^{[0]}\times m)$, where $n^{[0]}$ represents the number of variables for each piece of data, and $m$ is the total number of data points in the dataset. 
	- $y$ : This matrix has dimensions of $(l \times m)$, where $l$ denotes the number of possible classifications for each data point. In the case of the MNIST dataset, there are 10 possible classifications, so $l=10$.


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
$$db^{[c]}={1 \over m}\sum_{axe 1}dZ^{[c]}$$
$$dZ^{[c-1]}=W^{[c]^T} \cdot dZ^{[c]} \cdot A^{[c-1]}(1-A^{[c-1]})$$
- **Update**
$$W^{[c]}=W^{[c]} - \alpha \cdot dW^{[c]}$$
$$b^{[c]}=b^{[c]} - \alpha \cdot db^{[c]}$$

### For the MNIST dataset : 


To prepare the handwritten number images for algorithmic processing, I utilized the `idx2numpy` library. This step allowed me to create two input matrices: Matrix X (Dimensions: ($784 \times 60000$)): This matrix represents $60 000$ images, each measuring $28 \times 28$ pixels. Initially, it contains $784$ variables, each with values ranging from $0$ to $255$ (prior to standardization, which transforms them into values between $0$ and $1$). These variables correspond to the individual pixel values within each black-and-white photo.

## How to use it 

To run the program for the MNIST dataset, just download the project requirements and run the code. If you'd like to use it on another dataset, just read the comments on the code. Everything is explained.Same for `Perceptron`.





