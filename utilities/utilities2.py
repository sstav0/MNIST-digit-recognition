import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def load_data():
	imagefile = 'datasets/nmbrs/train-images.idx3-ubyte'
	imagefileTest = 'datasets/nmbrs/t10k-images.idx3-ubyte'
	labelfile = 'datasets/nmbrs/train-labels.idx1-ubyte'
	labelfileTest = 'datasets/nmbrs/t10k-labels.idx1-ubyte'
	imagearray = idx2numpy.convert_from_file(imagefile)
	imagearrayTest = idx2numpy.convert_from_file(imagefileTest)
	labelarray = idx2numpy.convert_from_file(labelfile)
	labelarrayTest = idx2numpy.convert_from_file(labelfileTest)
	
	return (imagearray, labelarray, imagearrayTest, labelarrayTest) 

