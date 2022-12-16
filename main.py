### YOUR CODE HERE
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = MyModel(model_configs).cuda()
	
	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')

		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test)
		np.save(args.save_dir + "/predictions.npy", predictions)
		classes = np.argmax(predictions, axis=1)
		np.save(args.save_dir + "/classes.npy", classes)

### END CODE HERE

