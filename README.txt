SeNet for image classification on CIFAR-10 dataset


libraries REQUIRED-
os, time, numpy, argparse, matplotlib, torch 1.7.0+cu110(this was compatible with my system), pickle, tqdm
===============================================================================================
USAGE-
===============================================================================================
MODE1: Training -
python main.py --mode train --data_dir "C:\Users\amit2\OneDrive\Desktop\project\cifar-10-batches-py\" --save_dir "C:\Users\amit2\OneDrive\Desktop\project\models\"

MODE2: Testing on Public Dataset -
python main.py --mode test --data_dir "C:\Users\amit2\OneDrive\Desktop\project\cifar-10-batches-py\" --save_dir "C:\Users\amit2\OneDrive\Desktop\project\models\"

MODE3: Testing on private dataset -
python main.py --mode predict --data_dir "C:\Users\amit2\OneDrive\Desktop\project\test_data_set\" --save_dir "C:\Users\amit2\OneDrive\Desktop\project\result_final\"

-data_dir - contains the CIFAR-10 dataset(MODE1 and MODE2) or the private training dataset(MODE3) 
-save_dir - directory to save the results of prediction on private dataset(MODE3), no use in other MODES(MODE1 and MODE2)

===============================================================================================
CONFIGS-
===============================================================================================

model_configs = {
	"name": 'MyModel',
	"depth": 18,
	"first_num_filters": 16,
	"num_classes": 10,
	"checkpoint_num_list": [150],
	"model_dir": '..\..\models'
}	

training_configs = {
	"learning_rate": 0.01,
	"batch_size": 64,
	"save_interval": 10,
	"weight_decay": 2e-4,
	"max_epoch":150
}

EXPLANATION-
depth = stack size of each stack of network.
first_num_filters - number of filters in the conv layers in first stack.
checkpoint_num_list - saved checkpoint to run for testing
model_dir - saved model directory


===============================================================================================
CODE DESCRIPTION(as per Project Description) - 
===============================================================================================
• main.py: Includes the code that loads the dataset and performs the training, testing and
prediction.

• DataLoader.py: Includes the code that defines functions related to data I/O.

	-train_valid_split() - splits the training dataset into train and validation datasets.
	-load_testing_images() - loads testing images.
	-load_data() - loads the CIFAR-10 dataset
• ImageUtils.py: Includes the code that defines functions for any (pre-)processing of the
images.

	-data_augmentation() - performs data_augmentation.
	-preprocess_image() - to perform preprocessing of images.
	-visualize() - to visulaize a single test image.
	-parse_record() - parses a record to a image and performs pre-processing.

• Configure.py: Includes dictionaries that set the model configurations, hyper-parameters,
training settings, etc. The dictionaries are imported to main.py

• Model.py: Includes the code that defines the your model in a class. The class is initialized
with the configuration dictionaries
	-train() - function for training.
	-evaluate() - function for calculating accuracy score.
	-predict_prob() - function that returns the probability of each class for a test dataset.

• Network.py: Includes the code that defines the network architecture. The defined network
will be imported and referenced in Model.py.
main classes-
	-MyNetwork -  main classe
	-se_net2 - proposed senet block, actually used in my model
	-se_net - original senet block architecture as described in the referenced paper, not used.
	-standard_block - block instantiating both resnet and squeeze block.
other classes -
	-input_layer, output_layer, batch_norm_celu_layer

===============================================================================================

