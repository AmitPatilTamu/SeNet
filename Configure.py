# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

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

### END CODE HERE