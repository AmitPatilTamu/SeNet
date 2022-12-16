import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import MyNetwork
from ImageUtils import parse_record, data_augmentation


""" This script defines the training, validation and testing process.
"""
class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.network = MyNetwork(
            self.config['depth'],
            self.config['num_classes'],
            self.config['first_num_filters']
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.closs = nn.CrossEntropyLoss()

        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, training_config, x_valid, y_valid):
	
        self.optimizer = torch.optim.SGD(self.network.parameters(),lr = training_config['learning_rate'], momentum = 0.9, weight_decay =training_config['weight_decay'])
        #self.optimizer = torch.optim.AdamW(self.network.parameters(), lr = training_config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=training_config['weight_decay'], amsgrad=False)
        #self.optimizer = torch.optim.Adam(self.network.parameters(), lr = training_config[learning_rate], betas=(0.9, 0.999), eps=1e-08, weight_decay=training_config[weight_decay] , amsgrad=False)

        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // training_config['batch_size']
        max_epoch = training_config['max_epoch']
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch == 40:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10
            if epoch == 110:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10

            bs = training_config['batch_size']
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                start = i*bs
                end = (i+1) * bs
                batch_x = curr_x_train[start:end,:]
                batch_y = curr_y_train[start:end]
                current_batch = []
                for j in range(training_config['batch_size']):
                    current_batch.append(parse_record(batch_x[j],True))
                current_batch = np.array(current_batch)

                tensor_x = torch.cuda.FloatTensor(current_batch)
                tensor_y = torch.cuda.LongTensor(batch_y)

                outputs = self.network(tensor_x)
                loss = self.closs(outputs,tensor_y)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % training_config['save_interval'] == 0:
                self.save(epoch)
        self.evaluate(x_valid, y_valid)

    def evaluate(self, x, y):
        self.network.eval()
        checkpoint_num_list = self.config['checkpoint_num_list']
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config['model_dir'], 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                x_pro0 = parse_record(x[i], False)
                x_pro1 = parse_record(x[i], False, "shearX")
                x_pro2 = parse_record(x[i], False, "shearY")
                x_pro3 = parse_record(x[i], False, "flipY")
                x_pro4 = parse_record(x[i], False, "flipX")

                x_input0 = np.array([x_pro0])
                x_input1 = np.array([x_pro1])
                x_input2 = np.array([x_pro2])
                x_input3 = np.array([x_pro3])
                x_input4 = np.array([x_pro4])

                x_tensor0 = torch.cuda.FloatTensor(x_input0)
                x_tensor1 = torch.cuda.FloatTensor(x_input1)
                x_tensor2 = torch.cuda.FloatTensor(x_input2)
                x_tensor3 = torch.cuda.FloatTensor(x_input3)
                x_tensor4 = torch.cuda.FloatTensor(x_input4)

                pred0 = self.network(x_tensor0)
                pred1 = self.network(x_tensor1)
                pred2 = self.network(x_tensor2)
                pred3 = self.network(x_tensor3)
                pred4 = self.network(x_tensor4)

                pre = (2*pred0 + pred1 + pred2 + pred3 + pred4)/6
                pre1 = torch.softmax(pre, dim=-1)
                max_elements,pred = torch.max(pre1,1)
                preds.append(pred)

            y = torch.tensor(y)
            preds = torch.tensor(preds)

            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y)/y.shape[0]))

    def predict_prob(self, x):
        self.network.eval()
        checkpoint_num_list = self.config['checkpoint_num_list']
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config['model_dir'], 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = np.ones((x.shape[0], 10))
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                x_pro0 = parse_record(x[i], False)
                x_pro1 = parse_record(x[i], False, "shearX")
                x_pro2 = parse_record(x[i], False, "shearY")
                x_pro3 = parse_record(x[i], False, "flipY")
                x_pro4 = parse_record(x[i], False, "flipX")

                x_input0 = np.array([x_pro0])
                x_input1 = np.array([x_pro1])
                x_input2 = np.array([x_pro2])
                x_input3 = np.array([x_pro3])
                x_input4 = np.array([x_pro4])

                x_tensor0 = torch.cuda.FloatTensor(x_input0)
                x_tensor1 = torch.cuda.FloatTensor(x_input1)
                x_tensor2 = torch.cuda.FloatTensor(x_input2)
                x_tensor3 = torch.cuda.FloatTensor(x_input3)
                x_tensor4 = torch.cuda.FloatTensor(x_input4)

                pred0 = self.network(x_tensor0)
                pred1 = self.network(x_tensor1)
                pred2 = self.network(x_tensor2)
                pred3 = self.network(x_tensor3)
                pred4 = self.network(x_tensor4)

                pre = (2*pred0 + pred1 + pred2 + pred3 + pred4)/6
                pre1 = torch.softmax(pre, dim=-1)
                pred = pre1.cpu().detach().numpy()
                preds[i,:] = pred
                ### END CODE HERE

        return preds
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config['model_dir'], 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config['model_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))