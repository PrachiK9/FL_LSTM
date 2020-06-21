
import numpy as np
from keras import backend as K
#define the training proces
import torch.nn.functional as F


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epoch = 20
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = True
        self.seed = 1
        self.log_interval = 200
        self.save_model = False

args = Arguments()


LOG_INTERVAL = 5
BATCH_SIZE = 100
EPOCHS = 20


def one_hot(labels):
	k=list(set(labels))
	n_class=len(k)
	l=np.zeros((len(labels),n_class))
	for i in range(len(labels)):
		lb=labels[i]
		idx=k.index(lb)
		l[i,idx]=1
	return l
def mcor(y_true, y_pred):
	#matthews_correlation
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos
	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos
	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)
	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)
	numerator = (tp * tn - fp * fn)
	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	return numerator / (denominator + K.epsilon())
def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall
	def precision(y_true, y_pred):

		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall))

def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()
    # Iterate through each gateway's dataset
    for idx, (data, target) in enumerate(federated_train_loader):
        batch_idx = idx+1
        # Send the model to the right gateway
        model.send(data.location)
        # Move the data and target labels to the device (cpu/gpu) for computation
        data, target = data.to(device), target.to(device)
        # Clear previous gradients (if they exist)
        optimizer.zero_grad()
        # Make a prediction
        output = model(data)
        # Calculate the cross entropy loss [We are doing classification]
        loss = F.cross_entropy(output, target)
        # Calculate the gradients
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Get the model back from the gateway
        model.get()
        if batch_idx==len(federated_train_loader) or (batch_idx!=0 and batch_idx % LOG_INTERVAL == 0):
            # get the loss back
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(federated_train_loader) * BATCH_SIZE,
                100. * batch_idx / len(federated_train_loader), loss.item()))
    return model