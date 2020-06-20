
import numpy as np
from keras import backend as K
import json

def rnn_model(n_class, data_size,window_size):

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


	def train_rnn(X_train, y_train,X_test,y_test, models,window_size,):


		for model_name in models.keys():
			model=models[model_name]
			hist_RNN=model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs =200, batch_size =5000,verbose=1)#80 train 20 test
			with open('hist-'+model_name+"["+str(window_size)+'].json', 'w') as f:
				json.dump(hist_RNN.history, f)
			model.save(model_name+"["+str(window_size)+"].h5")
