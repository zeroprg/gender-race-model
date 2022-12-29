import tensorflow.keras
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(30, 30, 3), name='Input_1')
	BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(Input_1)
	Conv2D_1 = Conv2D(name='Conv2D_1',filters= 32,kernel_size= 3,padding= 'same' ,activation= 'relu' )(BatchNormalization_1)
	Conv2D_4 = Conv2D(name='Conv2D_4',filters= 32,kernel_size= 3,activation= 'relu' )(Conv2D_1)
	MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1',pool_size= (2,2))(Conv2D_4)
	Dropout_1 = Dropout(name='Dropout_1',rate= 0.25)(MaxPooling2D_1)
	Conv2D_2 = Conv2D(name='Conv2D_2',filters= 64,kernel_size= 3,padding= 'same' ,activation= 'relu' )(Dropout_1)
	Conv2D_3 = Conv2D(name='Conv2D_3',filters= 64,kernel_size= 3,activation= 'relu' )(Conv2D_2)
	MaxPooling2D_2 = MaxPooling2D(name='MaxPooling2D_2',pool_size= (2,2))(Conv2D_3)
	Dropout_2 = Dropout(name='Dropout_2',rate= 0.25)(MaxPooling2D_2)
	Flatten_1 = Flatten(name='Flatten_1')(Dropout_2)
	Dense_1 = Dense(name='Dense_1',units= 512,activation= 'relu' )(Flatten_1)
	Dropout_3 = Dropout(name='Dropout_3',rate= 0.5)(Dense_1)
	Dense_2 = Dense(name='Dense_2',units= 6,activation= 'softmax' )(Dropout_3)

	model = Model([Input_1],[Dense_2])
	return aliases, model


from tensorflow.keras.optimizers import *

def get_optimizer():
	return SGD(decay=1e-6,momentum=0.9,nesterov=True)

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 35

def get_data_config():
	return '{"mapping": {"name": {"type": "Image", "port": "InputPort0", "shape": "", "options": {"pretrained": "None", "Augmentation": false, "rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "shear_range": 0, "horizontal_flip": false, "vertical_flip": false, "Scaling": 1, "Normalization": true, "Resize": true, "Width": "30", "Height": "30"}}, "rate": {"type": "Categorical", "port": "OutputPort0", "shape": "", "options": {}}}, "numPorts": 1, "samples": {"training": 3235, "validation": 808, "test": 0, "split": 1}, "dataset": {"name": "races.dataset", "type": "private", "samples": 4044}, "datasetLoadOption": "batch", "shuffle": true, "kfold": 1}'