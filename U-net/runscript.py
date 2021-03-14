from dataio import DataIO, ImageHandler
from unet_model import UNet
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

# Solving CUDNN Issues
config = ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)

# create all instances here
dat = DataIO().load_matfile_images_first('retina_training_STARE.mat')
display = ImageHandler()

# Run U-Net Model
model = UNet('Test_Model')
model.create_UNet_retina()
model.fit_model(*dat, nepochs=2000)


# get plot for training accuracy and loss
plot1 = model.plot_accuracy()
plot1.plot()
plot1.show()
plot2 = model.plot_loss()
plot2.plot()
plot2.show()
