from dataio import DataIO, ImageHandler
from unet_model import UNet
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# Solving CUDNN Issues
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = UNet('Test_Model')
model.create_UNet_retina()

test_img, test_label = DataIO().load_matfile_images_first('retina_test.mat')
totaldices = []
epochs = []

# Write your model name here
model.load_model('101')

out = model.produce_ouputs_mat(test_img[:, :, :, :], test_label[:, :, :, :])
i = ImageHandler()
dices = []
for e in range(len(out)):
    i.display_image(test_img[e, :, :, :].squeeze(), title='original image ' + str(e))
    i.display_image(out[e], title='model output '+ str(e))
    i.display_image(test_label[e, :, :, :], title='original label '+str(e))



