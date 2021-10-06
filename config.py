
# Import packages
import os

init_path = "plastic"
im_path = os.path.sep.join([init_path, "images"])
annots_path = os.path.sep.join([init_path, "annotations"])

init_output = "output"
model_path = os.path.sep.join([init_output, "detector.h5"])
lb_path = os.path.sep.join([init_output, "lb.pickle"])
plots_path = os.path.sep.join([init_output, "plots"])
test_paths = os.path.sep.join([init_output, "test_paths.txt"])

# Initialise number of epochs, learning rate and batch size for training
nber_epochs = 20
lr = 1e-4
batch_size = 32
