# Import Packages
import pickle
from data import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input

# MODEL PREPARATION
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False
flatten = vgg.output
flatten = Flatten()(flatten)

bound_box_head = Dense(128, activation="relu")(flatten)
bound_box_head = Dense(64, activation="relu")(bound_box_head)
bound_box_head = Dense(32, activation="relu")(bound_box_head)
bound_box_head = Dense(4, activation="sigmoid", name="bounding_box")(bound_box_head)

softmax_head = Dense(512, activation="relu")(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation="relu")(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmax_head)

model = Model(inputs=vgg.input, outputs=(bound_box_head, softmax_head))

# Loss
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error"
}

loss_wghts = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

optimizer = Adam(lr=config.lr)
model.compile(loss=losses, optimizer=optimizer, metrics=["accuracy"],
              loss_weights=loss_wghts)
print(model.summary())

train_targets = {
    "class_label": train_lbls,
    "bounding_box": train_bound_boxes
}

test_targets = {
    "class_label": test_lbls,
    "bounding_box": test_bound_boxes
}

# PREDICTION

print("[INFO] training model...")
H = model.fit(
    train_im,
    train_targets,
    validation_data=(test_im, test_targets),
    batch_size=config.batch_size,
    epochs=config.nber_epochs,
    verbose=1)

print("[INFO] saving object detector model...")
model.save(config.model_path, save_format="h5")

print("[INFO saving label binarizer...")
f = open(config.lb_path, "wb")
f.write(pickle.dumps(lb))
f.close()
