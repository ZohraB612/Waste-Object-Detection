# Import packages
from model import *
import matplotlib.pyplot as plt

# PLOT
# LOSS
loss_names = ["loss", "class_lbl_loss", "bound_box_loss"]
N = np.arange(0, config.nber_epochs)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(loss_names):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], kabel="val_" + l)
    ax[i].legend()

plt.tight_layout()
plotPath = os.path.sep.join([config.plots_path, "losses.png"])
plt.savefig(plotPath)
plt.close()

# ACCURACIES
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.plots_path, "accuracies.png"])
plt.savefig(plotPath)
