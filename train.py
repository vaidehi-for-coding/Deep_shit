import torch as t
from data import ChallengeDataset
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from trainer import Trainer
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csv_data = pd.read_csv('data.csv', sep=";")
# csv_data = csv_data.iloc[:100, :]

# train_data = ChallengeDataset(csv_data, "train")
# val_data = ChallengeDataset(csv_data, "val")
# to_train, to_validate = train_test_split(train_data, val_data)
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
to_train, to_validate = train_test_split(csv_data, random_state=30, test_size=0.05,
                                         stratify=csv_data[["crack", "inactive"]])

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# train_dl = ChallengeDataset(to_train, mode='train')
# val_test_dl = ChallengeDataset(to_validate, mode='val')
train_dl = t.utils.data.DataLoader(ChallengeDataset(to_train, mode='train'), batch_size=40)
val_test_dl = t.utils.data.DataLoader(ChallengeDataset(to_validate, mode='val'), batch_size=40)

# create an instance of our ResNet model
res_net_1 = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_criteria = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.SGD(res_net_1.parameters(), lr=0.001, momentum=0.9)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(res_net_1, loss_criteria, optim=optimizer, train_dl=train_dl, val_test_dl=val_test_dl, cuda=True,
                  early_stopping_patience=10)

# go, go, go... call fit on trainer
res = trainer.fit(15)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
