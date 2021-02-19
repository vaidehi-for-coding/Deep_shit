import torch as t
from sklearn.metrics import f1_score
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.actual_cracks = []
        self.actual_inactives = []
        self.pred_cracks = []
        self.pred_inactives = []
        self.f1_scores_cracks = []
        self.f1_scores_inactives = []

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        # m = self._model.cuda()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # TODO
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model.forward(x)
        # -calculate the loss
        loss = self._crit.forward(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        # TODO
        # predict
        prediction = self._model.forward(x).cuda()
        # prediction = self._model.forward(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit.forward(prediction, y)
        # return the loss and the predictions
        return loss, prediction

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        self._model.train().cuda()
        # self._model.train(mode=True)
        epoch_loss = 0.0

        for x, y in self._train_dl:
            if self._cuda is True:
                self._model.cuda()
                x = x.cuda()
                y = y.cuda()
            epoch_loss += self.train_step(x, y)
        # print("Epoch loss: ", epoch_loss)

        average_loss = epoch_loss / len(self._train_dl)

        # print(average_loss.item())

        return average_loss.item()

    def val_test(self):
        # TODO
        # set eval mode
        self._model.eval()
        total_loss = 0
        batch_predictions = []
        batch_labels = []
        # disable gradient computation - what?
        t.no_grad()
        # iterate through the validation set
        for x, y in self._val_test_dl:
            # transfer the batch to the gpu if given
            if self._cuda is True:
                self._model.cuda()
                x = x.cuda()
                y = y.cuda()
            # perform a validation step
            y = y.type(t.float)
            loss, prediction = self.val_test_step(x, y)
            total_loss += loss
            # print(prediction, y)
            # save the predictions and the labels for each batch
            batch_predictions.append(prediction.cpu().detach().numpy())
            batch_labels.append(y.cpu().detach().numpy())

        for i in range(len(self._val_test_dl)):
            b_pred = batch_predictions[i]
            b_label = batch_labels[i]
            #bp_encoded = ((b_pred > 0.5).cpu() * t.tensor([1]))
            #bl_encoded = (b_label.cpu() * t.tensor([1]))
            bp_encoded = ((b_pred > 0.5))
            bl_encoded = (b_label)
            bp_encoded_cracks = bp_encoded[:, 0]
            bp_encoded_inactive = bp_encoded[:, 1]
            bl_encoded_cracks = bl_encoded[:, 0]
            bl_encoded_inactive = bl_encoded[:, 1]
            self.actual_cracks += bl_encoded_cracks.tolist()
            self.actual_inactives += bl_encoded_inactive.tolist()
            self.pred_cracks += bp_encoded_cracks.tolist()
            self.pred_inactives += bp_encoded_inactive.tolist()

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in
        # designated functions
        average_loss = total_loss / len(self._val_test_dl)

        # return the loss and print the calculated metrics
        return average_loss.item()

    def fit(self, epochs=-1):
        print("Start training")
        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        epoch_train_losses = []
        epoch_val_losses = []
        patience_count = 0

        for i in range(epochs):
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            train_loss = self.train_epoch()
            epoch_train_losses.append(train_loss)
            epoch_val_loss = self.val_test()
            print(f"Epoch: {i + 1} | Train Loss: {train_loss} | Val Loss: {epoch_val_loss}")
            epoch_val_losses.append(epoch_val_loss)
            self.save_checkpoint(i)
            print(f"Model saved at Epoch: {i}")

            if i != 0 and epoch_val_losses[i] > epoch_val_losses[i - 1]:
                print(f"Loss not improved: Patience at {patience_count}")
                patience_count += 1
            else:
                patience_count = 0

            if patience_count >= self._early_stopping_patience:
                print("Early Stopping -> Stop training")
                return epoch_train_losses, epoch_val_losses

        f1_cracks = f1_score(self.actual_cracks, self.pred_cracks, average='macro')
        f1_inactives = f1_score(self.actual_inactives, self.pred_inactives, average='macro')
        self.f1_scores_cracks.append(f1_cracks)
        self.f1_scores_inactives.append(f1_inactives)
        f1_scores_cracks = np.array(self.f1_scores_cracks)
        f1_scores_inactives = np.array(self.f1_scores_inactives)
        f1_cracks_mean = np.mean(f1_scores_cracks)
        f1_inactives_mean = np.mean(f1_scores_inactives)
        f1_mean = np.mean((f1_cracks_mean, f1_inactives_mean))

        print("F1-MEAN-SCORE: ", f1_mean)

        return epoch_train_losses, epoch_val_losses
