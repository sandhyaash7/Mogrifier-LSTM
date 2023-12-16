import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
from torch import optim
from scipy.io import loadmat
import matplotlib.pyplot as plt

### DATA LOADER ###
data = loadmat("/content/data.mat")["S1_nolabel6"]
X, Y = data[:28000, :-1], data[:28000, -1]
print(X.max())
class EEGDataset(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    x = torch.tensor(self.X[idx, :]).type(torch.FloatTensor)
    y = torch.tensor(self.Y[idx] - 1).type(torch.LongTensor)

    return x, y


dataset = EEGDataset(X, Y)
trainset, testset = random_split(dataset, [21000, 7000])

### Define Model ###
class MogrifierLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r
    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0:
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct

class MOGEEGModel(nn.Module):
  def __init__(self, batch_size, hidden_size, device="cpu"):
    super().__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.device = device

    self.linear1 = nn.Linear(64,hidden_size)
    self.linear2 = nn.Linear(64,64)
    self.linear3 = nn.Linear(64,64)
    self.linear4 = nn.Linear(64,64)

    self.moglstm = MogrifierLSTMCell(input_size=1,
                                      hidden_size=self.hidden_size,
                                      mogrify_steps=5)


    self.pred = nn.Linear(64, 5)

  def forward(self, x):
    ## Hidden Linear Layers ###
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = F.relu(self.linear4(x))

    x = x.unsqueeze(-1)
    ht,ct = torch.zeros(self.batch_size, self.hidden_size).to(self.device), torch.zeros(self.batch_size, self.hidden_size).to(self.device)

    ### Pass through LSTM ###
    for t in range(64):
      x_t = x[:, t, :]
      ht, ct = self.moglstm(x_t, (ht, ct))

    x = self.pred(ht)
    return x

class EEGModel(nn.Module):
  def __init__(self):
    super().__init__()
    #self.batch_size = batch_size
    #self.hidden_size = hidden_size
    #self.device = device

    self.linear1 = nn.Linear(64,hidden_size)
    self.linear2 = nn.Linear(64,64)
    self.linear3 = nn.Linear(64,64)
    self.linear4 = nn.Linear(64,64)

    self.lstm = nn.LSTM(input_size=1,
                        hidden_size=64,
                        num_layers=2,
                        batch_first=True)


    self.pred = nn.Linear(64, 5)

  def forward(self, x):
    ## Hidden Linear Layers ###
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = F.relu(self.linear4(x))

    ### Unsqueeze and add 1 to last dimention for LSTM happiness ###
    x = x.unsqueeze(-1)

    ### Pass through LSTM ###
    x, (_,_) = self.lstm(x)

    ### Grab the last token ###
    x = x[:, -1, :]

    x = self.pred(x)
    return x

    batch_size = 128

### SELECT DEVICE ###
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on Device {device}")

### Load Model ###
hidden_size=64
model = MOGEEGModel(batch_size=batch_size, hidden_size=hidden_size, device=device).to(device)
#model = EEGModel().to(device)

### MODEL TRAINING INPUTS ###
epochs = 50
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()



trainloader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
valloader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=True)

### Reference ###
# https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/PyTorch%20for%20NLP/IMDB%20Classification/Sequence%20Classification.ipynb
def train(model, device, epochs, optimizer, loss_fn, batch_size, trainloader, valloader):
    log_training = {"epoch": [],
                    "training_loss": [],
                    "training_acc": [],
                    "validation_loss": [],
                    "validation_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}")
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []

        model.train() # Turn On BatchNorm and Dropout
        for image, label in tqdm(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            out = model.forward(image)

            ### CALCULATE LOSS ##
            loss = loss_fn(out, label)
            training_losses.append(loss.item())

            ### CALCULATE ACCURACY ###
            predictions = torch.argmax(out, axis=1)
            accuracy = (predictions == label).sum() / len(predictions)
            training_accuracies.append(accuracy.item())

            loss.backward()

            ### Just Incase of Exploding Gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        for image, label in tqdm(valloader):
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                out = model.forward(image)

                ### CALCULATE LOSS ##
                loss = loss_fn(out, label)
                validation_losses.append(loss.item())

                ### CALCULATE ACCURACY ###
                predictions = torch.argmax(out, axis=1)
                accuracy = (predictions == label).sum() / len(predictions)
                validation_accuracies.append(accuracy.item())

        training_loss_mean, training_acc_mean = np.mean(training_losses), np.mean(training_accuracies)
        valid_loss_mean, valid_acc_mean = np.mean(validation_losses), np.mean(validation_accuracies)

        log_training["epoch"].append(epoch)
        log_training["training_loss"].append(training_loss_mean)
        log_training["training_acc"].append(training_acc_mean)
        log_training["validation_loss"].append(valid_loss_mean)
        log_training["validation_acc"].append(valid_acc_mean)

        print("Training Loss:", training_loss_mean)
        print("Training Acc:", training_acc_mean)
        print("Validation Loss:", valid_loss_mean)
        print("Validation Acc:", valid_acc_mean)

    return log_training, model


training_logging, model = train(model=model,
                                device=device,
                                epochs=epochs,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                batch_size=batch_size,
                                trainloader=trainloader,
                                valloader=valloader)

n = np.arange(1,epochs+1)

plt.plot(n, training_logging["training_loss"], label="Training Loss")
plt.plot(n, training_logging["validation_loss"], label="Validation Loss")
plt.title("Training Loss Results")
plt.legend()
plt.show()


plt.plot(n, training_logging["training_acc"], label="Training Acc")
plt.plot(n, training_logging["validation_acc"], label="Validation Acc")
plt.title("Training Accuracy Results")
plt.legend()
plt.show()

### Save Model ###
torch.save(model.state_dict(), "/content/model_save.pt")

for name, param in model.named_parameters(): # Can iterate through all the layers and see the names and parameters
  print(name)
  print(param)
  #break
