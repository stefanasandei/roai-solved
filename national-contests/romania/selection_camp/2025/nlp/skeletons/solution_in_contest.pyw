# %%
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torchmetrics

# %%
seed = 333
torch.random.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device="cuda"

batch_size = 4

# %% [markdown]
# # Data prep

# %%
class MovementDataset(Dataset):
    def __init__(self, df_path: str):
        self.df = pd.read_csv(df_path)

        self.unique_ids = self.df["IDSample"].nunique()
        self.start_idx = self.df.iloc[0]["IDSample"]

    def len_frames(self, idx):
        row_idx = int(idx + self.start_idx)
        rows = self.df[self.df["IDSample"] == row_idx]
        return len(rows)
    
    def __getitem__(self, idx):
        row_idx = int(idx + self.start_idx)
        rows = self.df[self.df["IDSample"] == row_idx]

        cols = [f"J{i}X" for i in range(1, 25+1)] + [f"J{i}Y" for i in range(1, 25+1)] + [f"J{i}Z" for i in range(1, 25+1)]
        data = rows[cols].values.reshape(-1, 25 * 3)
        data = torch.tensor(data, dtype=torch.float32)

        if "Camera" in rows:
            labels = rows.iloc[0][["Camera", "Action"]].values
            labels[0] -= 1
            labels = torch.tensor(labels, dtype=torch.long).reshape(1, 2)

            return data, labels
        return data
    
    def __len__(self):
        return self.unique_ids

# %%
def collate(x):
    data, labels = [], []
    for i in range(len(x)):
        data.append(x[i][0])
        labels.append(x[i][1])
    data = pad_sequence(data, batch_first=True)
    labels = torch.cat(labels, dim=0)
    return data, labels

# %%
dataset_train = MovementDataset("train_data.csv")
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

# %%
batch = next(iter(dataloader_train))

# (batch_size, seq_len, 75); (batch_size, 2)
batch[0].shape

# %% [markdown]
# # Model selection

# %%
class JointNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_hidden = 128
        self.n_layers = 2

        self.lstm = nn.LSTM(input_size=75, hidden_size=self.n_hidden, num_layers=self.n_layers, batch_first=True)

        self.camera_head = nn.LazyLinear(3)

        self.action_head = nn.LazyLinear(5)

    def forward(self, x):
        # x: (batch_size, seq_len, n_hidden)
        batch_size, seq_len = x.shape[0], x.shape[1]

        h0 = torch.randn(self.n_layers, batch_size, self.n_hidden, device=device)
        c0 = torch.randn(self.n_layers, batch_size, self.n_hidden, device=device)
        features, _ = self.lstm(x, (h0, c0))
        features = torch.mean(features, dim=1)

        features = features.view(batch_size, -1)

        camera_pred = self.camera_head(features)
        action_pred = self.action_head(features)

        return camera_pred, action_pred

# %%
model = JointNet().to(device)

b = batch[0].to(device)
print(b.shape)
model(b)[1].shape

# %%
model = model.to(device)

# %% [markdown]
# # Training

# %%
should_train = True

# %%
lr = 5e-4
epochs = 35
losses = []

action_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5).to(device)

criterion_action = nn.CrossEntropyLoss() # 5

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, eta_min=1e-6)

# %%
if should_train:
    print("Training the model!")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (data, labels) in enumerate(tqdm(dataloader_train)):
            data, labels = data.to(device), labels.to(device)

            # forward pass
            _, logits_action = model(data)
            loss_action = criterion_action(logits_action, labels[:, 1])
            loss = loss_action

            action_accuracy(logits_action, labels[:, 1])

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            scheduler.step((0+epoch) + i / len(dataloader_train))

            # stats
            running_loss += loss.item()
            losses.append(loss.item())
        
        l = running_loss / len(dataloader_train)
        action_acc = action_accuracy.compute()

        print(f"Epoch {epoch}, loss={l:.2f}, action_acc={(action_acc*100):.1f}%")
    torch.save(model.state_dict(), "sol-action.pth")
else:
    print("Loading the model!")

    model.load_state_dict(torch.load("sol-action.pth"))

# %%
plt.plot(losses)

# %% [markdown]
# now train for the camera

# %%
lr = 2e-4
epochs = 80
losses = []

cam_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)

model_cam = JointNet().to(device)

criterion_camera = nn.CrossEntropyLoss() # 3

optimizer_cam = AdamW(model_cam.parameters(), lr=lr)
scheduler_cam = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_cam, T_0=80, eta_min=1e-6)

# %%
if should_train:
    print("Training the model!")
    for epoch in range(epochs):
        model_cam.train()
        running_loss = 0.0
        for i, (data, labels) in enumerate(tqdm(dataloader_train)):
            data, labels = data.to(device), labels.to(device)

            # # forward pass
            logits_camera, _ = model_cam(data)
            loss_camera = criterion_camera(logits_camera, labels[:, 0])
            loss = loss_camera

            cam_accuracy(logits_camera, labels[:, 0])

            # # backward pass
            optimizer_cam.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_cam.parameters(), 2.5)
            optimizer_cam.step()
            scheduler_cam.step((0+epoch) + i / len(dataloader_train))

            # stats
            running_loss += loss.item()
            losses.append(loss.item())
        
        l = running_loss / len(dataloader_train)
        cam_acc = cam_accuracy.compute()

        print(f"Epoch {epoch}, loss={l:.2f}, cam_acc={(cam_acc*100):.1f}%")
    torch.save(model_cam.state_dict(), "sol-camera.pth")
else:
    print("Loading the model!")

    model_cam.load_state_dict(torch.load("sol-camera.pth"))

# %% [markdown]
# # Submission

# %%
dataset_test = MovementDataset("test_data.csv")

df_test = pd.read_csv("test_data.csv")
ids = df_test["IDSample"].unique()

# %%
subtask1 = []
for i in range(len(ids)):
    subtask1.append(dataset_test.len_frames(i))

# %%
subtask2, subtask3 = [], []

for i in tqdm(range(len(ids))):
    b = dataset_test[i].unsqueeze(0).to(device)
    with torch.no_grad():
        o = model(b)
        o_cam = model_cam(b)
        action = o[1]
        camera = o_cam[0]
    act = F.softmax(action, dim=-1)
    act = torch.argmax(action)
    subtask2.append(act.detach().item())
    
    camera = F.softmax(camera, dim=-1)
    camera = torch.argmax(camera)+1
    subtask3.append(camera.detach().item())

# %%
subtask1 = pd.DataFrame({
    "datapointID": ids,
    "answer": subtask1,
    "subtaskID": 1
})

subtask2 = pd.DataFrame({
    "datapointID": ids,
    "answer": subtask2,
    "subtaskID": 2
})

subtask3 = pd.DataFrame({
    "datapointID": ids,
    "answer": subtask3,
    "subtaskID": 3
})

# %%
subtask2["answer"].value_counts()

# %%
submission = pd.concat([subtask1, subtask2, subtask3])
submission.head()

# %%
submission.to_csv("submission.csv", index=False)


