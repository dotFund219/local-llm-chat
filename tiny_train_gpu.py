import torch
import torch.nn as nn

# 1) choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 2) generate fake datas: y = 2x + 3 + noise
torch.manual_seed(0)
N = 2048
x = torch.randn(N, 1)
noise = 0.1 * torch.randn(N, 1)
y = 2.0 * x + 3.0 + noise

# 3) go to GPU
x, y = x.to(device), y.to(device)

# 4) Model/Loss/Optimize
model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # lr: learning step

# 5) Loop of learning
for epoch in range(1, 1201):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"epoch {epoch:3d} | loss {loss.item():.6f} | w {w:.4f} | b {b:.4f}")

print("final:", "w=", model.weight.item(), "b=", model.bias.item())