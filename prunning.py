import torch

model=torch.load("models/affectnet_emotions/enet_b2_8.pt",map_location=torch.device("cpu"))
model.eval()
#print(model)

def prune_weights(model, threshold):
  for name, param in model.named_parameters():
    if 'weight' in name:
      param.data = torch.where(param.abs() < threshold, torch.tensor(0.), param)

def train_and_prune(model, pruning_schedule):
  print(f'Pruning iteration')
  prune_weights(model, pruning_schedule)

pruning_schedule = 0.01
train_and_prune(model, pruning_schedule)

# Save the pruned model
torch.save(model.state_dict(), 'models/affectnet_emotions/enet_b0_8_best_afew.pt')


