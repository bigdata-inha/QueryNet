import torch
import random
import numpy as np
from agents import *

np.random.seed(42)
random.seed(42)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'
def record_grad(data, label):
    outputs = dict()
    grads = dict()
    channel_importance = dict()
    def save_grad(idx):
        def hook(grad):
            grads[idx] = grad
        return hook

    def cal_importance(grads_list, outputs_list):
        for n, m in agent.named_modules_list.items():
            if isinstance(m, torch.nn.Conv2d):
                grad = grads_list[n]
                output = outputs_list[n]
                importance = (grad * output).mean(dim=(2, 3))
                #total_importance = torch.abs(importance).squeeze()
                #channel_importance[n] += total_importance.data.cpu()
                channel_importance[n] = importance

    agent.optimaizer.zero_grad()
    num_batch = data.shape[0]
    data.requires_grad = True
    x = data.to(device)
    i = 0
    for m in agent.model.features:
        x = m(x)
        if isinstance(m, torch.nn.Conv2d):
            outputs[f'{i}.conv'] = x
            outputs[f'{i}.conv'].register_hook(save_grad(f'{i}.conv'))
            i += 1
    else:
        x = x.view(num_batch, -1)

    x = agent.model.classifier(x)
    one_hot = np.zeros((num_batch, x.shape[-1]), dtype=np.float64)
    one_hot[:, label] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.mean(one_hot.to(device) * x)
    one_hot.backward(retain_graph=True)

    cal_importance(grads, outputs)

    return outputs, grads, channel_importance

agent = Vgg16QueryNet()
agent.model = agent.model.to(device)
agent.loader.make_one_class_set(sub_class=5, batch_size=8)
data, labels = next(iter(agent.loader.sub_train_loader))

data1, labels1 = data[:1], labels[:1]
data2, labels2 = data[1:2], labels[1:2]
data0, labels0 = data[:2], labels[:2]

outputs1, grads1, channel_importances1 = record_grad(data1, labels1)
print(channel_importances1['0.conv'].argsort())

outputs1, grads1, channel_importances1 = record_grad(data1, labels1)
print(channel_importances1['0.conv'].argsort())

# outputs, grads, channel_importances = record_grad(data, labels)
