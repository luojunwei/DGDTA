import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.gatv2 import GATv2Model
from models.gatv2GCN import GATv2GCNModel
from utils import *


def train_epoch(model, device, data_loader, optimizer, epoch, log_interval):
    print(f'Training on {len(data_loader.dataset)} samples...')
    model.train()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(data_loader.dataset)} '
                  f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate_model(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Make prediction for {len(data_loader.dataset)} samples...')
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    datasets = ['kiba']
    model_class = GATv2GCNModel
    model_name = model_class.__name__

    cuda_device = "cuda:0"
    train_batch_size = 512
    test_batch_size = 512
    learning_rate = 0.0005
    log_interval = 20
    num_epochs = 1000

    print('Learning rate:', learning_rate)
    print('Epochs:', num_epochs)

    for dataset in datasets:
        print(f'\nRunning on {model_name}_{dataset}')
        train_file = f'data/processed/{dataset}_train.pt'
        test_file = f'data/processed/{dataset}_test.pt'
        if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
            print('Please run create_data.py to prepare data in PyTorch format!')
            continue

        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')

        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = random_split(train_data, [train_size, valid_size])

        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

        device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        model = model_class().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_valid_mse = float('inf')
        best_test_mse = float('inf')
        best_test_ci = 0
        best_epoch = -1
        model_file = f'model_{model_name}_{dataset}.model'
        result_file = f'result_{model_name}_{dataset}.csv'

        for epoch in range(num_epochs):
            train_epoch(model, device, train_loader, optimizer, epoch + 1, log_interval)
            print('Predicting for validation data')
            valid_labels, valid_preds = evaluate_model(model, device, valid_loader)
            valid_mse = mse(valid_labels, valid_preds)

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file)
                print('Predicting for test data')
                test_labels, test_preds = evaluate_model(model, device, test_loader)
                test_metrics = [rmse(test_labels, test_preds), mse(test_labels, test_preds), 
                                pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds)]
                with open(result_file, 'w') as f:
                    f.write(','.join(map(str, test_metrics)))
                best_test_mse = test_metrics[1]
                best_test_ci = test_metrics[-1]
                print(f'RMSE improved at epoch {best_epoch}; best_test_mse, best_test_ci: {best_test_mse}, {best_test_ci}, {model_name}, {dataset}')
            else:
                print(f'{valid_mse} No improvement since epoch {best_epoch}; best_test_mse, best_test_ci: {best_test_mse}, {best_test_ci}, {model_name}, {dataset}')

if __name__ == "__main__":
    main()