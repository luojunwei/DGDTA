import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gatv2 import GATv2Model
from models.gatv2GCN import GATv2GCNModel
from utils import *

def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval):
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    datasets = ['kiba'] #davis
    modeling = GATv2GCNModel    #GATv2Model
    model_st = modeling.__name__

    cuda_name = "cuda:0" if len(sys.argv) <= 3 else f"cuda:{int(0)}"
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000

    print('Learning rate:', LR)
    print('Epochs:', NUM_EPOCHS)

    for dataset in datasets:
        print(f'\nrunning on {model_st}_{dataset}')
        processed_data_file_train = f'data/processed/{dataset}_train.pt'
        processed_data_file_test = f'data/processed/{dataset}_test.pt'
        if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
            print('please run create_data.py to prepare data in pytorch format!')
            continue

        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_mse = float('inf')
        best_ci = 0
        best_epoch = -1
        model_file_name = f'model_{model_st}_{dataset}.model'
        result_file_name = f'result_{model_st}_{dataset}.csv'

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1, loss_fn, LOG_INTERVAL)
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                print(f'rmse improved at epoch {best_epoch}; best_mse,best_ci: {best_mse}, {best_ci}, {model_st}, {dataset}')
            else:
                print(f'{ret[1]} No improvement since epoch {best_epoch}; best_mse,best_ci: {best_mse}, {best_ci}, {model_st}, {dataset}')

            results_df = results_df.append({'epoch': epoch + 1, 'rmse': ret[0], 'mse': ret[1], 
                                            'pearson': ret[2], 'spearman': ret[3], 'ci': ret[4]}, 
                                           ignore_index=True)
        results_df.to_csv(result_file_name, index=False)

if __name__ == "__main__":
    main()