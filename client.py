import time

import torch
import utils
from torch.utils.data import DataLoader
import torch.nn as nn


class Client:
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args

        # Set up local dataset and dataloader for the client
        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.local_bs,
            shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        self.n_data = len(self.train_dataset)
        
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, local_model, round=None):
        """ Do a local training over the received global model """

        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.args.local_lr,
            weight_decay=1e-4,
            momentum=0.9
        )

        for epoch in range(self.args.local_epochs):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):

                optimizer.zero_grad()
                inputs, labels = inputs.to(
                    device=self.args.device, non_blocking=True
                ), labels.to(device=self.args.device, non_blocking=True)
                outputs = local_model(inputs)
                
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
    
                minibatch_loss = self.criterion(logits, labels)
                minibatch_loss.backward()
                optimizer.step()

            end = time.time()
            train_time = end - start
            print(
                "global round: %d/%d \t local epoch: %d \t client ID: %d \t loss: %.8f \t time: %.2f"
                % (
                    round,
                    self.args.training_rounds,
                    epoch + 1,
                    self.id,
                    minibatch_loss,
                    train_time,
                )
            )
