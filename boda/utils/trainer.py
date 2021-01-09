from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_loader,
        model,
        optimizer,
        criterion,
        valid_loader: Optional[DataLoader] = None,
        scheduler: Optional[str] = None,
        num_epochs = None,
        num_iterations = None,
        device: str = None
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _init_trainer(self):
        ...

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (images, targets) in enumerate(self.train_loader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                loss = sum(value for value in losses.values())
                loss.backward()

                self.optimizer.step()

                print(f'{epoch:>{len(str(self.num_epochs))}}/{self.num_epochs} | T: {loss::>7.4f}', end=' | ')
                for k, v in losses.items():
                    print(f'{k}: {v.item():>7.4f}', end=' | ')
                print()

    def train_one_step(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError


