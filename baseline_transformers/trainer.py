from typing import Dict

import torch
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from tqdm import tqdm

from model import ModelForClassification


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.n_epochs = config['n_epochs']
        self.optimizer = None
        self.opt_fn = lambda model: AdamW(
            model.parameters(), 
            config['lr'], 
            weight_decay=config['weight_decay']
        )
        self.model = None
        self.history = None
        self.loss_fn = CrossEntropyLoss(weight=self.config.get('weight', None), label_smoothing=self.config['label_smoothing'])
        self.device = config['device']
        self.verbose = config.get('verbose', True)
        self.num_steps = config.get('num_steps', None)

    def fit(self, model, train_dataloader, val_dataloaders):
        self.model = model.to(self.device)
        self.optimizer = self.opt_fn(model)
        from collections import defaultdict
        self.history = defaultdict(list)

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            train_info = self.train_epoch(train_dataloader)
            self.history['train_loss'].extend(train_info['loss'])

            for name, val_dataloader in val_dataloaders.items():
                val_info = self.val_epoch(val_dataloader)
                self.history[f'{name}_loss'].extend([val_info['loss']])
                self.history[f'{name}_acc'].extend([val_info['acc']])
                
        return self.model.eval()

    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []
        # if self.verbose:
        #     train_dataloader = tqdm(train_dataloader)
        train_dataloader = tqdm(train_dataloader) 
        for ind, batch in enumerate(train_dataloader ):
            ids = batch['ids'].to(self.device, dtype=torch.long)
            mask = batch['mask'].to(self.device, dtype=torch.long)
            targets = batch['targets'].to(self.device, dtype=torch.long)

            with torch.autocast('cuda', enabled=True):
                outputs = self.model(ids, mask)
                loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            if self.verbose:
                train_dataloader.set_description(f"Loss={loss_val:.3}")
            losses.append(loss_val)
            if self.num_steps is not None and ind > self.num_steps:
                break
            
        return {'loss': losses}

    def val_epoch(self, val_dataloader):
        self.model.eval()
        all_logits = []
        all_labels = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)
        with torch.no_grad():
            with torch.autocast('cuda', enabled=True):
                for batch in val_dataloader:
                    ids = batch['ids'].to(self.device, dtype=torch.long)
                    mask = batch['mask'].to(self.device, dtype=torch.long)
                    targets = batch['targets'].to(self.device, dtype=torch.long)
                    outputs = self.model(ids, mask)
                    all_logits.append(outputs)
                    all_labels.append(targets)
        all_labels = torch.cat(all_labels).to(self.device)
        all_logits = torch.cat(all_logits).to(self.device)
        loss = self.loss_fn(all_logits, all_labels).item()
        acc = (all_logits.argmax(1) == all_labels).float().mean().item()
        print(acc)
        if self.verbose:
            val_dataloader.set_description(f"Loss={loss:.3}; Acc:{acc:.3}")
        return {
            'acc': acc,
            'loss': loss,
            'logits': all_logits,
            'labels': all_labels
        }

    def predict(self, test_dataloader):
        if not self.model:
            raise RuntimeError("You should train the model first")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_dataloader:
                ids = batch['ids'].to(self.device, dtype=torch.long)
                mask = batch['mask'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask)
                predictions.extend(outputs.argmax(1).tolist())
        return asarray(predictions)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {
            "config": self.model.config,
            "trainer_config": self.config,
            "model_name": self.model.model_name,
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["config", "trainer_config", "model_state_dict"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = ModelForClassification(
            ckpt['model_name'],
            ckpt["config"]
        )
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.model.to(new_trainer.device)
        return new_trainer
