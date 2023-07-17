import os
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer

class IemocapSupConTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapSupConTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-base-v2')
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []


    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_loss = self.train_one_epoch()
            print('Train loss: ', train_loss)
            if epoch % 4 == 0:
                self.save_supcon_model(epoch)

        print('Results and model are saved!')

    def save_supcon_model(self, epoch):

        name = f'{self.args["model"]}_{self.args["modalities"]}_{self.args["custom"]}_supcon{epoch}_'

        name += f'imginvl{self.args["img_interval"]}_'
        name += f'seed{self.args["seed"]}'
        name += '.pt'

        torch.save(self.model, os.path.join(self.saving_path, 'models', name))

    def train_one_epoch(self):
        self.model.train()
        if self.args['model'] == 'mme2e' or self.args['model'] == 'mme2e_sparse':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            bsz = Y.shape[0]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                f1, f2 = torch.split(logits, [bsz, bsz], dim=0)
                logits = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()

        epoch_loss /= len(dataloader.dataset)
        return epoch_loss
