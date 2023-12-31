import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_iemocap, eval_sims_regression
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer, BertTokenizer

class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-base-v2')
        self.eval_func = eval_iemocap
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['loss'] in ['dwa', 'bce', 'mean', 'rruw', 'druw']:
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]

        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats, train_thresholds = self.train_one_epoch(epoch)
            valid_stats, valid_thresholds = self.eval_one_epoch()
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[-1] > self.best_valid_stats[-1]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1
            else:
                if (valid_stats[3][-1] + valid_stats[0][-1]) > (self.best_valid_stats[3][-1] + self.best_valid_stats[0][-1]):
                    for i in range(len(self.headers)):
                        for j in range(len(valid_stats[i])):
                            self.best_valid_stats[i][j] = valid_stats[i][j]

                    self.earlyStop = self.args['early_stop']
                    self.best_epoch = epoch
                    self.best_model = copy.deepcopy(self.model.state_dict())
                else:
                    self.earlyStop -= 1

                for i in range(len(self.headers)):
                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))

            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

    def train_one_epoch(self, epoch):
        self.model.train()
        if self.args['model'] == 'mme2e':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            if self.model.fusion == 'dict':
                Y_dict = {
                "audio": Y,
                "text": Y,
                "visual": Y
            }
            elif self.model.fusion == 'mlp':
                Y_dict = {
                "audio": Y,
                "text": Y,
                "visual": Y,
                "fusion": Y
            }

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                if self.model.fusion in ['dict', 'mlp']:
                    loss = self.criterion.forward(logits, Y_dict)
                else:
                    loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            if self.model.fusion in ['dict', 'mlp']:
                total_logits.append(logits[self.model.infer].cpu())
            else:
                total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        prefix = self.get_saving_file_name_prefix()
        prefix = f'{prefix}_{epoch}_'
        self.criterion.savefile(prefix)
        return self.eval_func(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            if self.model.fusion == 'dict':
                Y_dict = {
                "audio": Y,
                "text": Y,
                "visual": Y
            }
            elif self.model.fusion == 'mlp':
                Y_dict = {
                "audio": Y,
                "text": Y,
                "visual": Y,
                "fusion": Y
            }

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                if self.model.fusion in ['dict', 'mlp']:
                    loss = self.criterion.forward(logits, Y_dict)
                else:
                    loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            if self.model.fusion in ['dict', 'mlp']:
                total_logits.append(logits[self.model.infer].cpu())
            else:
                total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)

        return self.eval_func(total_logits, total_Y, thresholds)

class SimsTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(SimsTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = BertTokenizer.from_pretrained(f'bert-base-chinese')
        self.eval_func = eval_sims_regression
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        self.header = ['Phase', 'Acc2', 'MAE', 'Cor', 'F1']
        self.best_valid_stats = [0, 0, 0, 0]

        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats = self.train_one_epoch(epoch)
            valid_stats = self.eval_one_epoch()
            test_stats = self.eval_one_epoch('test')

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            train_stats_str = [f'{s:.4f}' for s in train_stats]
            valid_stats_str = [f'{s:.4f}' for s in valid_stats]
            test_stats_str = [f'{s:.4f}' for s in test_stats]
            print(tabulate([
                ['Train', *train_stats_str],
                ['Valid', *valid_stats_str],
                ['Test', *test_stats_str]
            ], headers=self.header))
            if valid_stats[-1] > self.best_valid_stats[-1]:
                self.best_valid_stats = valid_stats
                self.best_epoch = epoch
                self.earlyStop = self.args['early_stop']
            else:
                self.earlyStop -= 1

            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        print(tabulate([
            [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
        ], headers=self.header))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

    def train_one_epoch(self, epoch):
        self.model.train()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = {
                "audio": Y["audio"].to(device=self.device),
                "text": Y["text"].to(device=self.device),
                "visual": Y["visual"].to(device=self.device),
                "fusion": Y["fusion"].to(device=self.device),
            }

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion.forward(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y["fusion"].size(0)
                data_size += Y["fusion"].size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits.append(logits["fusion"].cpu())
            total_Y.append(Y["fusion"].cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        prefix = self.get_saving_file_name_prefix()
        prefix = f'{prefix}_{epoch}_'
        self.criterion.savefile(prefix)
        return self.eval_func(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid'):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = {
                "audio": Y["audio"].to(device=self.device),
                "text": Y["text"].to(device=self.device),
                "visual": Y["visual"].to(device=self.device),
                "fusion": Y["fusion"].to(device=self.device),
            }

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text)
                loss = self.criterion.forward(logits, Y)
                epoch_loss += loss.item() * Y["fusion"].size(0)
                data_size += Y["fusion"].size(0)

            total_logits.append(logits["fusion"].cpu())
            total_Y.append(Y["fusion"].cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)

        return self.eval_func(total_logits, total_Y)
