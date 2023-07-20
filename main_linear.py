import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.datasets import get_dataset_iemocap, collate_fn, HCFDataLoader, get_dataset_mosei, collate_fn_hcf_mosei
from src.models.mbt import SupConMBT
from src.trainers.emotionsupcontrainer import IemocapLinearTrainer

if __name__ == "__main__":
    start = time.time()

    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(int(args['cuda']))

    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                           img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        if args['hand_crafted']:
            train_loader = HCFDataLoader(dataset=train_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=True, num_workers=0)
            valid_loader = HCFDataLoader(dataset=valid_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=False, num_workers=0)
            test_loader = HCFDataLoader(dataset=test_dataset, feature_type=args['audio_feature_type'],
                                        batch_size=args['batch_size'], shuffle=False, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=0, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False,
                                      num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    if args['model'] == 'mlp':
        classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(32, 6)
            )
    else:
        classifier = nn.Linear(64, 6)

    #model = SupConMBT(args=args, device=device)
    #model.load_state_dict(torch.load(args['ckpt']))
    model = torch.load(args['ckpt'])
    model = model.to(device=device)
    classifier = classifier.to(device=device)
    model.eval()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
    else:
        scheduler = None

    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.BCEWithLogitsLoss()

    if args['dataset'] == 'iemocap' or 'mosei':
        trainer = IemocapLinearTrainer(args, model, classifier, criterion, optimizer, scheduler, device, dataloaders)

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
