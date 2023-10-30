import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.datasets import get_dataset_iemocap, collate_fn, collate_multimodal_fn, get_dataset_mosei, get_dataset_sims
from src.models.mbt import E2EMBT
from src.models.e2e import MME2E
from src.trainers.emotiontrainer import IemocapTrainer, SimsTrainer
from src.loss import criterion_factory, BCEWithLogitsLossWrapper

def load_state_dict(model,path):
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict,strict=False)
    return model

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

    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,  num_workers=0, collate_fn=collate_fn)
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
    elif args['dataset'] in ['simsv1', 'simv2'] :
        train_dataset = get_dataset_sims(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'], version=args['dataset'])
        valid_dataset = get_dataset_sims(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'], version=args['dataset'])
        test_dataset = get_dataset_sims(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'], version=args['dataset'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_multimodal_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_multimodal_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_multimodal_fn)


    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    if args['model'] == 'mme2e':
        model = MME2E(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mbt':
        model = E2EMBT(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.mbt.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    if args['resume'] != "":
        model = load_state_dict(model, args['resume'])
        model = model.to(device=device)

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
        criterion = BCEWithLogitsLossWrapper(args['infer'], pos_weight)
    else:
        if args['dataset'] in ['simsv1', 'simv2']:
            pos_weight = None
        else:
            pos_weight = train_dataset.getPosWeight()
            pos_weight = torch.tensor(pos_weight).to(device)
        criterion = criterion_factory(args['fusion'], args['loss'], args['temperature'], pos_weight, device)


    if args['dataset'] == 'iemocap' or args['dataset'] == 'mosei':
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)
    else:
        trainer = SimsTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
