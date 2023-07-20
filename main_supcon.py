import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.mvdatasets import get_dataset_iemocap, get_dataset_mosei, mv_collate_fn
# from src.models.e2e import MME2E
# from src.models.sparse_e2e import MME2E_Sparse
from src.models.mbt import SupConMBT
from src.models.e2e import MME2E
from src.models.baselines.lf_rnn import LF_RNN
from src.models.baselines.lf_transformer import LF_Transformer
from src.trainers.emotionsupcontrainer import IemocapSupConTrainer
from src.loss import MultiheadSupervisedContrastiveLossNoNegatives

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
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                    num_workers=0, collate_fn=mv_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False,
                                    num_workers=0, collate_fn=mv_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                    num_workers=0, collate_fn=mv_collate_fn)
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, collate_fn=mv_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=mv_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=mv_collate_fn)

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
        model = SupConMBT(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.mbt.parameters()},
                {'params': model.head.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mme2e_sparse':
        model = MME2E_Sparse(args=args, device=device)
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
    elif args['model'] == 'lf_rnn':
        model = LF_RNN(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_transformer':
        model = LF_Transformer(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
    else:
        scheduler = None

    criterion = MultiheadSupervisedContrastiveLossNoNegatives()

    if args['dataset'] == 'iemocap' or 'mosei':
        trainer = IemocapSupConTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
