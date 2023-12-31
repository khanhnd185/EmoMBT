import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Learning using Multitask learning for Emotion Recognition')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=True)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0.0)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-es', '--early-stop', help='Early stop', type=int, required=False, default=5)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', type=float, required=False, default=-1.0)
    parser.add_argument('-sc', '--scheduler', help='Use scheduler to optimizer', action='store_true')
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=0)
    parser.add_argument('--loss', help='loss function', type=str, required=False, default='bce')
    parser.add_argument('--optim', help='optimizer function: adam/sgd', type=str, required=False, default='adam')
    parser.add_argument('--text-lr-factor', help='Factor the learning rate of text model', type=int, required=False, default=10)

    # Model
    parser.add_argument('-mo', '--model', help='Which model', type=str, required=False, default='mme2e')
    parser.add_argument('--center', help='Center modal', type=str, required=False, default='text')
    parser.add_argument('--mbt', help='Type of MBT', type=str, required=False, default='mbt')
    parser.add_argument('--custom', help='Custome name', type=str, required=False, default='default')
    parser.add_argument('--resume', help='Load model name', type=str, required=False, default='')
    parser.add_argument('--fusion', help='How to fuse modalities', type=str, required=False, default='mlp')
    parser.add_argument('--infer', help='How to fuse modalities', type=str, required=False, default='visual')
    parser.add_argument('--feature-dim', help='Dimension of features outputed by each modality model', type=int, required=False, default=256)
    parser.add_argument('--trans-dim', help='Dimension of the transformer after CNN', type=int, required=False, default=512)
    parser.add_argument('--trans-nlayers', help='Number of layers of the transformer after CNN', type=int, required=False, default=2)
    parser.add_argument('--bot-nlayers', help='Number of bottleneck layers of the MBT', type=int, required=False, default=2)
    parser.add_argument('--trans-nheads', help='Number of heads of the transformer after CNN', type=int, required=False, default=8)
    parser.add_argument('--temperature', help='Temperature', type=float, default=10.0)

    # Data
    parser.add_argument('--num-emotions', help='Number of emotions in data', type=int, required=False, default=4)
    parser.add_argument('--img-interval', help='Interval to sample image frames', type=int, required=False, default=500)
    parser.add_argument('--text-max-len', help='Max length of text after tokenization', type=int, required=False, default=300)

    # Path
    parser.add_argument('--datapath', help='Path of data', type=str, required=False, default='./data')
    parser.add_argument('--dataset', help='Use which dataset', type=str, required=False, default='iemocap')

    # Evaluation
    parser.add_argument('-mod', '--modalities', help='what modalities to use', type=str, required=False, default='tav')
    parser.add_argument('--valid', help='Only run validation', action='store_true')
    parser.add_argument('--test', help='Only run test', action='store_true')

    # Checkpoint
    parser.add_argument('--ckpt', help='Path of checkpoint', type=str, required=False, default='')
    parser.add_argument('--ckpt-mod', help='Load which modality of the checkpoint', type=str, required=False, default='tav')

    # LSTM
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-nl', '--num-layers', help='num of layers of LSTM', type=int, required=False, default=1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size of LSTM', type=int, required=False, default=300)
    parser.add_argument('-bi', '--bidirectional', help='Use Bi-LSTM', action='store_true')
    parser.add_argument('--gru', help='Use GRU rather than LSTM', action='store_true')

    args = vars(parser.parse_args())
    return args
