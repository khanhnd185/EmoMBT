import torch
from torch import nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from transformers import AlbertModel
from typing import Optional, List
from src.utils import padTensor

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)

        inputs = inputs.permute(1, 0, 2)
        # inputs = self.pos_encoder(inputs)
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        if get_cls:
            return inputs[0]

        return inputs[1:].permute(1, 0, 2)



class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        # self.text_feature_affine = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, feature_dim)
        # )

    def forward(self, text, get_cls=False):
        # logits, hidden_states = self.albert(**text, output_hidden_states=True)
        ret = self.albert(**text)
        last_hidden_state = ret['last_hidden_state']

        if get_cls:
            cls_feature = last_hidden_state[:, 0]
            # cls_feature = self.text_feature_affine(cls_feature)
            return cls_feature

        text_features = self.text_feature_affine(last_hidden_state).sum(1)
        return text_features

class VggBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VggBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MME2E(nn.Module):
    def __init__(self, args, device):
        super(MME2E, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        self.feature_dim = args['feature_dim']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        text_cls_dim = 768
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])

        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' in self.mod:
            text_cls = self.T(text, get_cls=True)
            all_logits.append(self.t_out(text_cls))

        if 'v' in self.mod:
            faces = self.mtcnn(imgs)
            for i, face in enumerate(faces):
                if face is None:
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center
            faces = [self.normalize(face) for face in faces]
            faces = torch.stack(faces, dim=0).to(device=self.device)

            faces = self.V(faces)

            faces = self.v_flatten(faces.flatten(start_dim=1))
            faces = self.v_transformer(faces, imgs_lens, get_cls=True)

            all_logits.append(self.v_out(faces))

        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs = self.a_transformer(specs, spec_lens, get_cls=True)
            all_logits.append(self.a_out(specs))

        if len(self.mod) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    def crop_img_center(self, img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces,
        to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
