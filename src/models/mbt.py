import torch
from torch import nn
from src.utils import padTensor
from transformers import AlbertModel
from timm.models.layers import trunc_normal_
from torch.nn.modules.transformer import _get_clones

from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.vgg_block import VggBasicBlock

from torch.nn import TransformerEncoderLayer

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, size='base'):
        super(MME2E_T, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        self.text_feature_affine = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, text):
        last_hidden_state = self.albert(**text)['last_hidden_state']
        text_features = self.text_feature_affine(last_hidden_state)
        return text_features

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        return outputs

    def forward(self, inputs: torch.Tensor, lens):
        max_len = max(lens)

        mask = [([False] * (l + 1) + [True] * (max_len - l)) for l in lens]
        mask = torch.tensor(mask).to(device=inputs.device)

        inputs = list(inputs.split(lens, dim=0))
        inputs = [padTensor(inp, max_len) for inp in inputs]
        inputs = torch.stack(inputs, dim=0)

        inputs = self.prepend_cls(inputs)

        inputs = inputs.permute(1, 0, 2)
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        return inputs.permute(1, 0, 2)


class MBT(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_bottle_token):
        super(MBT, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_bottle_token = num_bottle_token
        self.cls_index = num_bottle_token
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.a_layers = _get_clones(encoder_layer, num_layers)
        self.v_layers = _get_clones(encoder_layer, num_layers)
        self.t_layers = _get_clones(encoder_layer, num_layers)

        self.bot = nn.Parameter(torch.zeros(1, num_bottle_token, dim))

        trunc_normal_(self.bot, std=.02)


    def get_mask(self, lens, device, is_t=False):
        if is_t:
            max_len = 99
        else:
            max_len = max(lens)
        mask = [([False] * (l + 1 + self.cls_index) + [True] * (max_len - l)) for l in lens]
        return torch.tensor(mask).to(device=device)

    def forward(self, v: torch.Tensor, v_lens, a: torch.Tensor, a_lens, t: torch.Tensor, t_lens):
        B = v.shape[0]

        mask_a = self.get_mask(a_lens, a.device)
        mask_v = self.get_mask(v_lens, v.device)
        mask_t = self.get_mask(t_lens, t.device, is_t=True)

        bot = self.bot.expand(B, -1, -1)
        v = torch.cat((bot, v), dim=1)
        a = torch.cat((bot, a), dim=1)
        t = torch.cat((bot, t), dim=1)

        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        t = t.permute(1, 0, 2)

        for i in range(self.num_layers):
            v = self.v_layers[i](src=v, src_key_padding_mask=mask_v)
            a = self.a_layers[i](src=a, src_key_padding_mask=mask_a)
            t = self.t_layers[i](src=t, src_key_padding_mask=mask_t)

            v[:self.num_bottle_token] = (v[:self.num_bottle_token] + a[:self.num_bottle_token] + t[:self.num_bottle_token]) / 3
            a[:self.num_bottle_token] = v[:self.num_bottle_token]
            t[:self.num_bottle_token] = v[:self.num_bottle_token]

        return t[self.cls_index], v[self.cls_index], a[self.cls_index]


class E2EMBT(nn.Module):
    def __init__(self, args, device):
        super(E2EMBT, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        self.fusion = args['fusion']
        bot_nlayers = args['bot_nlayers']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        self.T = MME2E_T(feature_dim=trans_dim)
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

        self.mbt = MBT(trans_dim, bot_nlayers, nheads, 2)
        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(trans_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        if self.fusion == 'gate':
            self.weighted_fusion = nn.Linear(trans_dim, self.num_classes, bias=False)
            self.activate = nn.Tanh()
            self.gated_activate = nn.Softmax(dim=1)
        else:
            self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' in self.mod:
            t = self.T(text)
            text_lens = text["attention_mask"][:, 1:].sum(1)

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
            v = self.v_transformer(faces, imgs_lens)


        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            a = self.a_transformer(specs, spec_lens)

        cls_t, cls_v, cls_a = self.mbt(v, imgs_lens, a, spec_lens, t, text_lens)

        if self.fusion == 'audio':
            return self.a_out(cls_a)
        elif self.fusion == 'visual':
            return self.v_out(cls_v)
        elif self.fusion == 'text':
            return self.t_out(cls_t)
        elif self.fusion == 'avg':
            return (self.a_out(cls_a) + self.v_out(cls_v) + self.t_out(cls_t)) / 3
        elif self.fusion == 'gate':
            ha = self.activate(self.a_out(cls_a))
            hv = self.activate(self.v_out(cls_v))
            ht = self.activate(self.t_out(cls_t))
            h = torch.cat((ha.unsqueeze(-1), hv.unsqueeze(-1), ht.unsqueeze(-1)), dim=-1)
            z = torch.cat((cls_a.unsqueeze(1), cls_v.unsqueeze(1), cls_t.unsqueeze(1)), dim=1)
            z = self.weighted_fusion(z)
            z = self.gated_activate(z)
            out = h * z.permute(0, 2, 1)
            return torch.sum(out, 2)
        else:
            all_logits.append(self.t_out(cls_t))
            all_logits.append(self.v_out(cls_v))
            all_logits.append(self.a_out(cls_a))
            return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    def crop_img_center(self, img: torch.tensor, target_size=48):
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
