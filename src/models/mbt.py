import torch
from torch import nn
from src.utils import padTensor
from transformers import AlbertModel, BertModel
from timm.models.layers import trunc_normal_
from torch.nn.modules.transformer import _get_clones

from torchvision import transforms
from facenet_pytorch import MTCNN

from torch.nn import TransformerEncoderLayer
from torchvggish import vggish

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, dataset='simsv1'):
        super(MME2E_T, self).__init__()
        if dataset in ['simsv1', 'simsv2']:
            self.albert = BertModel.from_pretrained('bert-base-chinese')
        else:
            self.albert = AlbertModel.from_pretrained(f'albert-base-v2')
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
        self.cls_index = 2 * self.num_bottle_token
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.a_layers = _get_clones(encoder_layer, num_layers)
        self.v_layers = _get_clones(encoder_layer, num_layers)
        self.t_layers = _get_clones(encoder_layer, num_layers)

        self.bot_av = nn.Parameter(torch.zeros(1, num_bottle_token, dim))
        self.bot_vt = nn.Parameter(torch.zeros(1, num_bottle_token, dim))
        self.bot_ta = nn.Parameter(torch.zeros(1, num_bottle_token, dim))

        trunc_normal_(self.bot_av, std=.02)
        trunc_normal_(self.bot_vt, std=.02)
        trunc_normal_(self.bot_ta, std=.02)


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

        bot_av = self.bot_av.expand(B, -1, -1)
        bot_vt = self.bot_vt.expand(B, -1, -1)
        bot_ta = self.bot_ta.expand(B, -1, -1)
        v = torch.cat((bot_av, bot_vt, v), dim=1)
        a = torch.cat((bot_av, bot_ta, a), dim=1)
        t = torch.cat((bot_vt, bot_ta, t), dim=1)

        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        t = t.permute(1, 0, 2)

        for i in range(self.num_layers):
            v = self.v_layers[i](src=v, src_key_padding_mask=mask_v)
            a = self.a_layers[i](src=a, src_key_padding_mask=mask_a)
            t = self.t_layers[i](src=t, src_key_padding_mask=mask_t)

            v[:self.num_bottle_token] = (v[:self.num_bottle_token] + a[:self.num_bottle_token]) / 2
            a[:self.num_bottle_token] = v[:self.num_bottle_token]

            a[self.num_bottle_token:self.cls_index] = (a[self.num_bottle_token:self.cls_index] + t[self.num_bottle_token:self.cls_index]) / 2
            t[self.num_bottle_token:self.cls_index] = a[self.num_bottle_token:self.cls_index]

            t[:self.num_bottle_token] = (t[:self.num_bottle_token] + v[self.num_bottle_token:self.cls_index]) / 2
            v[self.num_bottle_token:self.cls_index] = t[:self.num_bottle_token]

        return t[self.cls_index], v[self.cls_index], a[self.cls_index]


class OEMBT(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_bottle_token, center_modal='text'):
        super(OEMBT, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.center_modal = center_modal
        self.num_bottle_token = num_bottle_token
        self.cls_index = 2 * self.num_bottle_token
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.a_layers = _get_clones(encoder_layer, num_layers)
        self.v_layers = _get_clones(encoder_layer, num_layers)
        self.t_layers = _get_clones(encoder_layer, num_layers)

        self.bot_av = nn.Parameter(torch.zeros(1, num_bottle_token, dim))
        self.bot_vt = nn.Parameter(torch.zeros(1, num_bottle_token, dim))
        self.bot_ta = nn.Parameter(torch.zeros(1, num_bottle_token, dim))

        trunc_normal_(self.bot_av, std=.02)
        trunc_normal_(self.bot_vt, std=.02)
        trunc_normal_(self.bot_ta, std=.02)


    def get_mask(self, lens, device, is_t=False, is_center_modal=False):
        if is_t:
            max_len = 99
        else:
            max_len = max(lens)
        if is_center_modal:
            mask = [([False] * (l + 1 + self.cls_index) + [True] * (max_len - l)) for l in lens]
        else:
            mask = [([False] * (l + 1 + self.num_bottle_token) + [True] * (max_len - l)) for l in lens]
        return torch.tensor(mask).to(device=device)

    def forward(self, v: torch.Tensor, v_lens, a: torch.Tensor, a_lens, t: torch.Tensor, t_lens):
        B = v.shape[0]

        mask_a = self.get_mask(a_lens, a.device, is_center_modal=(self.center_modal=='audio'))
        mask_v = self.get_mask(v_lens, v.device, is_center_modal=(self.center_modal=='visual'))
        mask_t = self.get_mask(t_lens, t.device, is_t=True, is_center_modal=(self.center_modal=='text'))

        bot_av = self.bot_av.expand(B, -1, -1)
        bot_vt = self.bot_vt.expand(B, -1, -1)
        bot_ta = self.bot_ta.expand(B, -1, -1)
        if self.center_modal == 'audio':
            v = torch.cat((bot_av, v), dim=1)
            a = torch.cat((bot_av, bot_ta, a), dim=1)
            t = torch.cat((bot_ta, t), dim=1)
        elif self.center_modal == 'visual':
            v = torch.cat((bot_av, bot_vt, v), dim=1)
            a = torch.cat((bot_av, a), dim=1)
            t = torch.cat((bot_vt, t), dim=1)
        else:
            v = torch.cat((bot_vt, v), dim=1)
            a = torch.cat((bot_ta, a), dim=1)
            t = torch.cat((bot_vt, bot_ta, t), dim=1)

        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        t = t.permute(1, 0, 2)

        for i in range(self.num_layers):
            v = self.v_layers[i](src=v, src_key_padding_mask=mask_v)
            a = self.a_layers[i](src=a, src_key_padding_mask=mask_a)
            t = self.t_layers[i](src=t, src_key_padding_mask=mask_t)

            if self.center_modal == 'audio':
                a[:self.num_bottle_token] = (v[:self.num_bottle_token] + a[:self.num_bottle_token]) / 2
                v[:self.num_bottle_token] = a[:self.num_bottle_token]
                a[self.num_bottle_token:self.cls_index] = (t[:self.num_bottle_token] + a[self.num_bottle_token:self.cls_index]) / 2
                t[:self.num_bottle_token] = a[self.num_bottle_token:self.cls_index]
            elif self.center_modal == 'visual':
                v[:self.num_bottle_token] = (v[:self.num_bottle_token] + a[:self.num_bottle_token]) / 2
                a[:self.num_bottle_token] = v[:self.num_bottle_token]
                v[self.num_bottle_token:self.cls_index] = (t[:self.num_bottle_token] + v[self.num_bottle_token:self.cls_index]) / 2
                t[:self.num_bottle_token] = v[self.num_bottle_token:self.cls_index]
            else:
                t[:self.num_bottle_token] = (t[:self.num_bottle_token] + v[:self.num_bottle_token]) / 2
                v[:self.num_bottle_token] = t[:self.num_bottle_token]
                t[self.num_bottle_token:self.cls_index] = (t[:self.num_bottle_token] + a[self.num_bottle_token:self.cls_index]) / 2
                a[:self.num_bottle_token] = t[self.num_bottle_token:self.cls_index]

        if self.center_modal == 'audio':
            return t[self.num_bottle_token], v[self.num_bottle_token], a[self.cls_index]
        elif self.center_modal == 'visual':
            return t[self.num_bottle_token], v[self.cls_index], a[self.num_bottle_token]
        else:
            return t[self.cls_index], v[self.num_bottle_token], a[self.num_bottle_token]


class MBT2(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_bottle_token, y_is_text=False):
        super(MBT2, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_bottle_token = num_bottle_token
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.x_layers = _get_clones(encoder_layer, num_layers)
        self.y_layers = _get_clones(encoder_layer, num_layers)

        self.bot = nn.Parameter(torch.zeros(1, num_bottle_token, dim))
        self.y_is_text = y_is_text

        trunc_normal_(self.bot, std=.02)


    def get_mask(self, lens, device, is_t=False):
        if is_t:
            max_len = 99
        else:
            max_len = max(lens)
        mask = [([False] * (l + 1 + self.num_bottle_token) + [True] * (max_len - l)) for l in lens]
        return torch.tensor(mask).to(device=device)

    def forward(self, x: torch.Tensor, x_lens, y: torch.Tensor, y_lens):
        B = x.shape[0]

        mask_x = self.get_mask(x_lens, x.device)
        mask_y = self.get_mask(y_lens, y.device, is_t=self.y_is_text)

        bot = self.bot.expand(B, -1, -1)
        x = torch.cat((bot, x), dim=1)
        y = torch.cat((bot, y), dim=1)

        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        for i in range(self.num_layers):
            x = self.x_layers[i](src=x, src_key_padding_mask=mask_x)
            y = self.y_layers[i](src=y, src_key_padding_mask=mask_y)

            x[:self.num_bottle_token] = (x[:self.num_bottle_token] + y[:self.num_bottle_token]) / 2
            y[:self.num_bottle_token] = x[:self.num_bottle_token]

        return x[self.num_bottle_token], y[self.num_bottle_token]



class E2EMBT(nn.Module):
    def __init__(self, args, device):
        super(E2EMBT, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        self.infer = args['infer']
        self.fusion = args['fusion']
        bot_nlayers = args['bot_nlayers']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        self.T = MME2E_T(feature_dim=trans_dim, dataset=args['dataset'])
        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = torch.load('enet_b2_8.pt',map_location=device)
        self.V.classifier = torch.nn.Identity()

        self.A = vggish(postprocess=False)

        self.v_flatten = nn.Sequential(
            nn.Linear(1408, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        if len(self.mod) == 3:
            if args['mbt'] == "oembt":
                self.mbt = OEMBT(trans_dim, bot_nlayers, nheads, 1, center_modal=args['center'])
            else:
                self.mbt = MBT(trans_dim, bot_nlayers, nheads, 1)
        else:
            self.mbt = MBT2(trans_dim, bot_nlayers, nheads, 1, y_is_text=('t' in self.mod))

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(trans_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        if self.fusion == 'gate':
            self.weighted_fusion = nn.Linear(trans_dim, self.num_classes, bias=False)
            self.activate = nn.Tanh()
            self.gated_activate = nn.Softmax(dim=1)
        else:
            self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

        if self.mod == 'a':
            self.fusion = 'audio'
        elif self.mod == 't':
            self.fusion = 'text'
        elif self.mod == 'v':
            self.fusion = 'visual'
        

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' in self.mod:
            t = self.T(text)
            text_lens = text["attention_mask"][:, 1:].sum(1)

        if 'v' in self.mod:
            imgs = imgs.to(self.device)
            faces = self.V(imgs)
            faces = self.v_flatten(faces)
            v = self.v_transformer(faces, imgs_lens)


        if 'a' in self.mod:
            specs = self.A(specs)
            specs = self.a_flatten(specs)
            a = self.a_transformer(specs, spec_lens)

        if len(self.mod) == 3:
            cls_t, cls_v, cls_a = self.mbt(v, imgs_lens, a, spec_lens, t, text_lens)
        elif self.mod == 'a':
            cls_a = a[:,0,:]
        elif self.mod == 't':
            cls_t = t[:,0,:]
        elif self.mod == 'v':
            cls_v = v[:,0,:]
        else:
            if 'a' in self.mod and 't' in self.mod:
                cls_a, cls_t = self.mbt(a, spec_lens, t, text_lens)
            elif 'a' in self.mod and 'v' in self.mod:
                cls_v, cls_a = self.mbt(v, imgs_lens, a, spec_lens)
            elif 'v' in self.mod and 't' in self.mod:
                cls_v, cls_t = self.mbt(v, imgs_lens, t, text_lens)

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
        elif self.fusion == 'dict':
            return {
                "audio": self.a_out(cls_a),
                "text": self.v_out(cls_v),
                "visual": self.t_out(cls_t),
            }
        else:
            cls_t = self.t_out(cls_t)
            cls_v = self.v_out(cls_v)
            cls_a = self.a_out(cls_a)
            fusion = self.weighted_fusion(torch.stack([cls_t, cls_v, cls_a], dim=-1)).squeeze(-1)
            if self.fusion == 'fusion':
                return fusion
            return {
                "audio": cls_a,
                "text": cls_v,
                "visual": cls_t,
                "fusion": fusion,
            }

    def crop_img_center(self, img: torch.tensor, target_size=48):
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
