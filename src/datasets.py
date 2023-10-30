import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Tuple, Optional
from src.utils import load
from PIL import Image
from torchvggish import vggish_input

audio_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762]

def getEmotionDict() -> Dict[str, int]:
    return {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}

def get_dataset_iemocap(data_folder: str, phase: str, img_interval: int):
    main_folder = os.path.join(data_folder, 'IEMOCAP')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    emoDict = getEmotionDict()
    uttr_ids = open(os.path.join(data_folder, 'IEMOCAP_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[uttr_id]['text'] for uttr_id in uttr_ids]
    labels = [emoDict[meta[uttr_id]['label']] for uttr_id in uttr_ids]

    return IEMOCAP(
        main_folder=main_folder,
        utterance_ids=uttr_ids,
        texts=texts,
        labels=labels,
        label_annotations=list(emoDict.keys()),
        img_interval=img_interval
    )

def get_dataset_mosei(data_folder: str, phase: str, img_interval: int):
    main_folder = os.path.join(data_folder, 'MOSEI')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    ids = open(os.path.join(data_folder, 'MOSEI_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[id]['text'] for id in ids]
    labels = [meta[id]['label'] for id in ids]

    return MOSEI(
        main_folder=main_folder,
        ids=ids,
        texts=texts,
        labels=labels,
        img_interval=img_interval
    )

def get_dataset_sims(data_folder: str, phase: str, img_interval: int, version: str):
    main_folder = os.path.join(data_folder, 'CHSIMS')
    label_file = version + '.csv'
    with open(os.path.join(main_folder, label_file), encoding="utf-8") as f:
        lines = f.read().splitlines()

    ids = []
    texts = []
    labels = []
    annotations = []
    modes = []
    for line in lines[1:]:
        video_id, clip_id, remain = line.split(',', 2)
        text, label, labelt, labela, labelv, annotation, mode = remain.rsplit(',', 6)
        if phase != mode:
            continue
        ids.append(f"{video_id}/{clip_id}")
        texts.append(text)
        annotations.append(annotation)
        modes.append(mode)
        labels.append({
            "audio": float(labela),
            "text": float(labelt),
            "visual": float(labelv),
            "fusion": float(label),
        })

    return CHSIMS(
        main_folder=main_folder,
        ids=ids,
        texts=texts,
        labels=labels,
        img_interval=img_interval
    )

class MOSEI(Dataset):
    def __init__(self, main_folder: str, ids: List[str], texts: List[List[int]], labels: List[int], img_interval: int):
        super(MOSEI, self).__init__()
        self.ids = ids
        self.texts = texts
        self.labels = np.array(labels)
        self.main_folder = main_folder
        self.img_interval = img_interval
        self.transform = transforms.Compose(
            [
                transforms.Resize((260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]
        )

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def sample_imgs_by_interval(self, folder: str, fps: Optional[int] = 30) -> List[str]:
        files = glob.glob(f'{folder}/*')
        nums = len(files) - 1
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        if len(sampled) == 0:
            step = int(self.img_interval / 1000 * fps) // 4
            sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))

        return specs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        this_id = self.ids[ind]
        sample_folder = os.path.join(self.main_folder, this_id)

        sampledImgs = [
            self.transform(Image.open(imgPath)) # .transpose()
            for imgPath in self.sample_imgs_by_interval(sample_folder)
        ]

        specgrams = vggish_input.wavfile_to_examples(os.path.join(sample_folder, f'audio.wav'))

        return this_id, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

class IEMOCAP(Dataset):
    def __init__(self, main_folder: str, utterance_ids: List[str], texts: List[List[int]], labels: List[int],
                 label_annotations: List[str], img_interval: int):
        super(IEMOCAP, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.labels = F.one_hot(torch.tensor(labels)).numpy()
        self.label_annotations = label_annotations

        self.utteranceFolders = {
            folder.split('\\')[-1]: folder
            for folder in glob.glob(os.path.join(main_folder, '**/*'))
        }
        self.img_interval = img_interval
        self.transform = transforms.Compose(
            [
                transforms.Resize((260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]
        )

    def get_annotations(self) -> List[str]:
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        entries = utteranceFolder.split('_')
        return entries[0][-1] == entries[-1][0]

    def sample_imgs_by_interval(self, folder: str, imgNamePrefix: str, fps: Optional[int] = 30) -> List[str]:
        '''
        Arguments:
            @folder - utterance folder
            @imgNamePrefix - prefix of the image name (determines L/R)
            @interval - how many ms per image frame
            @fps - fps of the original video
        '''
        files = glob.glob(f'{folder}/*')
        nums = (len(files) - 5) // 2
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f'{imgNamePrefix}{i}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def cutWavToPieces(self, waveform, sampleRate):
        # Split the audio waveform by second
        total = int(np.ceil(waveform.size(-1) / sampleRate))
        waveformPieces = []
        for i in range(total):
            waveformPieces.append(waveform[:, i * sampleRate:(i + 1) * sampleRate])

        # Pad the last piece
        lastPieceLength = waveformPieces[-1].size(-1)
        if lastPieceLength < sampleRate:
            padLeft = (sampleRate - lastPieceLength) // 2
            padRight = sampleRate - lastPieceLength - padLeft
            waveformPieces[-1] = F.pad(waveformPieces[-1], (padLeft, padRight))
        return waveformPieces

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        uttrId = self.utterance_ids[ind]
        uttrFolder = self.utteranceFolders[uttrId]
        use_left = self.use_left(uttrId)
        suffix = 'L' if use_left else 'R'
        audio_suffix = 'L' if use_left else 'R'
        imgNamePrefix = f'image_{suffix}_'

        sampledImgs = [
            self.transform(Image.open(imgPath)) # .transpose()
            for imgPath in self.sample_imgs_by_interval(uttrFolder, imgNamePrefix)
        ]

        specgrams = vggish_input.wavfile_to_examples(os.path.join(uttrFolder, f'audio_{audio_suffix}.wav'))

        return uttrId, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

def collate_multimodal_fn(batch):
    utterance_ids = []
    texts = []
    labels = {
        "audio":[],
        "visual":[],
        "text":[],
        "fusion":[],
    }

    newSampledImgs = None
    imgSeqLens = []

    specgrams = []
    specgramSeqLens = []

    for dp in batch:
        utteranceId, sampledImgs, specgram, text, label = dp
        if len(sampledImgs) == 0:
            continue
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels["audio"].append(label["audio"])
        labels["visual"].append(label["visual"])
        labels["text"].append(label["text"])
        labels["fusion"].append(label["fusion"])

        imgSeqLens.append(len(sampledImgs))
        if newSampledImgs is None:
            newSampledImgs = torch.stack(sampledImgs, dim=0)
        else:
            sampledImgs = torch.stack(sampledImgs, dim=0)
            newSampledImgs = torch.cat((newSampledImgs, sampledImgs), dim=0)

        specgramSeqLens.append(specgram.shape[0])
        specgrams.append(specgram)

    imgs = newSampledImgs

    labels["audio"] = torch.tensor(labels["audio"], dtype=torch.float32).unsqueeze(-1)
    labels["visual"] = torch.tensor(labels["visual"], dtype=torch.float32).unsqueeze(-1)
    labels["text"] = torch.tensor(labels["text"], dtype=torch.float32).unsqueeze(-1)
    labels["fusion"] = torch.tensor(labels["fusion"], dtype=torch.float32).unsqueeze(-1)

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(specgrams, dim=0),
        specgramSeqLens,
        texts,
        labels
    )

def collate_fn(batch):
    utterance_ids = []
    texts = []
    labels = []

    newSampledImgs = None
    imgSeqLens = []

    specgrams = []
    specgramSeqLens = []

    for dp in batch:
        utteranceId, sampledImgs, specgram, text, label = dp
        if len(sampledImgs) == 0:
            continue
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels.append(label)

        imgSeqLens.append(len(sampledImgs))
        if newSampledImgs is None:
            newSampledImgs = torch.stack(sampledImgs, dim=0)
        else:
            sampledImgs = torch.stack(sampledImgs, dim=0)
            newSampledImgs = torch.cat((newSampledImgs, sampledImgs), dim=0)

        specgramSeqLens.append(specgram.shape[0])
        specgrams.append(specgram)

    imgs = newSampledImgs

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(specgrams, dim=0),
        specgramSeqLens,
        texts,
        torch.tensor(labels, dtype=torch.float32)
    )

class CHSIMS(Dataset):
    def __init__(self, main_folder: str, ids: List[str], texts: List[List[int]], labels: List[int], img_interval: int):
        super(CHSIMS, self).__init__()
        self.ids = ids
        self.texts = texts
        self.labels = labels
        self.main_folder = main_folder
        self.img_interval = img_interval
        self.transform = transforms.Compose(
            [
                transforms.Resize((260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]
        )

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def sample_imgs_by_interval(self, folder: str, fps: Optional[int] = 30) -> List[str]:
        files = glob.glob(f'{folder}/*')
        nums = len(files) - 1
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f'{i:05d}.jpg') for i in list(range(0, nums, step))]
        if len(sampled) == 0:
            step = int(self.img_interval / 1000 * fps) // 4
            sampled = [os.path.join(folder, f'{i:05d}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def cutSpecToPieces(self, spec, stride=32):
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))

        return specs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        this_id = self.ids[ind]
        sample_folder = os.path.join(self.main_folder, this_id)

        sampledImgs = [
            self.transform(Image.open(imgPath)) # .transpose()
            for imgPath in self.sample_imgs_by_interval(sample_folder)
        ]

        specgrams = vggish_input.wavfile_to_examples(os.path.join(sample_folder, f'audio.wav'))

        return this_id, sampledImgs, specgrams, self.texts[ind], self.labels[ind]
