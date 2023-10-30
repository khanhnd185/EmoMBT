import os
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

input_dir = './MOSEI_RAW_PROCESSED/'
output_dir = './MOSEI'
batch_size = 96

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=224, margin=2, post_process=False, device=device)


sessions = os.listdir(input_dir)
for i, session in enumerate(sessions):
    print("{}/{}".format(i+1, len(sessions)))
    if not os.path.isdir(os.path.join(input_dir, session)):
        continue
    os.mkdir(os.path.join(output_dir, session))

    if 'IEMOCAP' in output_dir:
        subsessions = os.listdir(os.path.join(input_dir, session))
        subsessions = [subsession for subsession in subsessions if os.path.isdir(os.path.join(input_dir, session, subsession))]

        for subsession in tqdm(subsessions):
            os.mkdir(os.path.join(output_dir, session, subsession))
            files = os.listdir(os.path.join(input_dir, session, subsession))

            image_input_files = []
            image_output_files = []
            for j, file in enumerate(files):
                input_file = os.path.join(input_dir, session, subsession, file)
                output_file = os.path.join(output_dir, session, subsession, file)
                if file.endswith(".jpg"):
                    image_input_files.append(Image.open(input_file))
                    image_output_files.append(output_file)
                elif not os.path.isdir(input_file):
                    shutil.copyfile(input_file, output_file)

                if len(image_input_files) == batch_size or ((j == (len(files)-1)) and len(image_input_files) > 0):
                    mtcnn(image_input_files, save_path=image_output_files)
                    image_input_files = []
                    image_output_files = []
    else:
        files = os.listdir(os.path.join(input_dir, session))

        image_input_files = []
        image_output_files = []
        for j, file in enumerate(files):
            input_file = os.path.join(input_dir, session, file)
            output_file = os.path.join(output_dir, session, file)
            if file.endswith(".jpg"):
                image_input_files.append(Image.open(input_file))
                image_output_files.append(output_file)
            elif not os.path.isdir(input_file):
                shutil.copyfile(input_file, output_file)

            if len(image_input_files) == batch_size or ((j == (len(files)-1)) and len(image_input_files) > 0):
                mtcnn(image_input_files, save_path=image_output_files)
                image_input_files = []
                image_output_files = []

print("Done")
