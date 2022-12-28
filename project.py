# 라이브러리 불러오기
# pip install torch
# pip install torchvision
import cv2
import torch
from torchvision import transforms, models
from ipywidgets import interact
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy

# 이미지 파일경로 불러오기
def list_image_file(data_dir, sub_dir): # './Covid19-dataset/train/', 'Normal'
    image_format = ['jpeg', 'jpg', 'png']
    image_files = []

    images_dir = os.path.join(data_dir, sub_dir) # './Covid19-dataset/train/Normal'
#    print(images_dir)
    for file_path in os.listdir(images_dir):
        if file_path.split(".")[-1] in image_format:
            image_files.append(os.path.join(sub_dir, file_path))
    return image_files

data_dir = './Covid19-dataset/train/'
normals_list = list_image_file(data_dir, 'Normal')
covids_list = list_image_file(data_dir, 'Covid')
pneumonias_list = list_image_file(data_dir, 'Viral Pneumonia')

print(len(normals_list))
print(len(covids_list))
print(len(pneumonias_list))

# 이미지파일을 RGB 3차원 배열로 불러오기
def get_RGB_image(data_dir, file_name):
    image_file = os.path.join(data_dir, file_name)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 이미지 데이터 시각화하기
min_num_files = min(len(normals_list), len(covids_list), len(pneumonias_list))

@interact(index=(0, min_num_files-1))
def show_samples(index=0):
    normal_image = get_RGB_image(data_dir, normals_list[index])
    covid_image = get_RGB_image(data_dir, covids_list[index])
    pneumonia_image = get_RGB_image(data_dir, pneumonias_list[index])

    plt.figure(figsize=(12,8))
    plt.subplot(131)
    plt.title('Normal')
    plt.imshow(np.real(normal_image))
    plt.subplot(132)
    plt.title('covid')
    plt.imshow(np.real(covid_image))
    plt.subplot(133)
    plt.title('Pneumonia')
    plt.imshow(np.real(pneumonia_image))
    plt.tight_layout()
    plt.show()

# 학습 데이터셋 클래스 만들기
train_data_dir = './Covid19-dataset/train/'
class_list = ['Normal', 'Covid', 'Viral Pneumonia']
print(normals_list)
print(covids_list)
print(pneumonias_list)

files_path = normals_list + covids_list + pneumonias_list

target = class_list.index(files_path[0].split(os.sep)[-2])
target
target = class_list.index(files_path[0].split(os.sep)[0])
target
# 데이터셋: 데이터(샘플, 정답)을 저장한것
# 데이터로더: 데이터셋을 접근하기 쉽게 객체(iterable)로 감싼 것
class Chest_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        normals = list_image_file(data_dir, 'Normal')
        covids = list_image_file(data_dir, 'Covid')
        pneumonias = list_image_file(data_dir, 'Viral Pneumonia')
        self.files_path = normals + covids +pneumonias
        self.transfrom = transform

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        image_file = os.path.join(self.data_dir, self.files_path[index])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # os.sep: 디렉토리 분리 문자를 리턴(/)
        target = class_list.index(self.files_path[index].split(os.sep)[-2])
        # target = class_list.index(self.files_path[index].split(os.sep)[0])
        if self.transfrom:
            image = self.transfrom(image)
            target = torch.Tensor([target]).long()

        return {'image': image, 'target':target}

dest = Chest_dataset(train_data_dir)

index = 200
plt.title(class_list[dest[index]['target']])
plt.imshow(dest[index]['image'])
plt.show()

# 배열을 연산가능한 텐서로 변환하기
# transforms.ToTensor() 를 사용해서 픽셀 값의 범위를 0 ~ 1로 조절
# 각채널을 평균 0.5, 표준편차 0.5로 정규화를 적용
# -1 ~ 1 범위로 변환
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
train_dset = Chest_dataset(train_data_dir, transformer)

index = 200
image = train_dset[index]['image']
label = train_dset[index]['target']

print(image.shape, label)

# 데이터로더 구현하기
def build_dataloader(train_data_dir, val_data_dir):
    dataloaders = {}
    train_dset = Chest_dataset(train_data_dir, transformer)
    dataloaders['train'] = DataLoader(train_dset, batch_size=4, shuffle=True, drop_last=True)

    val_dest = Chest_dataset(val_data_dir, transformer)
    dataloaders['val'] = DataLoader(train_dset, batch_size=1, shuffle=False, drop_last=False)
    return dataloaders

train_data_dir = './Covid19-dataset/train/'
val_data_dir = './Covid19-dataset/test/'
dataloaders = build_dataloader(train_data_dir, val_data_dir)

for i, d in enumerate(dataloaders['train']):
    print(i, d)
    if i == 0:
        break

d['target'].shape
d['target'].squeeze()

# classification 모델(VGG19) 불러오기
# pretrained=True: 미리 학습된 weight들을 가지고 옴
model = models.vgg19(pretrained=True)

from torchsummary import summary
from torch import nn

summary(model, (3, 224, 224), batch_size=1, device='cpu')

model

# 데이터에 맞게 모델 head부분을 수정하기
def build_vgg19_based_model(device_name = 'cpu'):
    device = torch.device(device_name)
    model = models.vgg19(pretrained = True)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(class_list)),
        nn.Softmax(dim=1)
    )
    return model.to(device)

model = build_vgg19_based_model(device_name='cpu')
summary(model, (3, 224, 224), batch_size=1, device='cpu')

# 손실함수 불러오기
loss_func = nn.CrossEntropyLoss(reduction='mean')
# Gradient 최적화 함수 불러오기
optimizer = torch.optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)
# 모델 검증을 위한 Accuracy 생성하기
torch.no_grad()
def get_accuracy(image, target, model):
    batch_size = image.shape[0]
    prediction = model(image)
    _, pred_label = torch.max(prediction, dim=1)
    is_correct = (pred_label == target)
    return is_correct.cpu().numpy().sum()/batch_size

# 모델 학습을 위한 함수 구현하기
device = torch.device('cpu')

def train_one_epoch(dataloaders, model, optimizer, loss_func, device):
    losses = {}
    accuracies = {}

    for tv in ['train', 'val']:
        running_loss = 0.0
        running_correct = 0
        if tv == 'train':
            model.train()
        else:
            model.train()
        for index, batch in enumerate(dataloaders[tv]):
            image = batch['image'].to(device)
            target = batch['target'].squeeze(dim=1).to(device)

            with torch.set_grad_enabled(tv == 'train'):
                prediction = model(image)
                loss = loss_func(prediction, target)

                if tv == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_correct += get_accuracy(image, target, model)

            if tv == 'train':
                if index % 10 == 0:
                    print(f"{index}/{len(dataloaders['train'])} - Running loss: {loss.item()}")
        losses[tv] = running_loss / len(dataloaders[tv])
        accuracies[tv] = running_correct / len(dataloaders[tv])
    return losses, accuracies

def save_best_model(model_state, model_name, save_dir='./trained_model'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_state, os.path.join(save_dir, model_name))
# 모델 학습 수행하기
device = torch.device('cpu')

train_data_dir = './Covid19-dataset/train/'
val_data_dir = './Covid19-dataset/test/'

dataloaders = build_dataloader(train_data_dir, val_data_dir)
model = build_vgg19_based_model(device_name='cpu')
loss_func = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)

num_epochs = 10

best_acc = 0.0
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []


for epoch in range(num_epochs):
    losses, accuracies = train_one_epoch(dataloaders, model, optimizer, loss_func, device)
    train_loss.append(losses['train'])
    val_loss.append(losses['val'])
    train_accuracy.append(accuracies['train'])
    val_accuracy.append(accuracies['val'])

    print(f"{epoch + 1}/{num_epochs} - Train Loss:{losses['train']}, Val_Loss:{losses['val']}")
    print(f"{epoch + 1}/{num_epochs} - Train Acc:{accuracies['train']}, Val Acc:{accuracies['val']}")

    if (epoch > 3) and (accuracies['val'] > best_acc):
        best_acc = accuracies['val']
        best_model = copy.deepcopy(model.state_dict())
        save_best_model(best_model, f'model_{epoch + 1:02d}.pth')

print(f'Best Accracy: {best_acc}')

plt.figure(figsize=(6, 5))
plt.subplot(211)
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid('on')
plt.legend()
plt.subplot(212)
plt.plot(train_accuracy, label='train')
plt.plot(val_accuracy, label='val')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid('on')
plt.legend()
plt.tight_layout()

# 테스트 이미지를 통한 학습모델 분류 성능 검증하기
data_test = './Covid19-dataset/test/'
class_list = ['Normal', 'Covid', 'Viral Pneumonia']

test_normals_list = list_image_file(data_test, 'Normal')
test_covids_list = list_image_file(data_test, 'Covid')
test_pneumonias_list = list_image_file(data_test, 'Viral Pneumonia')


def preprocess_image(image):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    tensor_image = transformer(image)  # C, H, W
    tensor_image = tensor_image.unsqueeze(0)  # B, C, H, W
    return tensor_image


def model_predict(image, model):
    tensor_image = preprocess_image(image)
    prediction = model(tensor_image)

    _, pred_label = torch.max(prediction.detach(), dim=1)
    print('pred_label1: ', pred_label)
    pred_label = pred_label.squeeze(0)
    print('pred_label2: ', pred_label)
    return pred_label.item()

ckpt = torch.load('./trained_model/model_10.pth')

model = build_vgg19_based_model()
model.load_state_dict(ckpt)
model.eval()

min_num_files = min(len(test_covids_list), len(test_normals_list), len(test_pneumonias_list))


interact(index=(0, min_num_files - 1))
def show_result(index=25):
    normal_image = get_RGB_image(data_test, test_normals_list[index])
    covid_image = get_RGB_image(data_test, test_covids_list[index])
    pneumonia_image = get_RGB_image(data_test, test_pneumonias_list[index])

    prediction_1 = model_predict(normal_image, model)
    prediction_2 = model_predict(covid_image, model)
    prediction_3 = model_predict(pneumonia_image, model)

    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title(f'Pred:{class_list[prediction_1]} | GT:Normal')
    plt.imshow(normal_image)
    plt.subplot(132)
    plt.title(f'Pred:{class_list[prediction_2]} | GT:Covid')
    plt.imshow(covid_image)
    plt.subplot(133)
    plt.title(f'Pred:{class_list[prediction_3]} | GT:Pneumonia')
    plt.imshow(pneumonia_image)
    plt.tight_layout()


# 1. 이미지 데이터셋 구축
# 2. Tochvision transforms 라이브러리를 활용한 텐서형 데이터 변환
# 3. VGG 19 모델을 불러와 Head 부분을 수정
# 4. Cross entropy Loss Function, SGDM 적용
# 5. 인간 추론원리와 닮은 딥러닝 결과 출력









