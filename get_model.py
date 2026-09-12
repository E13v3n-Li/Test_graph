from models.vgg import VGG16
from models.vit import ViT
from models.cnn4fmnist import CNN4Fmnist
from models.alexnet import AlexNet


def get_model(data):
    if data == 'cifar10':
        model = AlexNet()
    elif data == 'cifar100':
        model = VGG16(num_classes=100)
    elif data == 'fmnist':
        model = CNN4Fmnist()
    elif data == 'tiny':
        model = ViT()
    return model





