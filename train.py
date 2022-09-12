import argparse
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torch
import json
from PIL import Image
import numpy as np

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


def get_predict_args():
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('im_file', type=str, help='path to image')
    parser.add_argument('checkpoint', type=str, help='checkpoint used to load the model')
    parser.add_argument('--top_k', type=int, default=5,
                        help='returns the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='flower category names')
    parser.add_argument('--gpu', type=bool, default=True,
                        help='bool on whether to use GPU (True) or CPU (False)')
    return parser.parse_args()


def process_image(im):
    # TODO: Process a PIL image for use in a PyTorch model
    s = 256, 256

    im = im.crop((
        s[0] // 2 - 112,
        s[1] // 2 - 112,
        s[0] // 2 + 112,
        s[1] // 2 + 112))

    im.thumbnail(s, Image.ANTIALIAS)
    np_image = np.array(im)

    np_image = np_image / 255.

    channel_1 = np_image[:, :, 0]
    channel_2 = np_image[:, :, 1]
    channel_3 = np_image[:, :, 2]

    # Normalize image per channel
    channel_1 = (channel_1 - 0.485) / (0.229)
    channel_2 = (channel_2 - 0.456) / (0.224)
    channel_3 = (channel_3 - 0.406) / (0.225)

    np_image[:, :, 0] = channel_1
    np_image[:, :, 1] = channel_2
    np_image[:, :, 2] = channel_3

    # Transpose image
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #     print(image.shape)
    try:
        image = image.numpy().transpose((1, 2, 0))
    except:
        image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cpu');
    im = Image.open(path)
    process_im = process_image(im)
    process_im = np.expand_dims(process_im, axis=0)
    process_im_tensor = torch.Tensor(process_im)
    #     print(image_path.shape)
    #     print(process_im_tensor.shape)
    p = torch.exp(model(process_im_tensor))
    #     p = torch.exp(model(image_path))
    topp, topclass = p.topk(topk, dim=1)
    return topp, topclass
    # TODO: Implement the code to predict the class from an image file


def view_class(image, label, model, p, c, cat_names):
    d = c.view(c.shape[1])
    e = np.array(d)
    #     str(i.numpy()+
    class_list = [cat_names[str(i + 1)] for i in e]
    p = p.data.numpy().squeeze()

    imTest = image[0]
    imTest = imTest.numpy().transpose((1, 2, 0))
    labelStr = cat_names[str(label[0].numpy() + 1)]

    guessString = ""

    for i in range(len(e)):
        guessString += str(i + 1) + '. ' + class_list[i] + ' with probability ' + '{:.2f}'.format(p[i] * 100) + '% \n'
    print(guessString)


def load_checkpoint(path, model):
    cp = torch.load(path)
    model.load_state_dict(cp['state_dict'])
    e = cp['epoch']
    model.class_to_idx = cp['class_to_index']
    return model, e


def get_cat_names(jsonfile):
    with open(jsonfile, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


# def main():


in_arg_predict = get_predict_args()

classifier2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 512)),
    ('relu', nn.ReLU()),
    ('drop1', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(512, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))
model2 = models.resnet50(pretrained=True)
model2.fc = classifier2

test_data = datasets.ImageFolder('flowers' + '/test', transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

test_model, e = load_checkpoint(in_arg_predict.checkpoint, model2)
dataiter = iter(testloader)
image, label = dataiter.next()

im_path = 'flowers/valid/1/image_06739.jpg'
im = Image.open(r'flowers/valid/1/image_06739.jpg')

probs, classes = predict(im_path, test_model)

view_class(image, label, test_model, probs, classes, get_cat_names(in_arg_predict.category_names));
