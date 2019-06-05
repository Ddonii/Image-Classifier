import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sb
from PIL import Image

#Training part
def training(transfer, data_dir, save_dir, learning_rate, hidden_units, epochs = 3, process = 'gpu'):
    #determine transfer model
    if transfer == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif transfer == 'vgg16':
        model = models.vgg16(pretraind = True)
    else:
        print('Input Again!')

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    #data importing
    data_transforms =  {'train':transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
                        'valid':transforms.Compose([transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
                       }
    image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
                     }
    dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle=True),
                   'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32, shuffle=True)
                  }
    for param in model.parameters():
        param.require_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    if process == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    #training and validation
    epochs = epochs
    highest_accuracy = 0
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            model.eval()
            valid_loss = 0
            valid_accuracy = 0
            with torch.no_grad():
                for images, labels in dataloaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    valid_loss += criterion(logps, labels)
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                if valid_accuracy/len(dataloaders['valid']) >= highest_accuracy:
                    highest_accuracy = valid_accuracy/len(dataloaders['valid'])
                    best_parameter = model.state_dict()

        print("Epoch = {}/{}..".format(epoch+1,epochs),
              "Train Loss = {:.3f}..".format(running_loss/len(dataloaders['train'])),
              "Validation Loss = {:.3f}..".format(valid_loss/len(dataloaders['valid'])),
              "Validation Accuracy = {:.3f}".format(valid_accuracy/len(dataloaders['valid']))
             )
        model.train()
    print("Highest Accuracy = {:.3f}".format(highest_accuracy))
    model.load_state_dict(best_parameter)

    #Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size' : 25088,
                  'hidden_size' : hidden_units,
                  'output_size' : 102,
                  'transfer_model' : transfer,
                  'm_state_dict' : model.state_dict(),
                  'o_state_dict' : optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'learning_rate' : learning_rate,
                  'data_dir' : data_dir
                 }
    torch.save(checkpoint, save_dir)
    print('-----Complete------')
    return model

#Predicting part
def predicting(inputs, process, category_names, checkpoint, top_k = 5):
    #determine process type
    if process == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

    #import category_names
    with open(category_names, 'r') as f:
        flower_to_name = json.load(f)

    #Loading the checkpoint
    checkpoint = torch.load(checkpoint)
    input_size, output_size = checkpoint['input_size'], checkpoint['output_size']
    hidden_size = checkpoint['hidden_size']
    if checkpoint['transfer_model'] == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif checkpoint['transfer_model'] == 'vgg16':
        model = models.vgg16(pretrained = True)

    for param in model.parameters():
        param.require_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(hidden_size, output_size),
                                     nn.LogSoftmax(dim=1))
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['m_state_dict'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['o_state_dict'])

    #import data
    test_dir = checkpoint['data_dir'] + '/test'
    data_transforms = {'test' : transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
                      }
    image_datasets = {'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    dataloaders = {'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle=True)}

    #After loading, test Again
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images,labels in dataloaders['test']:
            images,labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps,labels)
            test_loss += loss.item()

        #Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print("Test loss: {:.3f}..".format(test_loss/len(dataloaders['test'])),
          "Test Acuracy: {:.3f}".format(accuracy/len(dataloaders['test']))
         )
    model.train();

    #process_image
    def process_image(image):
        # TODO: Process a PIL image for use in a PyTorch model
        im = Image.open(image)
        im = im.resize((256,256))
        im = im.crop((16,16,16+224,16+224))

        np.image = np.array(im)
        im_mean = np.array([0.485, 0.456, 0.406])
        im_std = np.array([0.229, 0.224, 0.225])
        np.image = np.image / 255
        normal = (np.image - im_mean) / im_std

        result_im = normal.transpose((2,0,1))
        result_im = torch.from_numpy(result_im)
        return result_im

    #Define 'imshow'   *Only need for display
    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        return ax

    #Define predict!!!!
    def predict(image_path, model, topk=5):
        # TODO: Implement the code to predict the class from an image file
        model.to('cpu')
        image = process_image(image_path)
        image = image.type(torch.FloatTensor)
        image = image.unsqueeze_(0)

        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, index = ps.topk(topk)
        probs, index = probs.detach().numpy().reshape(top_k), index.detach().numpy().reshape(top_k)

        #Convert index to class
        classes=[]
        for i in index:
            classes.append(list(model.class_to_idx.keys())[i])

            #Convert class to real name
        for idx, value in model.class_to_idx.items():
            model.class_to_idx[idx] = flower_to_name[idx]

        classes_sample=[]
        for i in classes:
            classes_sample.append(model.class_to_idx[str(i)])

        classes = classes_sample
        return probs, classes

    #Show the result (result don't need to display)
    image = inputs
    class_flower = image.split('/')[6]
    flower_name = flower_to_name[class_flower]
    probs, classes = predict(image, model, top_k)
    print('Flower name : {}'.format(flower_name))
    for probs, classes in zip(probs, classes):
        print('{} : {:.3f}'.format(classes, probs))
#Don't need to returns
