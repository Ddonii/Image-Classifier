import argparse
from project_function_form import training

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, nargs = '*', default = '/home/workspace/ImageClassifier/flowers',
                    help = 'path to the folder of flower images')
parser.add_argument('--save_dir', type = str, 
                    default = '/home/workspace/ImageClassifier/checkpoint.pth')
parser.add_argument('--arch', type = str, default = 'vgg19')
parser.add_argument('--learning_rate', type = int, default = 0.001)
parser.add_argument('--hidden_units', type = int, default = 2048)
parser.add_argument('--epochs', type = int, default = 8)
parser.add_argument('--process', type = str, default = 'gpu')

train_param = parser.parse_args()
print(train_param.process, train_param.hidden_units)

training(train_param.arch, train_param.data_dir, train_param.save_dir, train_param.learning_rate, 
         train_param.hidden_units, train_param.epochs, train_param.process)
