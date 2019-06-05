import argparse
from project_function_form import predicting

parser = argparse.ArgumentParser()

parser.add_argument('input', type = str, nargs = '*', default = '/home/workspace/ImageClassifier/flowers/test/22/image_05362.jpg')
parser.add_argument('checkpoint', type = str, nargs = '*' ,default = '/home/workspace/ImageClassifier/checkpoint.pth')
parser.add_argument('--process', type = str, default = 'gpu')
parser.add_argument('--top_k', type = int, default = 2)
parser.add_argument('--category_names', type = str, default = '/home/workspace/ImageClassifier/cat_to_name.json')

predict_param = parser.parse_args()

predicting(predict_param.input, predict_param.process, predict_param.category_names, predict_param.checkpoint, predict_param.top_k)