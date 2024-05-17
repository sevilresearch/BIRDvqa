# Loads in all the required images, questions, and answers

import os
from os import path
import json

BASE_PATH = path.dirname(__file__)

def read_questions(question_path):
    with open(path.join(BASE_PATH, question_path), 'r') as file:
        qs = json.load(file)
    
    questions = [q[0] for q in qs]
    answers = [q[1] for q in qs]
    image_ids = [q[2] for q in qs]
    return questions, answers, image_ids

def read_images_path(dir):
    images = {}

    # dir_path = path.join(BASE_PATH, dir)
    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            image_id = int(filename[:-4])
            images[image_id] = path.join(dir, filename)
    return images


