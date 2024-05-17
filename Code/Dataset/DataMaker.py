import pandas as pd
from random import randint
import json


def create_questions(Class, Image_ID):
    #Creating types of questions for model
    questions = [
        (f'what type of car is?', Class.lower()),
    ]

    return (list(map(lambda x: x + (Image_ID,), questions)))


carNames = pd.read_csv('../AirsimVQA/train/_annotations.csv')


# Create a dictionary to map filenames to IDs
filename_to_id = {}
current_id = 1

carNames = carNames.sort_values(by=['filename'])

for filename in carNames['filename'].unique():
    filename_to_id[filename] = current_id
    current_id += 1


# Create a new column with unique IDs based on filenames
carNames['ID'] = carNames['filename'].map(filename_to_id)


# Save the modified DataFrame back to CSV
carNames.to_csv('modified_file.csv', index=False)


carNames_test = pd.read_csv('../AirsimVQA/test/_annotations.csv')


# Create a dictionary to map filenames to IDs
filename_to_id = {}
current_id = 1

carNames_test = carNames_test.sort_values(by=['filename'])

for filename in carNames_test['filename'].unique():
    filename_to_id[filename] = current_id
    current_id += 1


# Create a new column with unique IDs based on filenames
carNames_test['ID'] = carNames_test['filename'].map(filename_to_id)

all_questions = []

for index, row in carNames.iterrows():
    all_questions += create_questions(row['class'],row['filename'])

all_questions_test = []

for index, row in carNames_test.iterrows():
    all_questions_test += create_questions(row['class'],row['filename'])


all_answers = list(set(map(lambda q: q[1], all_questions)))

with open('questions_and_answers.txt', 'w') as file:
    json.dump(all_questions, file)

with open('questions_and_answers_test.txt', 'w') as file:
    json.dump(all_questions_test, file)

with open('answers.txt', 'w') as file:
  for answer in all_answers:
    file.write(f'{answer}\n')




