import json
import itertools
import sys
# Replace 'your_file.json' with the path to your actual JSON file
file_path = '/scratch/irving.b/vqa/annotations/v2_OpenEnded_mscoco_train2014_questions.json'

other_file_path = '/scratch/irving.b/vqa/annotations/v2_mscoco_train2014_annotations.json'
# Open the JSON file and load its content into a dictionary
with open(file_path, 'r') as file:
    data = json.load(file)

with open(other_file_path, 'r') as file:
    data_2 = json.load(file)
# how to work with this data

from typing import List, Dict, Any

def get_score(count: int) -> float:
    """
    Calculate a score based on the count.

    Parameters:
    - count (int): The count of something (e.g., answers).

    Returns:
    - float: The calculated score, capped at 1.0.
    """
    return min(1.0, count / 3)

def process_annotations(annotations: List[Dict[str, Any]], config: Any) -> None:
    """
    Process a list of annotations, updating each with labels and scores.

    Parameters:
    - annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
    - config (Any): Configuration object that contains mapping of labels to IDs.

    Note: This function updates the annotations list in place.
    """
    for annotation in annotations:
        answers = annotation['answers']
        answer_count = {}

        # Count occurrences of each unique answer
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []

        # Calculate scores for answers that are mapped to labels
        for answer in answer_count:
            if answer not in list(config.label2id.keys()):
                continue
            labels.append(config.label2id[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        # Update the annotation with calculated labels and scores
        annotation['labels'] = labels
        annotation['scores'] = scores

annotations = data_2['annotations'][0:2]
print(annotations)

# hic sunc leones
labels = [[item['answer_id'] for item in annotation['answers']] for annotation in annotations]
unique_labels = [list(set(unique)) for unique in labels]
print(unique_labels)
# we want to make a dictionary mapping of labels2id
# then id2label
answers = [[item['answer'] for item in annotation['answers']] for annotation in annotations]



unique_answers = [list(set(unique)) for unique in answers]
print(unique_answers)

for answer in unique_answers:
    print({label:idx for idx, label in enumerate(answer)})
    for idx, answer in enumerate(answer):
        print(idx)
        print(answer)
# now, how to make the two dictionaries:
# first, id2label (give the dictionary an id, then map it to a label)
# a dictionary for each entry, correct? (correct, unfortunately)
# doing in this in batches would be more effective, no?

# naive implementation
# some dynamic dispatch
# oh my goodness bro hasn't used his brain in years
id2_labels_list = []
index = 0
for unique in unique_labels:
    id2label = {}
    for label in unique:
        id2label[label] = annotations[index]['answers'][label-1]['answer']  
    id2_labels_list.append(id2label)
    index += 1

# how to create id2labels 
#print(id2_labels_list)

# now: labels to ids?

#labels_to_ids = []
#index = 0
#for unique in unique_answers:
#    id2label = {}
    # then, we are going to create a 
#    for answer in unique:

sys.exit()
process_annotations(annotations)
print(annotations)