from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# 'pipeline' contains both the tokenizer and the model, which is useful in a sense
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
#example = "My name is Wolfgang and I live in Berlin"
#ner_results = nlp(example)

dataset = load_dataset("conll2003")
print(dataset)
# now, we train and test
#print(model(dataset["train"][0]))

print(dataset["train"][0]['tokens'])
example_input=dataset["train"][0]['tokens']
print(example_input)
#example_input = [' '.join(example) for example in example_input]
#print(example_input)
#print(' '.join(example_input))
example = tokenizer(example_input, return_tensors="pt", padding=True, truncation=True)
print(example)
output = model.forward(example['input_ids'])

# so we need to target the second column
print(output['logits'].shape)
#print(output['logits'])
#print(output['logits'])
print(dataset["train"][0]['ner_tags'])

