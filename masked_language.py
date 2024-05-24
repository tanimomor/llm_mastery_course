from transformers import AutoTokenizer, AutoModelForMaskedLM    #LM:languageModeling
from scipy.special import softmax
import numpy as np

model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

mask = tokenizer.mask_token

sentence = f"I want to {mask} pizza for tonight."
sentence

tokens = tokenizer.tokenize(sentence)
tokens

encoded_inputs = tokenizer(sentence, return_tensors='pt')
encoded_inputs

output = model(**encoded_inputs)
output

logits = output.logits
logits.shape
logits = logits.detach().numpy()[0]
logits.shape

mask_logits = logits[tokens.index(mask) + 1]
mask_logits

confidence_scores = softmax(mask_logits)
confidence_scores.shape

for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    # print(pred_token, score)
    print(sentence.replace(mask, pred_token))


for i in np.argsort(confidence_scores)[::-1][100:1000]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    # print(pred_token, score)
    print(sentence.replace(mask, pred_token))

    