from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

model_name = 'bert-base-cased'

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Tokenize me this please"
encoded_inputs = tokenizer(text, return_tensors='pt')
encoded_inputs

output = model(**encoded_inputs)
last_hidden_state = output.last_hidden_state
pooler_output = output.pooler_output

last_hidden_state.shape
pooler_output


def predict(text):
    encoded_inputs = tokenizer(text, return_tensors='pt')
    return model(**encoded_inputs)[0]

sentence1 = "There was a fly drinking from my soup."
sentence2 = "To become a commercial pilot, he had to fly for 1500 hours."

token1 = tokenizer.tokenize(sentence1)
token2 = tokenizer.tokenize(sentence2)
len(token1)
len(token2)

out1 = predict(sentence1)
out2 = predict(sentence2)
out1[0].shape
out2[0].shape

token1.index("fly")
token2.index("fly")

emb1 = out1[0:, token1.index('fly'), :].detach()
emb1.shape

emb2 = out2[0:, token2.index('fly'), :].detach()
emb2.shape

cosine(emb1[0], emb2[0])