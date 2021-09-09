from transformers import OpenAIGPTTokenizer
from utils import download_pretrained_model
import pickle


model_checkpoint = download_pretrained_model()
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
personality = []

p = input('Enter personality: ').split('.')

for i in p:
    personality.append(tokenizer.encode(i))

print(f'Tokenized personality is {personality}')

with open('personality.pickle','wb') as f:
    pickle.dump(personality,f)

print('saved to config')