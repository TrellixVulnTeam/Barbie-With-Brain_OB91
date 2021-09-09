from pytorch_transformers import OpenAIGPTTokenizer
from utils import download_pretrained_model

model_checkpoint = download_pretrained_model()
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
personality = []
p = input('Enter personality: ').split('.')
for i in p:
    personality.append(tokenizer.encode(i))
print(f'Tokenized personality is {personality}')
