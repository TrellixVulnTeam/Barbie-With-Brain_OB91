import torch
import random
import pickle
import os
import wikipedia
import wolframalpha
import speech_recognition as sr
from gtts import gTTS
from utils import download_pretrained_model, add_special_tokens, sample_sequence
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


random.seed(42)
torch.random.manual_seed(42)
print('Finished Imports\n\n')

wolframalpha_id = 'EUGH2G-G3TQ5ALQ43'
random.seed(42)
torch.random.manual_seed(42)

print('Variables set\n\n')

with open('args.pickle', 'rb') as f:
    args = pickle.load(f)
with open('personality.pickle', 'rb') as f:
    personality = pickle.load(f)

print('Loaded settings from disk\n\n')

print('Loading Chat Module\n\n')
model_checkpoint = download_pretrained_model()
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
add_special_tokens(model, tokenizer)
model.to('cpu')
history = []

print('Initializing Voice Module')
client = wolframalpha.Client(wolframalpha_id)
r = sr.Recognizer()
mic = sr.Microphone()


def query_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=4)
    except Exception as e:
        return f'somthing went wrong did you mean to say {query}'


def query_wolframalpha(query):
    try:
        res = client.query(query)
        return next(res.results).text
    except Exception as e:
        return f'somthing went wrong did you mean to say {query}'


def query_chatbot(raw_text, history):

    with torch.no_grad():
        out_ids = sample_sequence(
            personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2 * args['max_history'] + 1):]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


print('All Done')
while True:
    try:
        print('>>>')
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            # print(audio)
            raw_text = r.recognize_google(audio)
            if raw_text.startswith('search'):
                print('searching')
                out_text = query_wikipedia(raw_text[7:])
            elif raw_text.startswith('tell me what is'):
                print('wait...')
                out_text = query_wolframalpha(raw_text[16:])
            elif raw_text.lower() == 'bye' or raw_text.lower() == 'goodbye':
                exit()
            else:
                out_text = query_chatbot(raw_text, history)
            print('ME : ', raw_text)
            print('BOT : ', out_text)
        history.append(tokenizer.encode(raw_text))
        tts = gTTS(text=out_text, lang='en-US')
        tts.save("response.mp3")
        os.system("ffplay -nodisp -autoexit response.mp3  >/dev/null 2>&1")

    except Exception as e:
        print(e)
