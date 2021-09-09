# from gtts import gTTS
# import os
# tts = gTTS(text='Good morning abdul', lang='en-IN')
# tts.save("good.mp3")
# os.system("mpg321 good.mp3")

import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print(len(voices))
engine.setProperty('voice', voices[1].id)
engine.say("I will speak this text")
engine.runAndWait()
