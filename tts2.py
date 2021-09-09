from gtts import gTTS
import os
tts = gTTS(text='Good morning frank', lang='en-US')
tts.save("good.mp3")
os.system("mpg123 good.mp3")