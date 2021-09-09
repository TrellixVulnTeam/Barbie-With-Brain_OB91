import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
while True:
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            print(audio)
            raw_text = r.recognize_google(audio)
            print(raw_text)
    except Exception as e:
        print(e)
