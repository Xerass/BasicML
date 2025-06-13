import os
import sys
import pyaudio
import json
import queue
import keyboard
import pyautogui as gw
import time
from threading import Thread
from vosk import Model, KaldiRecognizer

path = path = "C:/Users/Jomar/VSCode/Python/Models/vosk-model-small-en-us-0.15"

# Find and activate the mGBA window
windows = [win for win in gw.getAllTitles() if "mGBA" in win]
if windows:
    # Activate the first matching window
    mGBA_window = gw.getWindowsWithTitle(windows[0])[0]
    mGBA_window.activate()
    time.sleep(0.5)  # Give time for the window to activate

#create a queue for all the inputs
input_queue = queue.Queue()

keymappings = {
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    "select" : "backspace",
    "start" : "enter",
    "be" : "k",
    "a" : "l",
    "lima" : "i",
    "romeo" : "o",
    "speed" : ["shift", "tab"]
}

def keyboard_exec():
   while True:
    #pop from queue
       action = input_queue.get()
       #if directional press
       if action in keymappings:
            keyboard.press(keymappings[action])
            time.sleep(0.1)  # Hold the key for 100ms
            keyboard.release(keymappings[action])

#a daemon thread is a background thread, since we just do keyboard commands it does not need to interrupt main functions           
keyboardThread = Thread(target = keyboard_exec, daemon = True)
keyboardThread.start()

SmallEnModel = Model(path)

#Listens at 16k Hz a second (16k instances taken in) for clearer audio
#typical human speaking is only around 300 to 3400 Hz to capture important frequencies, 16k is just a safe measure
#Kaldi itself is optitmized to work with this
recog = KaldiRecognizer(SmallEnModel, 16000)

#initializes a port audio / mic 
MicInput = pyaudio.PyAudio()

#create an input stream with the mic, specify 16k hertz for rate, only one channel fo mono (exptected data type of vosk libs)
#we format that input into a 16 bit signed int to numerically represent the sounds
#Set the input flag as true to set it to take in input
#frame buffer to 4k that is of the 16k hz taken a second it is processed 4k at a time for faster processing speed
InputStream = MicInput.open(rate = 16000, channels = 1, format = pyaudio.paInt16, input = True, frames_per_buffer=4000)

print("Vosk Model Initialized, Speak into the Microphone")

#continouos processing
while True:
    #read 4k hz of audio frames at a time
    data = InputStream.read(4000)

    #if there is audio input

    if recog.AcceptWaveform(data):
        #result returns a json so we need to load that
        result = json.loads(recog.Result())
        if 'text' in result and result['text']:
            text = result["text"].lower()
            print("Recognized:", text)
            
            words = text.split()
            for action in keymappings:
                if action in words:
                    input_queue.put(action)

            if 'exit' in result['text'].lower():
                print("Thank you for using the program!")
                break


#Close the streams
InputStream.stop_stream()
InputStream.close()
MicInput.terminate()