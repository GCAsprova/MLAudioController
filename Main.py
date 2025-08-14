from pynput import keyboard
import AudioRecorder as Ar
import os

def main():
    print("Press SPACE to record and process...")
    with keyboard.Listener(on_press=Ar.on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()