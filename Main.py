import AudioRecorder as Ar
from pynput import keyboard

def main():
    print("Press record button to start input...")
    with keyboard.Listener(on_press=Ar.on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()