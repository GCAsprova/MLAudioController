from pynput.keyboard import Key, Listener, Controller

keyboard = Controller()

def on_press(key):
    try:
        if key == Key.f2:
            print("F1 pressed — simulating 'A'")
            keyboard.press('a')
            keyboard.release('a')
    except Exception as e:
        print("Error:", e)

with Listener(on_press=on_press) as listener:
    listener.join()
