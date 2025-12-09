from threading import Thread
from time import sleep


def func():
    for i in range(5):
        print(f"from child thread: {i}")
        sleep(0.5)


th = Thread(target=func)
th.start()
print("App stop")
