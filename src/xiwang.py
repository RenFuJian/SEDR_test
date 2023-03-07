from time import sleep
from subprocess import run, PIPE
import requests

import psutil
from multiprocessing import Process
import os


def login():
    list = psutil.pids()
    for i in list:
        p = psutil.Process(i)
        if p.name() == "DrClient.exe":
            print(p.name())
            p.terminate()
            sleep(5)


    os.popen("C:\Drcom\DrUpdateClient\DrMain.exe")
    sleep(20)
    print(123)
cnt = 1
while True:
    r = run('ping www.baidu.com',
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            shell=True)
    if r.returncode:
        print('relogin 第{}次'.format(cnt))
        login()
        cnt += 1

    sleep(60) # 每1分钟检查一次