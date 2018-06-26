#just for auto-typing data into a spreadsheet

import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True, help="data file")
args = vars(ap.parse_args())

file = args["f"]

def write(string):
    os.system("xdotool type " + string)
    time.sleep(0.05)

def enter(n = 1):
    for i in range(0, n):
        os.system("xdotool key Return")
        os.system("xdotool key Return")
        time.sleep(0.05)

def unenter(n = 1):
    for i in range(0, n):
        os.system("xdotool keydown Control")
        os.system("xdotool key Up")
        os.system("xdotool keyup Control")
        time.sleep(0.05)

def tab(n = 1):
    for i in range(0, n):
        os.system("xdotool key Tab")
        time.sleep(0.05)

def untab(n = 1):
    for i in range(0, n):
        os.system("xdotool keydown Shift")
        os.system("xdotool key Tab")
        os.system("xdotool keyup Shift")
        time.sleep(0.05)

time.sleep(5)
for line in open(file).read().split('\n'):
    if len(line) > 0:
        mm, fc = line[1:-2].split(', ')

        enter()
        write(fc)










