import tkinter as tk
import PIL.Image, PIL.ImageTk

import cv2

import threading
import time

from exercise_compare import cli

import numpy as np
import openpifpaf
import torch
from scipy.spatial.distance import euclidean

from common import CocoPart, SKELETON_CONNECTIONS, write_on_image, visualise, normalise
from processor import Processor
from exercise import EXERCISE

import argparse
import base64
import csv
import time
import os
from itertools import chain

args = cli()

# Get exercise keypoints
exercise = EXERCISE[args.exercise]
exercise_img = cv2.imread(os.path.join("exercise_images", f"{args.exercise}.png"))
exercise_img = cv2.cvtColor(exercise_img, cv2.COLOR_BGR2RGB)

height, width = (480, 640)
exercise_img = cv2.resize(exercise_img, (640, 480))

# Resize image to multiple of 16 due to some unknown convention
width_height = (
    int(width * args.resolution // 16) * 16,
    int(height * args.resolution // 16) * 16,
)
print(f"Resize image from {(width, height)} to {width_height}")

# Initialise model
processor_singleton = Processor(width_height, args)


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.keypoint_sets = None
        self.keypoint_ovals = None
        self.pack()
        self.create_widgets()
        self.master.bind("<Escape>", self.terminate)
        self._job = None
        self.cap = None
        self.initCap()
        self.scoreLabel = None

    def initCap(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def start(self):
        self.update()
        self.initCap()

    def pause(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self.cap.release()

    def terminate(self, e=None):
        self.quit()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=640 * 2, height=480)
        self.canvas.pack()
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if frame is not None:
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoint_sets, scores, width_height = processor_singleton.single_image(
                b64image=base64.b64encode(
                    cv2.imencode(".jpg", self.cv2image)[1]
                ).decode("UTF-8")
            )
            self.cv2image = visualise(
                img=self.cv2image,
                keypoint_sets=keypoint_sets,
                width=width,
                height=height,
                vis_keypoints=args.joints,
                vis_skeleton=args.skeleton,
            )
            self.cv2image = np.hstack((exercise_img, self.cv2image))
            try:
                my_pose = [list(map(lambda x: [x[0], x[1]], keypoint_sets[0]))]
                exercise_pose = [list(map(lambda x: [x[0], x[1]], exercise[0]))]

                my_pose_norm = normalise(my_pose)
                exercise_pose_norm = normalise(exercise_pose)

                self.scoreLabel["text"] = str(
                    euclidean(list(chain(*my_pose[0])), list(chain(*exercise_pose[0])))
                )

            except Exception as err:
                print("Error:", err)
                pass

            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.cv2image))
            self.canvas.itemconfig(self.img_on_canvas, image=self.photo)

        self._job = self.after(1000 // 24, self.update)


root = tk.Tk()
root.title("Office Pose")
root.geometry("1400x600")

container = tk.Frame(root)
container.pack(side="top", fill="both", expand=True)

app = Application(master=root)
app.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

mainPage = tk.Frame(root)
mainPage.place(in_=container, x=0, y=0, relwidth=1, relheight=1)


def handleEnter():
    app.start()
    app.lift()


def handleBack():
    app.pause()
    mainPage.lift()


def onClose():
    app.terminate()


backButton = tk.Button(app, text="Back", padx=10, pady=2, command=handleBack)
backButton.place(relx=0.5, y=520, anchor="center")

app.scoreLabel = scoreLabel = tk.Label(app, text="Init")
scoreLabel.place(relx=0.5, y=550, anchor="center")

mainLabel = tk.Label(mainPage, text="Welcome to Office Pose")
mainLabel.place(relx=0.5, y=110, anchor="center")
mainButton = tk.Button(mainPage, text="Exercise", padx=10, pady=2, command=handleEnter)
mainButton.place(relx=0.5, y=175, anchor="center")

root.protocol("WM_DELETE_WINDOW", onClose)
root.mainloop()