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

from gui_component import HoverButton, Modal

args = cli()

# Get exercise keypoints
exercise = EXERCISE[args.exercise]
exercise_img = cv2.imread(os.path.join("exercise_images", f"{args.exercise}.png"))
exercise_img = cv2.cvtColor(exercise_img, cv2.COLOR_BGR2RGB)
height, width = (480, 640)
exercise_img = cv2.resize(
    exercise_img, (exercise_img.shape[0] * 640 // exercise_img.shape[1], 480)
)

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

        self.create_widgets()
        self.master.bind("<Escape>", self.terminate)
        self._job = None
        self.cap = None
        self.scoreLabel = None

    def initCap(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def start(self):
        self.initCap()
        self.update()

    def pause(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self.cap.release()

    def terminate(self, e=None):
        self.quit()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=640, height=480)
        self.canvas.place(x=1030, y=270, anchor=tk.CENTER)
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.exercise_canvas = tk.Canvas(self, width=640, height=480, bg="#eeeeee")
        self.exercise_canvas.place(x=370, y=270, anchor=tk.CENTER)
        self.exercise_img_on_canvas = self.exercise_canvas.create_image(
            320, 0, anchor=tk.N
        )
        self.exercise_photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(exercise_img))
        self.exercise_canvas.itemconfig(
            self.exercise_img_on_canvas, image=self.exercise_photo
        )

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if frame is not None:
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # keypoint_sets, scores, width_height = processor_singleton.single_image(
            #     b64image=base64.b64encode(
            #         cv2.imencode(".jpg", self.cv2image)[1]
            #     ).decode("UTF-8")
            # )
            # self.cv2image = visualise(
            #     img=self.cv2image,
            #     keypoint_sets=keypoint_sets,
            #     width=width,
            #     height=height,
            #     vis_keypoints=args.joints,
            #     vis_skeleton=args.skeleton,
            # )

            # try:
            #     my_pose = [list(map(lambda x: [x[0], x[1]], keypoint_sets[0]))]
            #     exercise_pose = [list(map(lambda x: [x[0], x[1]], exercise[0]))]

            #     my_pose_norm = normalise(my_pose)
            #     exercise_pose_norm = normalise(exercise_pose)

            #     self.scoreLabel["text"] = "Score: {:.4f}".format(
            #         euclidean(list(chain(*my_pose[0])), list(chain(*exercise_pose[0])))
            #     )

            # except Exception as err:
            #     print("Error:", err)
            #     pass

            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.cv2image))
            self.canvas.itemconfig(self.img_on_canvas, image=self.photo)

        self._job = self.after(1000 // 24, self.update)


######## Init ########
root = tk.Tk()
root.title("Office Pose")
root.geometry("1400x650")

container = tk.Frame(root)
container.pack(side="top", fill="both", expand=True)

app = Application(master=root)
app.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

mainPage = tk.Frame(root)
mainPage.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

######## Function ########
def onStart():
    app.start()
    app.lift()


def onBack():
    app.pause()
    mainPage.lift()


def onClose():
    app.terminate()


def onFreePlay():
    input = tk.StringVar()
    modal = Modal(root, input, "Select Exercise", ["a", "b", "c", "d"])
    root.wait_window(modal.top)
    print(input.get())


########## App ##########
backButton = HoverButton(app, text="Back", padx=10, pady=2, command=onBack, font="14")
backButton.place(relx=0.5, y=610, anchor=tk.CENTER)

app.scoreLabel = scoreLabel = tk.Label(app, text="Loading...", font="Helvetica 18 bold")
scoreLabel.place(relx=0.5, y=550, anchor=tk.CENTER)

######### Main ##########
mainLabel = tk.Label(mainPage, text="Welcome to Office Pose!", font="Helvetica 26 bold")
mainLabel.place(relx=0.5, y=150, anchor=tk.CENTER)

startButton = HoverButton(
    mainPage, text="Start", pady=2, width=15, command=onStart, font="Helvetica 16"
)
startButton.place(relx=0.5, y=270, anchor=tk.CENTER)

freePlayButton = HoverButton(
    mainPage,
    text="Free Play",
    pady=2,
    width=15,
    command=onFreePlay,
    font="Helvetica 16",
)
freePlayButton.place(relx=0.5, y=325, anchor=tk.CENTER)

exitButton = HoverButton(
    mainPage, text="Exit", pady=2, width=15, command=onClose, font="Helvetica 16"
)
exitButton.place(relx=0.5, y=375, anchor=tk.CENTER)

######### Run ##########
root.protocol("WM_DELETE_WINDOW", onClose)
root.mainloop()