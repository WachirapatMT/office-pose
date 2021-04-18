import tkinter as tk
import PIL.Image, PIL.ImageTk

import cv2

import numpy as np
import openpifpaf
import torch
from scipy.spatial.distance import euclidean

from exercise_compare import cli
from common import CocoPart, SKELETON_CONNECTIONS, write_on_image, visualise, normalise
from processor import Processor
from exercise import EXERCISE

import base64
import os
from itertools import chain

WIDTH = 640
HEIGHT = 480

import threading
class StoppableThread(threading.Thread):
  def __init__(self):
    super().__init__()
    self._stop_event = threading.Event()
  def stop(self):
    self._stop_event.set()
  def stopped(self):
    return self._stop_event.is_set()

class PoseThread(StoppableThread):
  def __init__(self, application=None):
    super().__init__()
    self.application = application
  def run(self):
    while not self.stopped():
      if(hasattr(self.application, 'cv2image')):
        keypoint_sets, scores, width_height = self.application.processor_singleton.single_image(
            b64image=base64.b64encode(
                cv2.imencode(".jpg", self.application.cv2image)[1]
            ).decode("UTF-8")
        )
        # print(keypoint_sets)
        self.application.keypoint_sets = keypoint_sets


class Application(tk.Frame):
    def __init__(self, master=None, args=cli()):
        super().__init__(master)
        self.master = master
        self.args = args

        self.keypoint_sets = None
        self.keypoint_ovals = None
        self._job = None
        self.cap = None
        self.scoreLabel = None

        self.master.bind("<Escape>", self.terminate)
        self.create_widgets()
        self.init_processor()

        self.poseThread = PoseThread(self)
        self.poseThread.start()

    def init_processor(self):
        # Resize image to multiple of 16 due to some unknown convention
        width_height = (
            int(WIDTH * self.args.resolution // 16) * 16,
            int(HEIGHT * self.args.resolution // 16) * 16,
        )
        # Initialise model
        self.processor_singleton = Processor(width_height, self.args)

    def init_exercise(self, exercise):
        self.exercise = EXERCISE[exercise]
        self.exercise_img = cv2.imread(
            os.path.join("exercise_images", f"{exercise}.png")
        )
        self.exercise_img = cv2.cvtColor(self.exercise_img, cv2.COLOR_BGR2RGB)
        self.exercise_img = cv2.resize(
            self.exercise_img,
            (self.exercise_img.shape[1] * HEIGHT // self.exercise_img.shape[0], HEIGHT),
        )
        self.exercise_canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, bg="#cccccc")
        self.exercise_canvas.place(x=370, y=270, anchor=tk.CENTER)
        self.exercise_img_on_canvas = self.exercise_canvas.create_image(
            WIDTH // 2, 0, anchor=tk.N
        )
        self.exercise_photo = PIL.ImageTk.PhotoImage(
            PIL.Image.fromarray(self.exercise_img)
        )
        self.exercise_canvas.itemconfig(
            self.exercise_img_on_canvas, image=self.exercise_photo
        )

    def init_cap(self):
        self.cap = cv2.VideoCapture(int(self.args.video))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def start(self, exercise="side_bend_right"):
        self.init_cap()
        self.init_exercise(exercise)
        self.update()

    def pause(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self.cap.release()

    def terminate(self, e=None):
        self.poseThread.stop()
        self.quit()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT)
        self.canvas.place(x=1030, y=270, anchor=tk.CENTER)
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if frame is not None:
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # keypoint_sets, scores, width_height = self.processor_singleton.single_image(
            #     b64image=base64.b64encode(
            #         cv2.imencode(".jpg", self.cv2image)[1]
            #     ).decode("UTF-8")
            # )
            if not self.args.compare and self.keypoint_sets is not None:
                self.cv2image = visualise(
                    img=self.cv2image,
                    keypoint_sets=self.keypoint_sets,
                    width=WIDTH,
                    height=HEIGHT,
                    vis_keypoints=self.args.joints,
                    vis_skeleton=self.args.skeleton,
                )

            try:
                keypoint_sets = self.keypoint_sets

                my_pose = [list(map(lambda x: [x[0], x[1]], keypoint_sets[0]))]
                exercise_pose = [list(map(lambda x: [x[0], x[1]], self.exercise[0]))]

                my_pose_norm = normalise(my_pose)
                exercise_pose_norm = normalise(exercise_pose)

                self.scoreLabel["text"] = "Score: {:.4f}".format(
                    euclidean(list(chain(*my_pose[0])), list(chain(*exercise_pose[0])))
                )

                if self.args.compare:
                    # comment out to use real image as a background
                    self.cv2image = np. ones((480, 640, 3), np.uint8) * 255

                    self.cv2image = visualise(
                        img=self.cv2image,
                        keypoint_sets=my_pose_norm,
                        width=WIDTH // 12,
                        height=HEIGHT // 12,
                        tranX=400,
                        tranY=110,
                        vis_keypoints=self.args.joints,
                        vis_skeleton=True,
                    )
                    self.cv2image = visualise(
                        img=self.cv2image,
                        keypoint_sets=exercise_pose_norm,
                        width=WIDTH // 12,
                        height=HEIGHT // 12,
                        tranX=150,
                        tranY=110,
                        vis_keypoints=self.args.joints,
                        vis_skeleton=True,
                    )

            except Exception as err:
                print("Error:", err)
                pass

            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.cv2image))
            self.canvas.itemconfig(self.img_on_canvas, image=self.photo)

        self._job = self.after(1000 // 24, self.update)

    def setShowSkeleton(self, isShow):
        self.args.skeleton = isShow
        self.args.joints = isShow
