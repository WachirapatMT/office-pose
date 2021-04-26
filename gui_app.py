import tkinter as tk
import PIL.Image, PIL.ImageTk

import cv2
import numpy as np
import torch
from scipy.spatial.distance import euclidean

from exercise_compare import cli
from common import CocoPart, SKELETON_CONNECTIONS, write_on_image, visualise, normalise
from processor import Processor
from exercise import EXERCISE

import time
import base64
import os
from itertools import chain

WIDTH = 640
HEIGHT = 480
COUNTDOWN = 5

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
            if hasattr(self.application, "cv2image"):
                (
                    keypoint_sets,
                    scores,
                    width_height,
                ) = self.application.processor_singleton.single_image(
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
        self.init_processor()

        self.poseThread = PoseThread(self)
        self.poseThread.start()

    def init_processor(self):
        # Resize image to multiple of 16 due to some unknown convention
        width_height = (
            int(WIDTH * self.args.resolution // 16) * 16,
            int(HEIGHT * self.args.resolution // 16) * 16,
        )
        if torch.cuda.is_available():
            self.args.device = torch.device("cuda")
            print('Using CUDA')
        else:
            self.args.device = torch.device("cpu")
            print('Using CPU')

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

    def start(self, exercise_list=["side_bend_right"]):
        self.is_finish = False
        self.countdown = COUNTDOWN
        self.exercise_list = exercise_list
        self.score = 0
        self.init_cap()
        self.create_widgets()
        self.init_exercise(self.exercise_list.pop(0))
        self.update()
        threading.Thread(target=self.thread_countdown).start()

    def start_next_exercise(self):
        if len(self.exercise_list) == 0:
            self.pause()
            self.finish()
        else:
            if self._job is not None:
                self.after_cancel(self._job)
            self.is_finish = False
            self.countdown = COUNTDOWN
            self.exercise_canvas.destroy()
            self.init_exercise(self.exercise_list.pop(0))
            self.scoreLabel["text"] = "Loading..."
            self.update()
            threading.Thread(target=self.thread_countdown).start()

    def pause(self):
        self.countdown = 0
        if self._job is not None:
            self.after_cancel(self._job)
        self.cap.release()

    def finish(self):
        self.scoreLabel['text'] = "Congratulation, you've finished the exercise"
        self.canvas.destroy()
        self.exercise_canvas.destroy()

        self.exercise_img = cv2.imread(
            os.path.join("exercise_images", "finish.jpg")
        )
        self.exercise_img = cv2.cvtColor(self.exercise_img, cv2.COLOR_BGR2RGB)
        self.exercise_img = cv2.resize(
            self.exercise_img,
            (self.exercise_img.shape[1] * HEIGHT // self.exercise_img.shape[0], HEIGHT),
        )
        canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, bg="#eeeeee")
        canvas.place(relx=0.5, y=270, anchor=tk.CENTER)
        self.exercise_img_on_canvas = canvas.create_image(
            WIDTH // 2, 0, anchor=tk.N
        )
        self.exercise_photo = PIL.ImageTk.PhotoImage(
            PIL.Image.fromarray(self.exercise_img)
        )
        canvas.itemconfig(
            self.exercise_img_on_canvas, image=self.exercise_photo
        )

    def terminate(self, e=None):
        self.countdown = 0
        self.poseThread.stop()
        self.quit()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT)
        self.canvas.place(x=1030, y=270, anchor=tk.CENTER)
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if self.is_finish == True:
            self.countdown = COUNTDOWN
            self.is_finish = False
            self.score = 0
            # self.pause()
            self.start_next_exercise()
        elif frame is not None:
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

                miss_weight_list = list(map(lambda x: [1,1] if x[2] >= 1e-5 else [10,10],keypoint_sets[0]))
                pose_weight_list = list(map(lambda x: [x[2],x[2]] if x[2] <= 1 else [1,1],self.exercise[0]))
                weight_list = list(a*b for (a,b) in zip(list(chain(*miss_weight_list)),list(chain(*pose_weight_list))))
                normalize_weight = sum(weight_list)

                score = euclidean(list(chain(*my_pose_norm[0])), list(chain(*exercise_pose_norm[0])),weight_list)/ (normalize_weight if normalize_weight!=0 else 1)
                max_score = euclidean([0]*len(list(chain(*exercise_pose_norm[0]))), list(chain(*exercise_pose_norm[0])),weight_list)/ (normalize_weight if normalize_weight!=0 else 1)
                self.score = 10*(1-(score/max_score)**0.75)

                self.scoreLabel["text"] = "Score: {:.4f} [{:.0f}]".format(
                    self.score,
                    self.countdown,
                )

                if self.args.compare:
                    # comment out to use real image as a background
                    self.cv2image = np.ones((480, 640, 3), np.uint8) * 255

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

    def thread_countdown(self):
        while self.countdown > 0:
            # print(f"Count down {self.countdown}")
                time.sleep(0.1)
                if self.score>7:
                    self.countdown -= 0.1
        self.is_finish = True
