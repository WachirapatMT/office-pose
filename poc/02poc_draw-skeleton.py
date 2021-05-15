WIDTH = 640
HEIGHT = 480
CAMERA = 0
FRAMERATE = 24
RESOLUTION = 0.4

import tkinter as tk
import PIL.Image, PIL.ImageTk

import cv2
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
width_resized = int(WIDTH * RESOLUTION // 16) * 16
height_resized = int(HEIGHT * RESOLUTION // 16) * 16

import threading
import time
class StoppableThread(threading.Thread):
  def __init__(self):
    super().__init__()
    self._stop_event = threading.Event()
  def stop(self):
    self._stop_event.set()
  def stopped(self):
    return self._stop_event.is_set()

from processor import Processor
import base64
import torch
import argparse
processor_args = argparse.Namespace()
processor_args.seed_threshold = 0.5
processor_args.instance_threshold = 0.2
processor_args.keypoint_threshold = None
processor_args.decoder_workers = None
processor_args.experimental_decoder = False
processor_args.extra_coupling = 0.0
processor_args.force_complete_pose = True
processor_args.debug_pif_indices = []
processor_args.debug_paf_indices = []
processor_args.debug_file_prefix = None
processor_args.profile_decoder = None
processor_args.fixed_b = None
processor_args.pif_fixed_scale = None
processor_args.pif_th = 0.1
processor_args.paf_th = 0.1
processor_args.connection_method = 'blend'
processor_args.checkpoint = None
processor_args.basenet = None
processor_args.headnets = ['pif', 'paf']
processor_args.pretrained = True
processor_args.two_scale = False
processor_args.multi_scale = False
processor_args.multi_scale_hflip = True
processor_args.cross_talk = 0.0
processor_args.head_dropout = 0.0
processor_args.head_quad = 0
processor_args.resolution = 1
processor_args.video = None
processor_args.joints = False
processor_args.skeleton = True
processor_args.compare = True
processor_args.device = torch.device(type='cpu')

from common import SKELETON_CONNECTIONS

class PoseThread(StoppableThread):
  def __init__(self, application=None):
    super().__init__()
    self.application = application
    self.processor = Processor((width_resized,height_resized), processor_args)
  def run(self):
    while not self.stopped():
      if(hasattr(self.application, 'cv2image')):
        keypoint_sets, _, _ = self.processor.single_image(
          b64image=base64.b64encode(cv2.imencode('.jpg', self.application.cv2image)[1]).decode('UTF-8')
        )
        # print(keypoint_sets)
        self.application.keypoint_sets = keypoint_sets

class Application(tk.Frame):
  def __init__(self, master=None):
    super().__init__(master)
    self.master = master
    self.keypoint_sets = None
    self.keypoint_ovals = None
    self.skeleton_lines = None
    self.pack()
    self.create_widgets()
    self.master.bind('<Escape>', self.terminate)
    self.update()

    self.poseThread = PoseThread(self)
    self.poseThread.start()

  def terminate(self, e):
    self.poseThread.stop()
    self.quit()

  def create_widgets(self):
    self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT)
    self.canvas.pack()
    self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)
  def update(self):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if frame is not None:
      self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.cv2image))
      self.canvas.itemconfig(self.img_on_canvas, image=self.photo)
    if self.keypoint_sets is not None and len(self.keypoint_sets) > 0:
      if self.skeleton_lines is None:
        self.skeleton_lines = [None]*len(SKELETON_CONNECTIONS)
      for i, (p1_index, p2_index, _) in enumerate(SKELETON_CONNECTIONS):
        p1 = (self.keypoint_sets[0][p1_index][0] * WIDTH, self.keypoint_sets[0][p1_index][1] * HEIGHT)
        p2 = (self.keypoint_sets[0][p2_index][0] * WIDTH, self.keypoint_sets[0][p2_index][1] * HEIGHT)
        if(self.skeleton_lines[i] is None):
          self. skeleton_lines[i] = self.canvas.create_line(p1, p2, fill='blue', width=8)
        else:
          self.canvas.coords(self.skeleton_lines[i], p1[0], p1[1], p2[0], p2[1])

      if self.keypoint_ovals is None:
        self.keypoint_ovals = [None]*len(self.keypoint_sets[0])
      for i, keypoint in enumerate(self.keypoint_sets[0]):
        x, y = keypoint[0], keypoint[1]
        if(self.keypoint_ovals[i] is None):
          self.keypoint_ovals[i] = self.canvas.create_oval((x*WIDTH-5,y*HEIGHT-5,x*WIDTH+5,y*HEIGHT+5), fill='red')
        else:
          self.canvas.coords(self.keypoint_ovals[i], x*WIDTH-5, y*HEIGHT-5,x*WIDTH+5,y*HEIGHT+5)
    self.after(1000//FRAMERATE, self.update)
root = tk.Tk()
app = Application(master=root)
app.mainloop()