import tkinter as tk

from common import cli
from gui_app import Application
from gui_component import HoverButton, Modal
from exercise import EXERCISE

######## Init ########
root = tk.Tk()
root.title("Office Pose")
root.geometry("1400x650")

container = tk.Frame(root)
container.pack(side="top", fill="both", expand=True)

app = Application(master=root, args=cli())
app.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

mainPage = tk.Frame(root)
mainPage.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

######## Function ########
def onStart():
    app.start(list(EXERCISE.keys()))
    app.lift()


def onBack():
    app.pause()
    mainPage.lift()


def onClose():
    app.terminate()


def onFreePlay():
    input = tk.StringVar()
    modal = Modal(
        root,
        input,
        "Select Exercise",
        [ex.replace("_", " ").capitalize() for ex in EXERCISE.keys()],
    )
    root.wait_window(modal.top)
    if len(input.get()) > 0:
        app.start([input.get().replace(" ", "_").lower()])
        app.lift()


def onShowSkeleton():
    app.setShowSkeleton(showSkeleton.get())


########## App ##########
backButton = HoverButton(app, text="Back", padx=10, pady=2, command=onBack, font="14")
backButton.place(relx=0.5, y=610, anchor=tk.CENTER)

app.scoreLabel = scoreLabel = tk.Label(app, text="Loading...", font="Helvetica 18 bold")
scoreLabel.place(relx=0.5, y=550, anchor=tk.CENTER)

showSkeleton = tk.IntVar()
skeletonCheckBox = tk.Checkbutton(
    app,
    text="Show Skeleton",
    variable=showSkeleton,
    onvalue=1,
    offvalue=0,
    command=onShowSkeleton,
    font="Helvetica 10 bold",
)
skeletonCheckBox.place(x=1385, y=640, anchor=tk.SE)

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