import tkinter as tk


class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self["background"] = "#c2c2c2"

    def on_leave(self, e):
        self["background"] = self.defaultBackground


class Modal:
    def __init__(self, master, input, title, choice):
        self.input = input

        self.top = tk.Toplevel(master)
        self.top.geometry("310x150")
        self.top.transient(master)
        self.top.grab_set()
        self.top.title(title)
        self.top.bind("<Return>", self.ok)
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)

        self.label = tk.Label(
            self.top, text="Select an exercise", font="Helvetica 12 bold"
        )
        self.label.place(relx=0.5, y=30, anchor=tk.CENTER)

        op = tk.OptionMenu(self.top, self.input, *choice)
        op.config(width=20, font="Helvetica 12")
        op.place(relx=0.5, y=70, anchor=tk.CENTER)
        menu = self.top.nametowidget(op.menuname)
        menu.config(font="Helvetica 12")

        button = HoverButton(
            self.top, width=8, text="OK", command=self.ok, font="Helvetica 10"
        )
        button.place(relx=0.5, y=110, anchor=tk.CENTER)

    def ok(self, event=None):
        if len(self.input.get()) == 0:
            self.label["text"] = "Please select one exercise"
            return
        self.top.destroy()

    def cancel(self, event=None):
        self.top.destroy()