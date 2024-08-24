from tkinter import *
from tkinter.colorchooser import askcolor
import threading

from PIL import Image, ImageGrab
import pyautogui
import pygetwindow as gw
import io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.adamw import adamw
from yolo.mnist_train_simple_yolo import SimpleYOLO, transform
import torch.nn.functional as F


class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.geometry("600x600")
        self.root.title("Paint Window")
        self.window_title = "Paint Window"

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        # self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        # self.brush_button.grid(row=0, column=1)

        # self.color_button = Button(self.root, text='color', command=self.choose_color)
        # self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.clear_button = Button(self.root, text='clear', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=2)

        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Escape>', self.close_window)
        self.root.after(100, self.print_window_size)
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.objects = []
        self.screenshot_img = None

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def clear_canvas(self):
        self.c.delete("all")
        self.objects.clear()

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            obj_id = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                        width=self.line_width, fill=paint_color,
                                        capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.objects.append(obj_id)  # Store the object's ID for undo

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def undo(self, event=None):
        if self.objects:
            # Remove the last object added to the canvas
            last_object = self.objects[max(0, len(self.objects) - 50):len(self.objects)]
            for item in last_object:
                self.c.delete(item)
            self.objects = self.objects[0: len(self.objects) - 50]

    def get_window_size(self):
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        return width, height

    def print_window_size(self):
        size = self.get_window_size()
        print(f"Window size: {size}")

    def close_window(self, event=None):
        self.root.destroy()

    def predict(self):
        # Hide the button grid
        self.pen_button.grid_forget()
        self.eraser_button.grid_forget()
        self.clear_button.grid_forget()
        self.predict_button.grid_forget()
        self.choose_size_button.grid_forget()

        self.root.update()  # Force update
        self.root.update_idletasks()  # Ensure all events are processed

        left = self.root.winfo_rootx()
        top = self.root.winfo_rooty()
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        bbox = (left, top, width, height)
        img = pyautogui.screenshot(region=bbox)
        self.screenshot_img = img  # image is stored here
        print(self.screenshot_img)

        # Show the button grid again
        self.pen_button.grid(row=0, column=0)
        self.eraser_button.grid(row=0, column=3)
        self.clear_button.grid(row=0, column=2)
        self.predict_button.grid(row=0, column=1)
        self.choose_size_button.grid(row=0, column=4)

        self.run_inference()

    def run_inference(self):
        model = SimpleYOLO()
        model.load_state_dict(torch.load('cnn_deep_model.pth', weights_only=True))
        model.eval()

        input_tensor = transform(self.screenshot_img)  # grayscale and totensor
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)

        output = output.squeeze(0)  # Remove batch dimension
        print(output.shape)
        predictions = list(output)
        predictions = output[4:len(output)]
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(predictions)

        plt.clf()
        plt.bar(classes, predictions, color='skyblue')
        plt.show()


if __name__ == '__main__':
    paint_app = Paint()
