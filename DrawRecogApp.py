# Copyright (c) [2023] [Naomi Arroyo]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageDraw
from ttkthemes import ThemedTk

# Load the trained model
print("Loading the trained model...")
model = tf.keras.models.load_model(r'C:\yourpathgoeshere\DrawingRecog_model.h5')

# Initialize variables for drawing
drawing = False
last_x = 0
last_y = 0


def predict_shape():
    print("Predicting the shape...")

    # Create a blank PIL image
    canvas_image = Image.new("RGB", (280, 280), "white")

    # Transfer the drawing from tkinter canvas to the PIL image
    draw = ImageDraw.Draw(canvas_image)
    for item in canvas.find_all():
        x0, y0, x1, y1 = canvas.coords(item)
        draw.line([(x0, y0), (x1, y1)], fill="black", width=5)

    # Process the image for prediction
    canvas_image = canvas_image.resize((28, 28), Image.ANTIALIAS).convert('L')
    canvas_np = np.array(canvas_image)
    canvas_np = cv2.bitwise_not(canvas_np)  # Invert colors
    canvas_np = canvas_np / 255.0

    # Make the prediction
    prediction = model.predict(canvas_np.reshape(1, 28, 28))
    predicted_label = np.argmax(prediction)

    shape_names = ["square", "circle", "triangle", "rhombus"]
    predicted_shape = shape_names[predicted_label]

    # Display the prediction
    messagebox.showinfo("Prediction", f"The model predicts this is a: {predicted_shape}")

    # Ask for user feedback
    is_correct = messagebox.askyesno("Feedback", f"Is the drawn shape a {predicted_shape}?")

    if is_correct:
        print(f"Model was correct! It's a {predicted_shape}.")
    else:
        print(f"Model was wrong. It thought it was a {predicted_shape}.")


def clear_canvas():
    print("Clearing the canvas...")
    canvas.delete("all")


def on_mouse_down(event):
    global drawing, last_x, last_y
    print("Mouse button pressed.")
    drawing = True
    last_x = event.x
    last_y = event.y


def on_mouse_move(event):
    global drawing, last_x, last_y
    if drawing:
        print("Drawing on canvas...")
        canvas.create_line(last_x, last_y, event.x, event.y, width=5)
        last_x = event.x
        last_y = event.y


def on_mouse_up(event):
    global drawing
    print("Mouse button released.")
    drawing = False


# Set up the themed GUI window
print("Setting up the GUI window...")
app = ThemedTk(theme="arc")
app.title("Shape Predictor")

title_label = ttk.Label(app, text="Draw & Predict Shapes", font=("Arial", 20, "bold"))
title_label.pack(pady=20)

canvas_frame = ttk.Frame(app, padding=10)
canvas_frame.pack(pady=20)

canvas = tk.Canvas(canvas_frame, bg="white", width=280, height=280)
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

predict_button = ttk.Button(canvas_frame, text="Predict Shape", command=predict_shape)
predict_button.grid(row=1, column=0, sticky=tk.W + tk.E, padx=10, pady=5)

clear_button = ttk.Button(canvas_frame, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=1, sticky=tk.W + tk.E, padx=10, pady=5)

# Attach mouse event listeners to canvas
print("Binding mouse events...")
canvas.bind('<Button-1>', on_mouse_down)
canvas.bind('<B1-Motion>', on_mouse_move)
canvas.bind('<ButtonRelease-1>', on_mouse_up)

print("Starting the main GUI loop...")
app.mainloop()