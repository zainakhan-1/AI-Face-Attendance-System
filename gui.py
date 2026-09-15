import tkinter as tk
from tkinter import simpledialog, messagebox
import os
from capture_faces import capture_faces
from train_model import train_model
from recognize_face import recognize_and_mark

def handle_capture():
    name = simpledialog.askstring("Student Name", "Enter student name:")
    if name:
        capture_faces(name)
        messagebox.showinfo("Success", f"Faces captured for {name}")

def handle_train():
    train_model()
    messagebox.showinfo("Success", "Training completed.")

def handle_recognize():
    recognize_and_mark()

app = tk.Tk()
app.title("Face Attendance System")
app.geometry("300x250")

tk.Label(app, text="Face Attendance System", font=("Helvetica", 16)).pack(pady=10)
tk.Button(app, text="Capture Faces", command=handle_capture, width=25).pack(pady=5)
tk.Button(app, text="Train Model", command=handle_train, width=25).pack(pady=5)
tk.Button(app, text="Recognize & Mark Attendance", command=handle_recognize, width=25).pack(pady=5)
tk.Button(app, text="Exit", command=app.quit, width=25).pack(pady=10)

app.mainloop()
