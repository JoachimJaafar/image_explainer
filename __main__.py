import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

from explainer import Explainer

import os
import sys

app = tk.Tk() 
w = app.winfo_reqwidth()
h = app.winfo_reqheight()
ws = app.winfo_screenwidth()
hs = app.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
app.geometry('800x200+%d+%d' % (x, y))
app.resizable(width=0, height=0)

def on_close():
    if os.path.exists('./tmp'):
        os.rmdir('./tmp')
    app.destroy()

app.protocol("WM_DELETE_WINDOW",  on_close)

labelTop = tk.Label(app,
                    text = "Choose the model to use.")

values=[
    'alexnet', 
    'resnet18', 
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
    'squeezenet1_0',
    'squeezenet1_1',
    'inception_v3',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'googlenet',
    'mobilenet_v2',
    'mnasnet0_5',
    'mnasnet1_0',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0'
]
comboExample = ttk.Combobox(app, state="readonly", values=values)
comboExample.current(1)

is_running = False

def run_main():
    global is_running
    if not is_running:
        is_running = True
        file_path = askopenfilename()
        if file_path == '' or type(file_path) != tuple:
            labelTop.configure(text="Running ...")
            labelTop.update_idletasks()
            prediction = Explainer(comboExample.get(), values).main(file_path)
            labelTop.configure(text=prediction)
        is_running = False
    


button = tk.Button(app, text ="Explain image", command = run_main)

labelTop.pack(expand=True)
comboExample.pack(expand=True)
button.pack(expand=True)

try:
    app.mainloop()
except KeyboardInterrupt:
    if os.path.exists('./tmp'):
        os.rmdir('./tmp')
    app.destroy()