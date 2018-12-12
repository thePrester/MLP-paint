from tkinter import messagebox
from mlp import *
import util as ut
import tkinter as tk
import os

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 550
SUPPORTED_CLASSES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']

class DatasetGUI():
    def __init__(self, master):
        self.master = master
        master.title("Drawing GUI")

        self.points = list()
        self.model = None
        self.selected_class = tk.StringVar()
        self.selected_class.set('alpha')

        self.label = tk.Label(master, text="Press left-click and hold to draw")
        self.label.pack()

        self.main_frame = tk.Frame(master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.main_frame.pack(side=tk.BOTTOM)

        self.clear_button = tk.Button(self.main_frame, text="Clear", command=self.clear)
        self.clear_button.grid(row=0, column=0)

        self.close_button = tk.Button(self.main_frame, text="Close", command=master.quit)
        self.close_button.grid(row=0, column=1)

        self.save_button = tk.Button(self.main_frame, text="Save", command=self.save_features)
        self.save_button.grid(row=0, column=2)

        self.path_label = tk.Label(self.main_frame, text="File path:")
        self.path_label.grid(row=0, column=3)

        self.path_entry = tk.Entry(self.main_frame)
        self.path_entry.grid(row=0, column=4)

        self.m_label = tk.Label(self.main_frame, text="Number of points:")
        self.m_label.grid(row=0, column=5)

        self.m_entry = tk.Entry(self.main_frame)
        self.m_entry.grid(row=0, column=6)

        self.class_dropdown = tk.OptionMenu(self.main_frame, self.selected_class, *SUPPORTED_CLASSES)
        self.class_dropdown.grid(row=0, column=7)

        self.opt_label = tk.Label(self.main_frame, text="Optimization method:")
        self.opt_label.grid(row=1, column=0)

        self.opt_entry = tk.Entry(self.main_frame)
        self.opt_entry.grid(row=1, column=1)

        self.epochs_label = tk.Label(self.main_frame, text="Number of epochs:")
        self.epochs_label.grid(row=1, column=2)

        self.epochs_entry = tk.Entry(self.main_frame)
        self.epochs_entry.grid(row=1, column=3)

        self.batch_label = tk.Label(self.main_frame, text="Batch size:")
        self.batch_label.grid(row=1, column=4)

        self.batch_entry = tk.Entry(self.main_frame)
        self.batch_entry.grid(row=1, column=5)

        self.lr_label = tk.Label(self.main_frame, text="Learning rate:")
        self.lr_label.grid(row=1, column=6)

        self.lr_entry = tk.Entry(self.main_frame)
        self.lr_entry.grid(row=1, column=7)

        self.architecture_label = tk.Label(self.main_frame, text="Architecture:")
        self.architecture_label.grid(row=1, column=8)

        self.architecture_entry = tk.Entry(self.main_frame)
        self.architecture_entry.grid(row=1, column=9)

        self.predict_button = tk.Button(self.main_frame, text="Predict", command=self.predict_handle)
        self.predict_button.grid(row=2, column=4)

        self.train_button = tk.Button(self.main_frame, text="Train", command=self.train_handle)
        self.train_button.grid(row=2, column=3)

        self.init_button = tk.Button(self.main_frame, text="Init", command=self.init_handle)
        self.init_button.grid(row=2, column=2)

        self.drawing_canvas = tk.Canvas(master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.drawing_canvas.configure(background="white")
        self.drawing_canvas.bind("<B1-Motion>", self.draw_point)
        self.drawing_canvas.pack(expand=tk.YES, fill=tk.BOTH)

    def draw_point(self, event):
        self.points.append((event.x, event.y))
        x1, y1 = event.x - 1, event.y - 1
        x2, y2 = event.x + 1, event.y + 1
        self.drawing_canvas.create_oval(x1, y1, x2, y2, fill='black')

    def train_handle(self):
        if self.model is None:
            messagebox.showinfo(title="Model not initialized", message="Model must be initialized first")
            return
        try:
            dataset_path = self.path_entry.get()
            with open(dataset_path) as fp:
                dataset = [(np.array(line.split(';')[0].split(','), dtype=np.float64),
                        np.array(line.split(';')[1].split(','), dtype=np.float64)) for line in fp.readlines()]
            opt_method = self.opt_entry.get()
            batch_size = int(self.batch_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            self.model.fit(dataset, opt_method, batch_size, lr=lr, epochs=epochs)
        except ValueError as e:
            messagebox.showerror(title="Error", message=str(e))

    def init_handle(self):
        try:
            n_inputs = 2 * int(self.m_entry.get())
            n_classes = 5
            architecture = [int(layer) for layer in self.architecture_entry.get().split(',')]
            print(architecture)
            self.model = MLP(n_inputs, n_classes, architecture)
        except ValueError as e:
            messagebox.showerror(title="Error", message=str(e))
            return

    def predict_handle(self):
        if self.model is None:
            messagebox.showinfo(title="Model not initialized", message="Model must be initialized first")
        try:
            M = int(self.m_entry.get())
            x = np.array(ut.calculate_features(self.points, M))
            result = self.model.predict(x)
            output_decoder = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
            index = np.argmax(result)
            messagebox.showinfo(title="Prediction", message=output_decoder[index])
        except ValueError as e:
            messagebox.showerror(title="Error", message=str(e))
            return

    def save_features(self):
        filepath = self.path_entry.get()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            M = int(self.m_entry.get())
            data_class = self.selected_class.get()
            result = ut.calculate_features(self.points, M)
            with open(filepath, 'a') as fp:
                entry_str = ','.join([str(r) for r in result]) + ';' + \
                    ut.CLASS_MAP[data_class] + '\n'
                fp.write(entry_str)
        except ValueError as e:
            messagebox.showerror(title="Error", message=str(e))
            return

    def clear(self):
        self.points = list()
        self.drawing_canvas.delete("all")


def main():
    root = tk.Tk()
    my_gui = DatasetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
