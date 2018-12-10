from tkinter import messagebox
from mlp import *
import util as ut
import tkinter as tk
import os

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700

class DatasetGUI():
    def __init__(self, master):
        self.master = master
        master.title("Drawing GUI")

        self.points = list()
        self.model = None

        self.label = tk.Label(master, text="Press left-click and hold to draw")
        self.label.pack()

        self.menu_frame = tk.Frame(master, width=CANVAS_WIDTH + 5, height=10)
        self.menu_frame.pack(side=tk.BOTTOM)

        self.clear_button = tk.Button(self.menu_frame, text="Clear", command=self.clear)
        self.clear_button.grid(row=0, column=0)

        self.close_button = tk.Button(self.menu_frame, text="Close", command=master.quit)
        self.close_button.grid(row=0, column=1, padx=10)

        self.save_button = tk.Button(self.menu_frame, text="Save", command=self.save_features)
        self.save_button.grid(row=0, column=3, padx=10)

        self.predict_button = tk.Button(self.menu_frame, text="Predict", command=self.predict_handle)
        self.predict_button.grid(row=2, column=4)

        self.train_button = tk.Button(self.menu_frame, text="Train", command=self.train_handle)
        self.train_button.grid(row=2, column=3)

        self.init_button = tk.Button(self.menu_frame, text="Init", command=self.init_handle)
        self.init_button.grid(row=2, column=2)

        self.save_location = tk.Entry(self.menu_frame)
        self.save_location.bind("<FocusIn>", lambda event: self.save_location.delete(0, 'end'))
        self.save_location.insert(0, "enter file location...")
        self.save_location.grid(row=0, column=2)

        self.m_entry = tk.Entry(self.menu_frame)
        self.m_entry.bind("<FocusIn>", lambda event: self.m_entry.delete(0, 'end'))
        self.m_entry.insert(0, 'M')
        self.m_entry.grid(row=0, column=4)

        self.class_entry = tk.Entry(self.menu_frame)
        self.class_entry.bind("<FocusIn>", lambda event: self.class_entry.delete(0, 'end'))
        self.class_entry.insert(0, 'class (1-5)')
        self.class_entry.grid(row=0, column=5)

        self.opt_entry = tk.Entry(self.menu_frame)
        self.opt_entry.bind("<FocusIn>", lambda event: self.opt_entry.delete(0, 'end'))
        self.opt_entry.insert(0, "enter optimization method")
        self.opt_entry.grid(row=1, column=1)

        self.epochs_entry = tk.Entry(self.menu_frame)
        self.epochs_entry.bind("<FocusIn>", lambda event: self.epochs_entry.delete(0, 'end'))
        self.epochs_entry.insert(0, "enter number of epochs")
        self.epochs_entry.grid(row=1, column=2)

        self.b_size_entry = tk.Entry(self.menu_frame)
        self.b_size_entry.bind("<FocusIn>", lambda event: self.b_size_entry.delete(0, 'end'))
        self.b_size_entry.insert(0, "enter batch size")
        self.b_size_entry.grid(row=1, column=3)

        self.lr_entry = tk.Entry(self.menu_frame)
        self.lr_entry.bind("<FocusIn>", lambda event: self.lr_entry.delete(0, 'end'))
        self.lr_entry.insert(0, "enter learning rate")
        self.lr_entry.grid(row=1, column=4)

        self.architecture_entry = tk.Entry(self.menu_frame)
        self.architecture_entry.bind("<FocusIn>", lambda event: self.architecture_entry.delete(0, 'end'))
        self.architecture_entry.insert(0, "enter architecture (ints separated by comma)")
        self.architecture_entry.grid(row=1, column=5)

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
            dataset_path = self.save_location.get()
            with open(dataset_path) as fp:
                dataset = [(np.array(line.split(';')[0].split(','), dtype=np.float64),
                        np.array(line.split(';')[1].split(','), dtype=np.float64)) for line in fp.readlines()]
            opt_method = self.opt_entry.get()
            batch_size = int(self.b_size_entry.get())
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
        filepath = self.save_location.get()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            M = int(self.m_entry.get())
            data_class = int(self.class_entry.get())
            if data_class < 1 or data_class > 5:
                raise ValueError
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
