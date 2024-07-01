import tkinter as tk

class ScrollableListApp:
    def __init__(self, root, frame):
        self.root = root
        
        # Create a frame for the scrollable list
        self.frame = frame

        # Create a canvas inside the frame
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT)

        # Add a scrollbar to the frame
        self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT)

        # Configure the canvas to use the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # Create a frame inside the canvas to hold the labels
        self.list_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.list_frame, anchor='nw')

        # Counter for labels
        self.label_count = 0

    def on_canvas_configure(self, event):
        # Update the scroll region to encompass the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def add_label(self, image_name):
        # Add a new label to the list frame
        self.label_count += 1
        new_label = tk.Label(self.list_frame, text=f"{image_name}")
        new_label.pack()

        # Update the scroll region to include the new label
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))