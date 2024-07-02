import tkinter as tk
from tkinter import StringVar, Listbox, Scrollbar


class AutoCompleteApp:
    def __init__(self, parent, options):
        self.parent = parent
        self.options = options
        
        self.text_var = StringVar()
        self.textbox = tk.Entry(self.parent, textvariable=self.text_var, width=50)
        self.textbox.pack(pady=10)
        
        self.dropdown_frame = tk.Frame(self.parent)
        self.dropdown_frame.pack()
        
        self.listbox = Listbox(self.dropdown_frame, width=50)
        self.listbox.pack(side=tk.LEFT)
        
        self.scrollbar = Scrollbar(self.dropdown_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        
        self.text_var.trace("w", self.show_dropdown)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        
        self.populate_listbox()
        self.dropdown_frame.pack_forget()

    def show_dropdown(self, *args):
        if self.listbox.size() > 0:
            self.dropdown_frame.pack()
        else:
            self.dropdown_frame.pack_forget()
    
    def populate_listbox(self):
        self.listbox.delete(0, tk.END)
        for item in self.options:
            self.listbox.insert(tk.END, item)
    
    def on_select(self, event):
        selected = self.listbox.get(self.listbox.curselection())
        self.text_var.set(selected)
        self.dropdown_frame.pack_forget()
