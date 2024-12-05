import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import Parameters,Model
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel
from tkinter import simpledialog
from tkinter.filedialog import asksaveasfilename, askopenfilename
from fitting.fitting_functions import FittingFunctions
import matplotlib.ticker as ticker
import os

class ParameterEntryTable(tk.Toplevel): #inherits from tk.Toplevel
    def __init__(self, parent, param_names):
        super().__init__(parent)
        self.param_names = param_names
        self.entry_vars = {name: {'value': tk.StringVar(value = '1'), 'min': tk.StringVar(value = '-inf'), 'max': tk.StringVar(value = 'inf'), 'vary':tk.BooleanVar(value = True)} for name in param_names}
        self.title("Enter initial values")
        for i, name in enumerate(param_names):
            tk.Label(self, text = f'{name}:').grid(row=i+1, column=0)
            tk.Entry(self, textvariable = self.entry_vars[name]['value'],width = 5).grid(row=i+1, column=1)
            tk.Entry(self, textvariable = self.entry_vars[name]['min'],width = 5).grid(row=i+1, column=2)
            tk.Entry(self, textvariable = self.entry_vars[name]['max'],width = 5).grid(row=i+1, column=3)
            tk.Checkbutton(self, text = 'Vary', variable = self.entry_vars[name]['vary']).grid(row = i+1, column = 4)
        tk.Button(self, text = 'OK', command = self.confirm).grid(row=len(param_names)+1, column=0, columnspan=5)
        tk.Label(self, text = 'value').grid(row = 0, column = 1)
        tk.Label(self, text = 'min').grid(row = 0, column = 2)
        tk.Label(self, text = 'max').grid(row = 0, column = 3)

    def confirm(self):
        # Get parameter values and 'vary' status from entry widgets
        param_info = [{'name': name,
                       'value': float(self.entry_vars[name]['value'].get()),
                          'min': float(self.entry_vars[name]['min'].get()),
                            'max': float(self.entry_vars[name]['max'].get()),
                       'vary': bool(self.entry_vars[name]['vary'].get())}
                      for name in self.param_names]
        
        self.destroy()  # Close the parameter entry window
        self.callback(param_info)