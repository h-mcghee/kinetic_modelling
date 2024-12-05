import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from tkinter.filedialog import asksaveasfilename, askopenfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

import numpy as np


import lmfit
from lmfit import Parameters,Model
from tkinter import simpledialog
from fitting.fitting_functions import FittingFunctions
from gui.parameter_entry import ParameterEntryTable
# from functions import background_subtract
import matplotlib.ticker as ticker
import os

script_dir = os.path.dirname(__file__)
style_path = os.path.join(script_dir, '../styles/style.mplstyle')
plt.style.use(style_path)

class PlottingApp:
    def __init__(self, root):

        #instance variables
        self.x = None
        self.y = None
        self.result = None
        self.error = None
        self.log_var = tk.BooleanVar()
        self.norm_var = tk.BooleanVar()
        self.bkg_var = tk.BooleanVar()
        self.linthresh = tk.StringVar()
        self.bkg_lb = tk.StringVar()
        self.bkg_ub = tk.StringVar()
        self.truncate = tk.StringVar()
        self.fitting_functions = FittingFunctions()
        self.function_names = None
        self.root = root
        self.selected_model = None

        self.root.option_add('*Label.font', ('Arial', 12, 'bold'))
        self.root.option_add('*Button.font', ('Arial', 12, 'bold'))
        self.root.option_add('*Entry.font', ('Arial', 12, 'bold'))

        self.file_name = tk.StringVar()
        self.file_name.set('No file selected')

        #this could all be in a 'set_up' function
        self.root.title("Plotting App")
        
        

        #file frame

        file_frame = tk.Frame(self.root, relief = 'solid',bd = 2)
        file_frame.grid(row = 0, column = 0)

        self.open_button = tk.Button(file_frame, text="Open", command=self.open_file)
        self.open_button.grid(row = 0, column = 0)

        self.file_label = tk.Label(file_frame, textvariable = self.file_name)
        self.file_label.grid(row = 0,column = 1)

        #plot frame

        plot_frame = tk.Frame(self.root, relief = 'solid',bd = 2)
        plot_frame.grid(row = 1, column = 0)

        self.log_label = tk.Label(plot_frame, text = 'Log')
        self.log_label.grid(row = 0, column = 0)
        self.log_button = tk.Checkbutton(plot_frame, variable = self.log_var)
        self.log_button.grid(row = 1, column = 0)

        self.linthresh_label = tk.Label(plot_frame, text = 'Thresh:')
        self.linthresh_label.grid(row = 0, column = 1)
        self.linthresh = tk.Entry(plot_frame,width = 6)
        self.linthresh.insert(0, '1000')
        self.linthresh.grid(row = 1, column = 1)

        self.bkg_label = tk.Label(plot_frame, text = 'Bkg')
        self.bkg_label.grid(row = 0, column = 2)
        self.bkg_button = tk.Checkbutton(plot_frame,variable = self.bkg_var)
        self.bkg_button.grid(row = 1, column = 2)

        self.bkg_lb_label = tk.Label(plot_frame, text = 'lb')
        self.bkg_lb_label.grid(row = 0, column = 3)
        self.bkg_lb = tk.Entry(plot_frame,width = 6)
        self.bkg_lb.grid(row = 1, column = 3)

        self.bkg_ub_label = tk.Label(plot_frame, text = 'ub')
        self.bkg_ub_label.grid(row = 0, column = 4)
        self.bkg_ub = tk.Entry(plot_frame,width = 6)
        self.bkg_ub.grid(row = 1, column = 4)

        self.remove_label = tk.Label(plot_frame, text = 'Remove Indices')
        self.remove_label.grid(row = 0, column = 5)
        self.remove = tk.Entry(plot_frame,width = 6)
        self.remove.grid(row = 1, column = 5)

        self.norm_label = tk.Label(plot_frame, text = 'norm')
        self.norm_label.grid(row = 0, column = 6)
        self.norm_button = tk.Checkbutton(plot_frame, variable = self.norm_var)
        self.norm_button.grid(row = 1, column = 6)


        self.plot_button = tk.Button(plot_frame, text="Plot Data", command=self.plot_data)
        self.plot_button.grid(row = 0, column = 7,rowspan = 2)

        #fit frame
        fit_frame = tk.Frame(self.root, relief = 'solid',bd = 2)
        fit_frame.grid(row = 2, column = 0)

        self.model_options = self.get_function_names()
        self.model_options.append('multi')
        self.model_combobox = ttk.Combobox(fit_frame, values=self.model_options)
        self.model_combobox.grid(row = 0, column = 0)

        self.fit_button = tk.Button(fit_frame, text="Fit Data", command=self.select_model)
        self.fit_button.grid(row = 0, column = 1)

        self.clear_button = tk.Button(fit_frame, text="Clear Fit", command=self.clear_fit)
        self.clear_button.grid(row = 0,column = 2)

        self.result_label = tk.Label(root, text="")
        self.result_label.grid()

        self.save_button = tk.Button(root, text="Save Canvas", command=self.save_canvas)
        self.save_button.grid()

        self.save_txt_button = tk.Button(root, text="Save txt", command=self.save_txt)
        self.save_txt_button.grid()

        self.fig, self.ax = plt.subplots(figsize=(3,2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        # Destroy the FigureCanvasTkAgg instance to close the Matplotlib figure
        self.canvas.get_tk_widget().destroy()
        self.root.destroy()

    def open_file(self):
        #lets you browse for a file and then sets the label. 
        file_path = askopenfilename()
        data = np.genfromtxt(file_path)
        self.original_x = self.x = data[:, 0]
        self.original_y = self.y = data[:, 1] #/ np.max(data[:, 1])
        try: 
            self.original_error =  self.error = data[:, 2]
        except:
            self.original_error = self.error =  None
            print('no error')
        self.file_path = file_path
        self.file_name.set(os.path.basename(file_path))


    def plot_data(self):
        self.x = self.original_x
        self.y = self.original_y
        self.error = self.original_error
        try:
            self.ax.clear()
            colors = plt.cm.viridis(np.linspace(0,0.7,2))
            if self.remove.get():
                remove_indices = [int(i) for i in self.remove.get().split(',')]
                self.x = np.delete(self.x,remove_indices)
                self.y = np.delete(self.y,remove_indices)
                if self.error is not None:
                    self.error = np.delete(self.error,remove_indices)
            
            if self.bkg_var.get():
                lb = float(self.bkg_lb.get())
                ub = float(self.bkg_ub.get())
                self.y = background_subtract(self.x,self.y,lb,ub)
            
            if self.norm_var.get():
                norm = np.max(self.y)
                self.y = self.y/norm
                self.error = self.error/norm

            self.ax.plot(self.x, self.y,'o',markersize = 2,color = colors[0],label = 'exp')
            if self.error is not None:
                self.ax.fill_between(self.x, self.y - self.error, self.y + self.error, color = colors[0], alpha = 0.2,edgecolor = 'none')
       
            self.ax.set_xlabel('Delay / ps')
            self.ax.set_ylabel('Norm. intensity / a.u')
            log = self.log_var.get()
            if log:
                linthresh = float(self.linthresh.get())
                self.ax.set_xscale('symlog',linthresh = linthresh)
                self.ax.axvline(linthresh,color = 'k',linestyle = '--',alpha = 0.5)
                ax = plt.gca()
                linear_ticks = np.arange(np.min(self.x), linthresh, linthresh/10)  # Adjust the range and step to your needs
                log_ticks = np.arange(linthresh,np.max(self.x)+100,100)  # Adjust the range and step to your needs
                all_ticks = np.concatenate((linear_ticks, log_ticks))
                ax.xaxis.set_minor_locator(ticker.FixedLocator(all_ticks))
            self.fig.tight_layout()
            self.canvas.draw()
            

        except Exception as e:
            print(e)


    def get_function_names(self):
        return [func for func in dir(FittingFunctions) if callable(getattr(FittingFunctions, func)) and not func.startswith("__")]

#split this into select, and then fit. 
    def select_model(self):
        self.selected_model = self.model_combobox.get()

        if self.selected_model == 'multi':
            num_functions = simpledialog.askinteger("Input", "Enter the number of functions to be summed:")
            a = []

            for i in range(0,num_functions):
                function_name = simpledialog.askstring("Input", f"Enter the name of the function to be used for comp {i+1}:")
                selected_function = getattr(self.fitting_functions, function_name)
                a.append(Model(selected_function, prefix = f'comp{i+1}_'))

            composite_model = a[0]
            for model in a[1:]:
                composite_model += model

            model = composite_model

        else:

            selected_function = getattr(self.fitting_functions, self.selected_model)

            model = Model(selected_function)

        #checks that data has been assigned
        if self.x is not None and self.y is not None:
            # Get initial parameter values from the user
            initial_params = self.get_initial_params(model.param_names) #calls function to return dictionary of params

            try:
                self.result = model.fit(self.y, initial_params, x=self.x)
                self.update_result_label(self.result)
                self.ax.plot(self.x, self.result.best_fit, color = 'cornflowerblue',label = 'fit')
                self.ax.plot(self.x, self.y - self.result.best_fit, color = 'red',label = 'residuals')

                if self.selected_model == 'multi':
                    comps = self.result.eval_components(x=self.x)
                    for i in range(0,num_functions):
                        self.ax.plot(self.x,comps[f'comp{i+1}_'],linestyle = '--',label = f'comp{i+1}_')

                self.ax.legend()
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror('Error', f"Fit failed: {str(e)}")


    def get_initial_params(self, param_names):
        entry_table = ParameterEntryTable(self.root, param_names) #create an instance of ParameterEntryTable
        initial_params = Parameters()

        def callback(param_info):
            for info in param_info:
                name = info['name']
                value = info['value']
                min = info['min']
                max = info['max']
                vary = info['vary']
                initial_params.add(name,value,min = min, max = max,vary = vary)

        entry_table.callback = callback

        self.root.wait_window(entry_table)

        return initial_params

    def update_result_label(self, result):
        # Update the label with optimized parameter values compiles into a text file
        text = "Optimised Values:\n"
        for param_name, param in result.params.items():
            text += f"{param_name}: {param.value:.4f} ± {param.stderr:.4f}\n" 
        text+= f"Reduced Chi-squared: {result.redchi}"
        self.result_label.config(text=text)

    def clear_fit(self):
        # Clear the fit plot and update the label
        while len(self.ax.lines) > 1:
            self.ax.lines.pop()
        self.ax.legend([self.ax.lines[0]], [self.ax.lines[0].get_label()])
        self.result_label.config(text="Fit Cleared")

    def save_canvas(self):
        fpath = asksaveasfilename()

        if os.path.exists(fpath):
            result = tk.messagebox.askquestion("File exists", "The file already exists. Do you want to overwrite?", icon='warning')
            if result == 'no':
                return

        self.fig.savefig(fpath)

    def save_txt(self):
        fpath = asksaveasfilename()
        if os.path.exists(fpath):
            result = tk.messagebox.askquestion("File exists", "The file already exists. Do you want to overwrite?", icon='warning')
            if result == 'no':
                return
        if self.selected_model == 'multi':
            components = self.result.eval_components()
            component_arrays = [components[name] for name in components]
            if self.error is not None:
                np.savetxt(fpath, np.column_stack((self.x, self.y, self.error, self.result.best_fit, *component_arrays)))
            else:
                np.savetxt(fpath, np.column_stack((self.x, self.y, self.result.best_fit, *component_arrays)))

        else:
            if self.error is not None:
                np.savetxt(fpath, np.column_stack((self.x, self.y, self.error, self.result.best_fit)))
            else:
                np.savetxt(fpath, np.column_stack((self.x, self.y, self.result.best_fit)))
