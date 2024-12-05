# **Kinetic modelling app**

## **Description**

This is a Python-based plotting and fitting application build using Tkinter, Matplotlib, and LMfit. The application allows users to load, plot and fit 2D data to various kinetic models (including composite models). The results can be saved to text files. 

## Usage

### 1.	Launch the application

Run the main.py script

`python main.py`

### 2.	Load data

Click the **Open** button to select and load a .txt or .csv file containing the data.

The file must at least two *columns* of data (x, and y values) with an optional third column for errors (which will display as shaded error bars). You may need to manipulate your data file prior to loading to ensure the right format.

### 3.	Plot data

Click **Plot data** to display the data on a graph

You can adjust settings for:

* Log scale (optional, with a linear threshold - i.e. the plot is linear up until this point and on a logscale thereafter)
* Background subtraction ()input bounds **lb** and **ub** from which to subtract the mean signal)
* Remove indices (take out anomalous data points - currently you need to specify the exact index of that data point)
* Normalsation (normalises the data to maximum y-value)

### 4.	Fit Data

Select a model from the dropdown menu (e.g. Guassian, exp_decay...)

Optionally, fit a composite model by selecting **multi**. The dialog box will then guide you through selecting a composite of the available functions.

Enter your estimated parameters using the pop up box and click **Fit data**

The resulting fit, residual and optimised parameters will be displayed on the plot. 

### 5.	Save results

Save canvas as an image file or save data as a text file. The data is saved as a new file, containing the original data with the best-fit values appended as an additional column. 


## File Structure

```
kinetic_modelling/			#Root folder of project
├── src/				#Source code folder
│   └── main.py				#Main entry point of application
│   ├── fitting/			#Folder for fitting-related code
│   │   ├── __init__.py			#Initialisation file for the fitting module
│   │   ├── fitting_functions.py	#Contains class for functions for fitting models (can be expanded)
│   ├── gui/				#Folder for the GUI-related code
│   │   ├── __init__.py			#Initialisation for the GUI module
│   │   ├── functions.py		#Helper functions used within the GUI
│   │   ├── main_window.py		#Contains the main window code for the TKinter GUI
│   │   └── parameter_entry.py		#pop up window for parameter input during fitting
│   ├── styles/				#Folder for style configurations
│   │   └── style.mplstyle		#Matplotlib style file for the custom plot appearance
├── README.md				#Documentation
├── requirements.txt			#List of python dependencies required
├── TODO.md				#To-do list for future improvements



```
