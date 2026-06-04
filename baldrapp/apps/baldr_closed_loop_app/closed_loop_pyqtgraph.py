#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import sys
import importlib.resources as resources
import os 
import pickle
#import aotools - removed dependancy: use common/phasescreens.py instead
from PyQt5 import QtWidgets, QtCore,  QtGui
import pyqtgraph as pg
import traceback


## Testing pip installation locally (for developers)
# rm -rf dist build *.egg-info
# pip install --upgrade pip setuptools wheel twine
# python setup.py sdist bdist_wheel
# pip install dist/BaldrApp-0.1.2-py3-none-any.whl  --force-reinstall
# pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e
# closed_loop_pyqtgraph.py <--- check it runs
# 
# after or between rebuilds rm -r build  , rm -r dist 

# once happy with the build to upload to pypi (needs to be new version each upload)
#twine upload dist/*. tokens in $HOME/.pypirc

# clean buiold and uploads that avoids "license-file error"
#rm -rf dist build *.egg-info
#rm -rf temp_build_env      
#python -m venv temp_build_env    
#source temp_build_env/bin/activate    
#pip install --upgrade pip setuptools wheel twine
#python setup.py sdist bdist_wheel
#twine upload dist/*
#https://pypi.org/project/BaldrApp/0.1.3/

def add_project_root_to_sys_path(project_root_name="BaldrApp"):
    """
    Adds the project root directory to sys.path to allow importing from shared modules.
    
    Args:
        project_root_name (str): The name of the project root directory.
    """
    try:
        # Attempt to use __file__ to get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback to current working directory (useful in interactive environments)
        current_dir = os.getcwd()

    # Traverse up the directory tree to find the project root
    project_root = current_dir
    while True:
        if os.path.basename(project_root) == project_root_name:
            break
        new_project_root = os.path.dirname(project_root)
        if new_project_root == project_root:
            # Reached the filesystem root without finding the project root
            project_root = None
            break
        project_root = new_project_root

    if project_root and project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added '{project_root}' to sys.path")
    elif project_root is None:
        print(f"Error: '{project_root_name}' directory not found in the directory hierarchy.")
    else:
        print(f"'{project_root}' is already in sys.path")

# Call the function to add the project root
add_project_root_to_sys_path()

#from baldrapp.common import *  # Forward all imports
# ^this is necessary if using Strehlmodel or something from common submodules
from baldrapp.common import baldr_core as bldr
#from baldrapp.common.baldr_core import PixelWiseStrehlModel
from baldrapp.common import DM_basis
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens
from baldrapp.common import DM_registration
#from baldrapp.configurations import BALDR_UT_J3

# Create a class to redirect stdout to the QTextEdit widget
class OutputRedirector:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        """Write the output to the QTextEdit widget."""
        self.text_edit.append(text)

    def flush(self):
        """Required for file-like object interface, can be left empty."""
        pass

class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Main layout (QGridLayout) for the entire window
        self.layout = QtWidgets.QGridLayout()

        # Initialize command history and index
        self.command_history = []  # List to store the history of commands
        self.history_index = -1    # Index to track the current position in the history

        # Buttons
        self.run_button = QtWidgets.QPushButton("Run")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.zero_gains_button = QtWidgets.QPushButton("Set Gains to Zero")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.save_button = QtWidgets.QPushButton("Save Telemetry")
    
        # Buttons for saving and loading the zwfs_ns state
        self.save_zwfs_ns_button = QtWidgets.QPushButton("Save config.")
        self.load_zwfs_ns_button = QtWidgets.QPushButton("Load config.")

        # Make buttons smaller
        button_size = QtCore.QSize(120, 30)  # Set the button size
        self.run_button.setFixedSize(button_size)
        self.pause_button.setFixedSize(button_size)
        self.zero_gains_button.setFixedSize(button_size)
        self.reset_button.setFixedSize(button_size)
        self.save_button.setFixedSize(button_size)
        self.save_zwfs_ns_button.setFixedSize(button_size)
        self.load_zwfs_ns_button.setFixedSize(button_size)

        # Connect buttons to functions
        self.run_button.clicked.connect(self.run_loop)
        self.pause_button.clicked.connect(self.pause_loop)
        self.zero_gains_button.clicked.connect(self.set_gains_to_zero)
        self.reset_button.clicked.connect(self.reset)
        self.save_button.clicked.connect(self.save_telemetry)  # Placeholder function
        self.save_zwfs_ns_button.clicked.connect(self.save_zwfs_ns_to_json)
        self.load_zwfs_ns_button.clicked.connect(self.load_zwfs_ns_from_json) 

        # Visual "LED" for running state
        self.status_led = QtWidgets.QLabel()
        self.status_led.setFixedSize(20, 20)
        self.update_led(False)  # Initialize as "not running" (False)

        # Terminal-like prompt setup (left side)
        self.prompt_history = QtWidgets.QTextEdit()
        self.prompt_history.setReadOnly(True)  # History is read-only
        self.prompt_input = QtWidgets.QLineEdit()
        self.prompt_input.returnPressed.connect(self.handle_command)

        # Redirect print() output to QTextEdit
        sys.stdout = OutputRedirector(self.prompt_history)  # Redirect stdout

        # PyQtGraph plots (top half)
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Adding the plots above
        # image_titles = ["DM Disturbance", "Input Phase", "Intensity", "Reconstructed Command"]
        # image_x_labels = ["X actuator", "X-Axis", "Pixels", "X actuator"]
        # image_y_labels = ["Y actuator", "Y-Axis", "Pixels", "Y actuator"]

        # self.image_plots = []
        # for i in range(4):
        #     img_view = self.plot_widget.addPlot(row=0, col=i)
        #     img_view.setTitle(image_titles[i], color='b', size="12pt")
        #     img_view.setLabel('left', image_y_labels[i], color='r', size="10pt")
        #     img_view.setLabel('bottom', image_x_labels[i], color='g', size="10pt")

        #     img_item = pg.ImageItem()
        #     img_view.addItem(img_item)
        #     self.image_plots.append(img_item)

        # self.image_plots = []
        # for i in range(4):
        #     img_view = pg.ViewBox()  # Create a ViewBox instead of addPlot
        #     self.plot_widget.addItem(img_view, row=0, col=i)  # Add ViewBox to the layout
            
        #     # Create a LabelItem for the titles and add to the layout
        #     title = pg.LabelItem(image_titles[i], size="12pt", color='b')
        #     self.plot_widget.addItem(title, row=1, col=i)
            
        #     # Set labels using LabelItems or other widget elements, since ViewBox does not have labels.
        #     img_item = pg.ImageItem()
        #     img_view.addItem(img_item)  # Add the ImageItem to the ViewBox
        #     self.image_plots.append(img_item)
            
        # Adding the buttons in two columns at the bottom right
        button_layout = QtWidgets.QGridLayout()
        button_layout.addWidget(self.run_button, 0, 0)
        button_layout.addWidget(self.pause_button, 0, 1)
        button_layout.addWidget(self.zero_gains_button, 1, 0)
        button_layout.addWidget(self.reset_button, 1, 1)
        button_layout.addWidget(self.save_button, 2, 0, 1, 2)  # Save Telemetry button in 1 column span
        # Adding these buttons to the button layout (assuming it's on the right side)
        button_layout.addWidget(self.save_zwfs_ns_button, 3, 0, 1, 2)  # Save button in the next row
        button_layout.addWidget(self.load_zwfs_ns_button, 4, 0, 1, 2)  # Load button in the next row

        # Command prompt (bottom left)
        command_layout = QtWidgets.QVBoxLayout()
        command_layout.addWidget(self.prompt_history)
        command_layout.addWidget(self.prompt_input)

        # Add widgets to the main layout
        self.layout.addWidget(self.plot_widget, 0, 0, 1, 2)  # Plots at the top
        self.layout.addLayout(command_layout, 1, 0)  # Command prompt on the bottom left
        self.layout.addLayout(button_layout, 1, 1)  # Buttons on the bottom right

        # Add the LED next to the buttons
        self.layout.addWidget(self.status_led, 2, 1, QtCore.Qt.AlignRight)

        # Set stretch to ensure the top takes most space
        self.layout.setRowStretch(0, 4)  # The plot area (row 0) takes up more space
        self.layout.setRowStretch(1, 1)  # The bottom row takes less space
        self.layout.setColumnStretch(0, 3)  # The command prompt area gets more space
        self.layout.setColumnStretch(1, 1)  # The button area gets less space

        # Set the layout
        self.setLayout(self.layout)

        # Initialize a completer (UPDATED SECTION)
        self.completer = QtWidgets.QCompleter(self)  # Create the completer
        self.prompt_input.setCompleter(self.completer)  # Set it to work with the input field
        self.update_completer()  # Initialize completer suggestions


        # Connect the returnPressed signal to the command handler
        self.prompt_input.returnPressed.connect(self.handle_command)

        # Timer for running the AO_iteration in a loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run_AO_iteration)
        self.loop_running = False


        # Initialize lists for storing PlotDataItems for e_HO_list and e_TT_list
        self.e_HO_items = []  # List to store PlotDataItems for HO error plots
        self.e_TT_items = []  # List to store PlotDataItems for TT error plots
        self.line_plots = []  # List to store PlotItem for the actual plots

        # Create Image and Line plots in PyQtGraph
        image_titles = ["DM Disturbance", "Input Phase", "Intensity", "Reconstructed Command"]
        #image_x_labels = ["X actuator", "X-Axis", "Pixels", "X actuator"]
        #image_y_labels = ["Y actuator", "Y-Axis", "Pixels", "Y actuator"]


        self.image_plots = []
        for i in range(4):
            img_view = self.plot_widget.addPlot(row=0, col=i)
            
            # Set title and labels for each image plot
            img_view.setTitle(image_titles[i], color='b', size="12pt")  # Set the title
            #img_view.setLabel('left', image_y_labels[i], color='r', size="10pt")  # Set Y-axis label
            #img_view.setLabel('bottom', image_x_labels[i], color='g', size="10pt")  # Set X-axis label
            #title = self.plot_widget.addLabel(image_titles[i], row=0, col=i)  # Title on row 0, image on row 1
    
            img_item = pg.ImageItem()
            img_view.addItem(img_item)
            self.image_plots.append(img_item)

        plot_titles = ["HO Error Plot", "TT Error Plot", "Strehl Ratio", "RMSE"]  # Titles for each plot
        plot_y_labels = ["HO Error", "TT Error", "Strehl", "RMSE"]  # Y-axis labels
        plot_x_labels = ["Iterations", "Iterations", "Iterations", "Iterations"]  # X-axis labels
                
        for i in range(4):
            line_plot = self.plot_widget.addPlot(row=1 + (i // 2), col=(i % 2) * 2, colspan=2)
            self.line_plots.append(line_plot)  # Store the PlotItem, not the result of plot()
            line_plot.setTitle(plot_titles[i], color='b', size="12pt")
            line_plot.setLabel('left', plot_y_labels[i], units=None, color='r', size="10pt")
            line_plot.setLabel('bottom', plot_x_labels[i], units=None, color='g', size="10pt")

        
        # Add the layout to the widget
        self.setLayout(self.layout)

        # Timer for running the AO_iteration in a loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run_AO_iteration)
        self.loop_running = False


    def update_completer(self):
        """Update the completer suggestions based on current global variables."""
        # Get the list of global variable names
        global_vars = list(globals().keys())

        # Create a QCompleter model with global variable names
        model = QtGui.QStandardItemModel(self.completer)
        for var in global_vars:
            item = QtGui.QStandardItem(var)
            model.appendRow(item)

        # Set the model in the completer
        self.completer.setModel(model)
        

    def run_loop(self):
        if not self.loop_running:
            self.loop_running = True
            self.update_led(True) 
            self.timer.start(100)  # Adjust time in ms if needed

    def pause_loop(self):
        self.loop_running = False
        self.update_led(False) 
        self.timer.stop()

    def set_gains_to_zero(self):
        self.pause_loop()
        try:
            zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
        except:
            print('no zwfs_ns.ctrl.TT_ctrl found so cannot change gains')

        try:    
            zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()
        except:
            print('no zwfs_ns.ctrl.HO_ctrl found so cannot change gains')
    
        self.run_loop()

    def reset(self ):
        self.pause_loop()
        _ = bldr.reset_telemetry( zwfs_ns )
        try:
            zwfs_ns.ctrl.TT_ctrl.reset()
        except:
            print('no zwfs_ns.ctrl.TT_ctrl found so cannot reset')

        try:
            zwfs_ns.ctrl.HO_ctrl.reset()
        except:
            print('no zwfs_ns.ctrl.HO_ctrl found so cannot reset')

        for plot_item in self.line_plots:
            plot_item.clear()  # This removes all lines from the plot
    
        # Clear the stored PlotDataItems (HO and TT errors)
        self.e_HO_items.clear()
        self.e_TT_items.clear()
    
 
    def update_led(self, running):
        """Update the LED color depending on whether the system is running."""
        if running:
            self.status_led.setStyleSheet("background-color: green; border-radius: 10px;")
        else:
            self.status_led.setStyleSheet("background-color: red; border-radius: 10px;")

    def save_telemetry(self):
        # save telelmetry from the simulation 
        
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save zwfs_ns.telem as fits", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        if file_name:
            
            bldr.save_telemetry( zwfs_ns.telem , savename = file_name )


    def save_zwfs_ns_to_json(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save zwfs_ns as JSON", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        if file_name:
            try:
                serialized_data = serialize_object(zwfs_ns)
                with open(file_name, 'w') as json_file:
                    json.dump(serialized_data, json_file, indent=4)
                print(f"Saved zwfs_ns to {file_name}")
            except Exception as e:
                print(f"Error saving zwfs_ns: {str(e)}")

    # def load_zwfs_ns_from_json(self):
    #     options = QtWidgets.QFileDialog.Options()
    #     options |= QtWidgets.QFileDialog.DontUseNativeDialog
    #     file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load zwfs_ns from JSON", "", "JSON Files (*.json);;All Files (*)", options=options)
        
    #     if file_name:
    #         try:
    #             with open(file_name, 'r') as json_file:
    #                 loaded_data = json.load(json_file)
                
    #             global zwfs_ns
    #             zwfs_ns = deserialize_object(loaded_data)
    #             print(f"Loaded zwfs_ns from {file_name}")
    #         except Exception as e:
    #             print(f"Error loading zwfs_ns: {str(e)}")




    def load_zwfs_ns_from_json(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load zwfs_ns from JSON", "", "JSON Files (*.json);;All Files (*)", options=options)

        if file_name:
            try:
                with open(file_name, 'r') as json_file:
                    loaded_data = json.load(json_file)

                global zwfs_ns
                zwfs_ns = deserialize_object(loaded_data)  # Load data into zwfs_ns
                
                # reset the telemetry
                zwfs_ns = bldr.reset_telemetry( zwfs_ns )
                
                # Reinitialize zwfs_ns.ctrl.TT_ctrl and zwfs_ns.ctrl.HO_ctrl based on their ctrl_type
                if hasattr(zwfs_ns, 'ctrl'):

                    if hasattr(zwfs_ns.ctrl, 'TT_ctrl'):
                        if zwfs_ns.ctrl.TT_ctrl.ctrl_type == 'PID':
                            zwfs_ns.ctrl.TT_ctrl = bldr.PIDController(
                                kp=zwfs_ns.ctrl.TT_ctrl.kp,
                                ki=zwfs_ns.ctrl.TT_ctrl.ki,
                                kd=zwfs_ns.ctrl.TT_ctrl.kd,
                                upper_limit=zwfs_ns.ctrl.TT_ctrl.upper_limit,
                                lower_limit=zwfs_ns.ctrl.TT_ctrl.lower_limit,
                                setpoint=zwfs_ns.ctrl.TT_ctrl.setpoint
                            )
                        elif zwfs_ns.ctrl.TT_ctrl.ctrl_type == 'Leaky':
                            zwfs_ns.ctrl.TT_ctrl = bldr.LeakyIntegrator(
                                ki=zwfs_ns.ctrl.TT_ctrl.ki,
                                lower_limit=zwfs_ns.ctrl.TT_ctrl.lower_limit,
                                upper_limit=zwfs_ns.ctrl.TT_ctrl.upper_limit,
                                kp=zwfs_ns.ctrl.TT_ctrl.kp
                            )

                    if hasattr(zwfs_ns.ctrl, 'HO_ctrl'):
                        if zwfs_ns.ctrl.HO_ctrl.ctrl_type == 'PID':
                            zwfs_ns.ctrl.HO_ctrl = bldr.PIDController(
                                kp=zwfs_ns.ctrl.HO_ctrl.kp,
                                ki=zwfs_ns.ctrl.HO_ctrl.ki,
                                kd=zwfs_ns.ctrl.HO_ctrl.kd,
                                upper_limit=zwfs_ns.ctrl.HO_ctrl.upper_limit,
                                lower_limit=zwfs_ns.ctrl.HO_ctrl.lower_limit,
                                setpoint=zwfs_ns.ctrl.HO_ctrl.setpoint
                            )
                        elif zwfs_ns.ctrl.HO_ctrl.ctrl_type == 'Leaky':
                            zwfs_ns.ctrl.HO_ctrl = bldr.LeakyIntegrator(
                                ki=zwfs_ns.ctrl.HO_ctrl.ki,
                                lower_limit=zwfs_ns.ctrl.HO_ctrl.lower_limit,
                                upper_limit=zwfs_ns.ctrl.HO_ctrl.upper_limit,
                                kp=zwfs_ns.ctrl.HO_ctrl.kp
                            )

                print(f"Loaded zwfs_ns from {file_name}")
            except Exception as e:
                print(f"Error loading zwfs_ns: {str(e)}")



    def handle_command(self):
        # Get the user input command
        command = self.prompt_input.text()

        # Add the command to history and the command list
        self.command_history.append(command)
        self.history_index = len(self.command_history)

        # Display the command in the prompt history
        self.prompt_history.append(f"> {command}")

        try:
            # Execute the command in the same environment using exec()
            # Use `globals()` to allow the command to access and modify variables
            exec(command, globals() ) #, locals())
            self.prompt_history.append("Command executed.")
            
            #if 'dm_disturbance' in command:
            #    self.prompt_history.append("dm_disturbance updated.")
                
                
        except Exception as e:
            # Capture and display errors in the prompt history
            error_msg = traceback.format_exc()  # Get detailed error message
            self.prompt_history.append(f"Error: {str(e)}\n{error_msg}")

        self.update_completer()
        # Clear the input field
        self.prompt_input.clear()



    def keyPressEvent(self, event):
        """Handle arrow up/down to navigate through the command history."""
        if event.key() == QtCore.Qt.Key_Up:
            # Go up in the history
            if self.history_index > 0:
                self.history_index -= 1
                self.prompt_input.setText(self.command_history[self.history_index])
        elif event.key() == QtCore.Qt.Key_Down:
            # Go down in the history
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.prompt_input.setText(self.command_history[self.history_index])
            else:
                self.history_index = len(self.command_history)
                self.prompt_input.clear()



    # def check_input(self):
        
    #     user_input = self.text_input.text()
    #     # self.pause_loop()
    #     # Placeholder: Add conditions for user input processing
    #     print(f"User input: {user_input}")  # Replace with actual processing logic
        
    #     if 'kpHO*=' in user_input:
    #         factor = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.HO_ctrl.kp *= factor
    #     if 'kpTT*=' in user_input:
    #         factor = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.TT_ctrl.kp *= factor
    #     if 'kiHO*=' in user_input:
    #         factor = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.HO_ctrl.ki *= factor
    #     if 'kiTT*=' in user_input:
    #         factor = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.TT_ctrl.ki *= factor

        
    #     if 'kpHO[' in user_input:
    #         index = int( user_input.split('[')[1].split(']')[0] )
    #         value = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.HO_ctrl.kp[index] = value
    #     if 'kpTT[' in user_input:
    #         index = int( user_input.split('[')[1].split(']')[0] )
    #         value = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.TT_ctrl.kp[index] = value
    #     if 'kiHO[' in user_input:
    #         index = int( user_input.split('[')[1].split(']')[0] )
    #         value = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.HO_ctrl.ki[index] = value
    #     if 'kiTT[' in user_input:
    #         index = int( user_input.split('[')[1].split(']')[0] )
    #         value = float( user_input.split('=')[-1] )
    #         zwfs_ns.ctrl.TT_ctrl.ki[index] = value
           
        #self.run_loop()

    def run_AO_iteration(self):
        
        if dynamic_opd_input:
            atm_opd_input = update_opd_input( scrn, scaling_factor=phase_scaling_factor )
        else:
            atm_opd_input = 0 * zwfs_ns.grid.pupil_mask

        opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
            sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )

        opd_input = np.sum( [  atm_opd_input, opd_current_dm ] , axis=0 )
        
        # Call the AO iteration function from your module
        if method=='zonal_interp_no_projection':
            kwargs = {"N0_dm":N0_dm, "HO_ctrl": zwfs_ns.ctrl.HO_ctrl } #zonal_ctrl_dict['HO_ctrl']  } 
            _ = bldr.AO_iteration( opd_input=opd_input, amp_input=amp_input, opd_internal=0*opd_internal, zwfs_ns=zwfs_ns, dm_disturbance = dm_disturbance ,\
                record_telemetry=True, method='zonal_interp_no_projection', detector=zwfs_ns.detector, obs_intermediate_field=True, \
                use_pyZelda = use_pyZelda, include_shotnoise=True, **kwargs)
            
        elif method=='MVM-TT-HO':
            kwargs = {"I0":zwfs_ns.reco.I0, "HO_ctrl": zwfs_ns.ctrl.HO_ctrl, "TT_ctrl": zwfs_ns.ctrl.TT_ctrl }
            _ = bldr.AO_iteration( opd_input, amp_input, opd_internal,  zwfs_ns, dm_disturbance=dm_disturbance, \
                record_telemetry=True , method = 'MVM-TT-HO', detector=zwfs_ns.detector, use_pyZelda=use_pyZelda, **kwargs )
        else:
            raise UserWarning('no valid method input. try "MVM-TT-HO"')
        # Retrieve telemetry data
        im_dm_dist = util.get_DM_command_in_2D(zwfs_ns.telem.dm_disturb_list[-1])
        im_phase = zwfs_ns.telem.field_phase[-1]
        im_int = zwfs_ns.telem.i_list[-1]/np.mean(zwfs_ns.telem.i_list[-1])   - zwfs_ns.reco.I0/np.mean( zwfs_ns.reco.I0 ) #zwfs_ns.telem.i_list[-1]
        im_cmd = util.get_DM_command_in_2D(np.array(zwfs_ns.telem.c_TT_list[-1]) + np.array(zwfs_ns.telem.c_HO_list[-1]))


        # Update images in the PyQtGraph interface
        self.image_plots[0].setImage(im_dm_dist)
        self.image_plots[1].setImage(im_phase)
        self.image_plots[2].setImage(im_int)
        self.image_plots[3].setImage(im_cmd)

        # Update line plots
        # Check if line data exists
        """if len(zwfs_ns.telem.e_HO_list) > 0:
            #self.line_plots[0].clear()
            self.line_plots[0].getViewBox().clear()
            for row in np.array( zwfs_ns.telem.e_HO_list ).T:  # Assuming each item is a row (1D array)
                #self.line_plots[0].setData(row) # Plot each row in blue
                self.line_plots[0].plot(np.arange(len(row)), row, pen='b')

        if len(zwfs_ns.telem.e_TT_list) > 0:
            #self.line_plots[1].clear()
            self.line_plots[1].getViewBox().clear()
            for row in np.array( zwfs_ns.telem.e_TT_list ).T:
                #self.line_plots[1].setData(row) # Plot each row in red
                self.line_plots[1].plot(np.arange(len(row)), row, pen='b')
        """        



        # Plot e_HO_list
        if len(np.array(zwfs_ns.telem.e_HO_list).T) > 0:
            for i, row in enumerate(np.array(zwfs_ns.telem.e_HO_list).T):
                if i < len(self.e_HO_items):
                    # Update existing plot data
                    self.e_HO_items[i].setData(np.arange(len(row)), row)
                else:
                    # Create new PlotDataItem and store it
                    plot_item = self.line_plots[0].plot(np.arange(len(row)), row, pen='b')  # Create the line
                    self.e_HO_items.append(plot_item)  # Store the PlotDataItem

        # Plot e_TT_list
        if len(np.array(zwfs_ns.telem.e_TT_list).T ) > 0:
            for i, row in enumerate(np.array(zwfs_ns.telem.e_TT_list).T):
                if i < len(self.e_TT_items):
                    # Update existing plot data
                    self.e_TT_items[i].setData(np.arange(len(row)), row)
                else:
                    # Create new PlotDataItem and store it
                    plot_item = self.line_plots[1].plot(np.arange(len(row)), row, pen='r')  # Create the line
                    self.e_TT_items.append(plot_item)  # Store the PlotDataItem


        # Update 1D telemetry data (strehl and rmse)
        if len(zwfs_ns.telem.strehl) > 0:
            self.line_plots[2].plot(np.arange(len(zwfs_ns.telem.strehl)), zwfs_ns.telem.strehl)

        if len(zwfs_ns.telem.rmse_list) > 0:
            self.line_plots[3].plot(np.arange(len(zwfs_ns.telem.rmse_list)), zwfs_ns.telem.rmse_list)
            
        # if len(zwfs_ns.telem.strehl) > 0:
        #     self.line_plots[2].setData(zwfs_ns.telem.strehl)
        #     #self.line_plots[2].getViewBox().autoRange()

        # if len(zwfs_ns.telem.rmse_list) > 0:
        #     self.line_plots[3].setData(zwfs_ns.telem.rmse_list)
        #     #self.line_plots[3].getViewBox().autoRange()



def update_opd_input(scrn, scaling_factor = 0.1):
    # update rolling phase screen if dynamic_opd_input 
    scrn.add_row()
    opd_input = scaling_factor * zwfs_ns.optics.wvl0 / (2*np.pi) * zwfs_ns.grid.pupil_mask * scrn.scrn
    return opd_input 
    
def serialize_object(obj, path="root"):
    """
    Recursively serializes an object, handling SimpleNamespace, custom objects, and
    ensuring basic Python types (int, float, str, etc.) and NumPy arrays are serialized properly.
    The 'path' argument helps in identifying the problematic object during debugging.
    """
    if isinstance(obj, SimpleNamespace):
        return {key: serialize_object(value, path=f"{path}.{key}") for key, value in vars(obj).items()}
    elif isinstance(obj, dict):
        # Handle invalid keys in the dictionary
        serialized_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple keys to strings
                key = str(key)
            if not isinstance(key, (str, int, float, bool, type(None))):
                print(f"Invalid key type {type(key)} at path: {path} -> Key: {key}")
                raise TypeError(f"Invalid key type: {type(key)}. Key: {key}, Path: {path}")
            serialized_dict[key] = serialize_object(value, path=f"{path}.{key}")
        return serialized_dict
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (list, tuple)):
        return [serialize_object(item, path=f"{path}[{idx}]") for idx, item in enumerate(obj)]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj  # Primitive types are directly serializable
    elif hasattr(obj, '__dict__'):  # Custom object case, serialize its attributes
        return {key: serialize_object(value, path=f"{path}.{key}") for key, value in vars(obj).items()}
    else:
        return str(obj)  # Convert non-serializable objects to strings

def namespace_to_dict(namespace, exclude_keys=None):
    """
    Recursively converts a SimpleNamespace (and nested custom objects) to a dictionary.
    Excludes specific keys during the conversion.
    
    Parameters:
    - namespace: The SimpleNamespace to convert.
    - exclude_keys: List of keys (strings) to exclude from the conversion.
    """
    if exclude_keys is None:
        exclude_keys = []
    
    output_dict = {}
    for key, value in vars(namespace).items():
        if key not in exclude_keys:
            output_dict[key] = serialize_object(value, path=key)
    
    return output_dict

def save_namespace_as_json(namespace, filename, exclude_keys=None):
    """
    Converts a SimpleNamespace to a dictionary and saves it as a JSON file.
    Excludes specified keys from being serialized.
    
    Parameters:
    - namespace: The SimpleNamespace to convert.
    - filename: The name of the output JSON file.
    - exclude_keys: List of keys (strings) to exclude from the conversion.
    """
    dict_representation = namespace_to_dict(namespace, exclude_keys)
    
    with open(filename, 'w') as json_file:
        json.dump(dict_representation, json_file, indent=4)

def deserialize_object(obj):
    """
    Recursively deserializes a JSON object back into the appropriate types, such as 
    SimpleNamespace, np.ndarray, and primitive types. Converts lists back to np.array
    where applicable.
    """
    # If the object is a dictionary, check for special cases
    if isinstance(obj, dict):
        # Check for any special markers like '__tuple__' or others
        if "__tuple__" in obj:
            return tuple(obj["__tuple__"])  # Convert back to a tuple
        else:
            # Convert the dict back to a SimpleNamespace or process nested dicts
            deserialized_dict = {key: deserialize_object(value) for key, value in obj.items()}
            return SimpleNamespace(**deserialized_dict)

    # If the object is a list, we assume it was originally an np.array
    elif isinstance(obj, list):
        return np.array([deserialize_object(item) for item in obj])

    # Handle base cases: primitive types
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    # If none of the above cases match, return the object as is (or raise an error)
    return obj

# Function to load the model from a pickle file
def load_model_from_pickle(filename):
    """

    #BUG HERE - need to dump in specific model class if environment doesnt know the model.

    Loads the StrehlModel object from a pickle file.
    
    Args:
        filename (str): The file path from where the model should be loaded.
    
    Returns:
        StrehlModel: The loaded StrehlModel instance.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        
    return model

# def deserialize_object(obj, path="root"):
#     """
#     Recursively deserialize an object, converting lists back to tuples where needed
#     and handling SimpleNamespace recreation.
#     """
#     if isinstance(obj, dict):
#         # Detect if it's a SimpleNamespace-equivalent dictionary
#         if "__tuple__" in obj:
#             return tuple(deserialize_object(item, path=path) for item in obj["__tuple__"])
#         else:
#             # Reconstruct SimpleNamespace if applicable
#             return SimpleNamespace(**{key: deserialize_object(value, path=f"{path}.{key}") for key, value in obj.items()})
#     elif isinstance(obj, list):
#         return [deserialize_object(item, path=f"{path}[{idx}]") for idx, item in enumerate(obj)]
#     elif isinstance(obj, (int, float, str, bool, type(None))):
#         return obj  # Primitive types remain unchanged
#     else:
#         return obj  # Return as-is if no conversion rule applies




# def load_namespace_from_json(file_path):
#     """
#     Load the JSON file and convert it back into a SimpleNamespace object.
#     """
#     import json
#     with open(file_path, 'r') as f:
#         data = json.load(f)

#     # Convert serialized JSON back into SimpleNamespace objects
#     return deserialize_object(data)

# Example usage
#save_namespace_as_json(zwfs_ns, f'/Users/bencb/Downloads/zwfs_ns_config_.json', exclude_keys=['telem'])
#zwfs_ns2 = load_namespace_from_json('/Users/bencb/Downloads/zwfs_ns_config_.json')


#####################
## REAL TIME SIMULATION APP 
####################

def main(): 
    # # configure our zwfs 
    # grid_dict = {
    #     "D":1, # diameter of beam 
    #     "N" : 64, # number of pixels across pupil diameter
    #     #"padding_factor" : 4, # how many pupil diameters fit into grid x axis
    #     # TOTAL NUMBER OF PIXELS = padding_factor * N 
    #     "dim": 64 * 4
    #     }

    # optics_dict = {
    #     "wvl0" :1.65e-6, # central wavelength (m) 
    #     "F_number": 21.2, # F number on phasemask
    #     "mask_diam": 1.06, # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
    #     "theta": 1.57079, # phaseshift of phasemask 
    # }

    # dm_dict = {
    #     "dm_model":"BMC-multi-3.5",
    #     "actuator_coupling_factor":0.7,# std of in actuator spacing of gaussian IF applied to each actuator. (e.g actuator_coupling_factor = 1 implies std of poke is 1 actuator across.)
    #     "dm_pitch":1,
    #     "dm_aoi":0, # angle of incidence of light on DM 
    #     "opd_per_cmd" : 3e-6, # peak opd applied at center of actuator per command unit (normalized between 0-1) 
    #     "flat_rmse" : 20e-9 # std (m) of flatness across Flat DM  
    #     }

    # grid_ns = SimpleNamespace(**grid_dict)
    # optics_ns = SimpleNamespace(**optics_dict)
    # dm_ns = SimpleNamespace(**dm_dict)

    # zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

   

    # # initialize our ZWFS instrument
    #method='MVM-TT-HO' 
    method='zonal_interp_no_projection'
    #proj_path = os.getcwd() ###<---avoid this and use resources
    wvl0=1.25e-6
    use_pyZelda=False #True
    plot_intermediate=False # plotting calibration results?

    # Determine the path to the configuration file relative to this script
    config_ini = os.path.join(os.path.dirname(__file__), '../../configurations/BALDR_UT_J3.ini')

    # Normalize the path for cross-platform compatibility
    config_ini = os.path.abspath(config_ini)

    # with resources.path("baldrapp.configurations", "BALDR_UT_J3.ini") as config_path:
    #     config_ini = str(config_path)
    #config_ini = proj_path  + '/baldrapp/configurations/BALDR_UT_J3.ini'#'/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
    
    zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

    # #######################################################
    # # try with zonal method

    # short hand for pupil dimensions (pixels)
    #dim = zwfs_ns.grid.N * zwfs_ns.grid.padding_factor # should match zwfs_ns.pyZelda.pupil_dim
    # spatial differential in pupil space 
    dx = zwfs_ns.grid.D / zwfs_ns.grid.N
    # get required simulation sampling rate to match physical parameters 
    dt = dx * zwfs_ns.atmosphere.pixels_per_iteration / zwfs_ns.atmosphere.v # s # simulation sampling rate

    print(f'current parameters have effective wind velocity = {round(zwfs_ns.atmosphere.v )}m/s')
    scrn = phasescreens.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)


    # first stage AO 
    # basis_cropped = ztools.zernike.zernike_basis(nterms=150, npix=zwfs_ns.pyZelda.pupil_diameter)
    # # we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter) 
    # # so we need to put basis in the same frame  
    # basis_template = np.zeros( zwfs_ns.pyZelda.pupil.shape )
    # basis = np.array( [ util.insert_concentric( np.nan_to_num(b, 0), basis_template) for b in basis_cropped] )

    # #pupil_disk = basis[0] # we define a disk pupil without secondary - useful for removing Zernike modes later

    # Nmodes_removed = 14 # Default will be to remove Zernike modes 

    # # vibrations 
    # mode_indicies = [0, 1]
    # spectrum_type = ['1/f', '1/f']
    # opd = [50e-9, 50e-9]
    # vibration_frequencies = [15, 45] #Hz


    # input flux scaling (photons / s / wavespace_pixel / nm) 
    photon_flux_per_pixel_at_vlti = zwfs_ns.throughput.vlti_throughput * (np.pi * (zwfs_ns.grid.D/2)**2) / (np.pi * zwfs_ns.pyZelda.pupil_diameter/2)**2 * util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude, band = zwfs_ns.stellar.waveband, wavelength= 1e9*wvl0)

    amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
    
    opd_internal = util.apply_parabolic_scratches(np.zeros( zwfs_ns.grid.pupil_mask.shape ) , dx=dx, dy=dx, list_a= [ 0.1], list_b = [0], list_c = [-2], width_list = [2*dx], depth_list = [100e-9])

    opd_flat_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.dm_flat  , gain=zwfs_ns.dm.opd_per_cmd, \
                    sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                        x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )

    # quicker way - make sure get frames returns the same as the above!!!! 
    I0 = bldr.get_I0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
        zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)

    N0 = bldr.get_N0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
        zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)


    # Build a basic Interaction matrix (IM) for the ZWFS
    basis_name = 'Zonal_pinned_edges'
    Nmodes = 100
    #M2C_0 = DM_basis.construct_command_basis( basis= basis_name, number_of_modes = Nmodes, without_piston=True).T  


    # we must first define our pupil regions before building 
    zwfs_ns = bldr.classify_pupil_regions( opd_input=0*opd_internal,  amp_input=amp_input ,  opd_internal=opd_internal,  zwfs_ns=zwfs_ns , detector= zwfs_ns.detector , use_pyZelda = False) # For now detector is just tuple of pixels to average. useful to know is zwfs_ns.grid.N is number of pixels across pupil. # from this calculate an appropiate binning for detector 

    zwfs_ns = bldr.build_IM( zwfs_ns ,  calibration_opd_input = 0*opd_internal , calibration_amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil  , \
                opd_internal = opd_internal,  basis = basis_name, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
                    imgs_to_mean = 1, detector=zwfs_ns.detector)

    # from IM register the DM in the detector pixelspace 
    zwfs_ns = bldr.register_DM_in_pixelspace_from_IM( zwfs_ns, plot_intermediate_results=plot_intermediate )


    zwfs_ns = bldr.fit_linear_zonal_model( zwfs_ns, opd_internal, iterations = 100, photon_flux_per_pixel_at_vlti = photon_flux_per_pixel_at_vlti , \
        pearson_R_threshold = 0.6, phase_scaling_factor=0.2,   plot_intermediate=plot_intermediate , fig_path = None)

    #clear pupil on dm 
    N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

    # calibrate a model to map a subset of pixel intensities to Strehl Ratio 
    # should eventually come back and debug for model_type = lin_comb - since it seemed to work better intially
    #strehl_model = bldr.calibrate_strehl_model( zwfs_ns, save_results_path = fig_path, train_fraction = 0.6, correlation_threshold = 0.6, \
    #    number_of_screen_initiations = 60, scrn_scaling_grid = np.logspace(-2, -0.5, 5), model_type = 'PixelWiseStrehlModel' ) #lin_comb') 

    # or read one in  
    #strehl_model_file = proj_path  + '/baldrapp/configurations/strehl_model_config-BALDR_UT_J3_2024-10-19T09.28.27.pkl'
    # bug loading this since model class not explicitly imported here. works locally - need to cjheck
    #strehl_model = load_model_from_pickle(filename=strehl_model_file)

    #HO_ctrl.reset()
    #zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = True) # HO = 'leaky'
    # there was a reason why i made the controller dict seperate... but cant remember why
    zwfs_ns = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = False) # HO = 'leaky'
    # init all gains to 0

    #dm_disturbance = np.random.randn(140) * 1e-7
    dm_disturbance = np.zeros( 140 )


    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
    zwfs_ns = bldr.reset_telemetry( zwfs_ns )

    kwargs = {"N0_dm":N0_dm, "HO_ctrl": zwfs_ns.ctrl.HO_ctrl }#zonal_ctrl_dict['HO_ctrl']  } 
    phase_scaling_factor  = 0.07


    dynamic_opd_input = True


    """
    phase_scaling_factor
    zwfs_ns.ctrl.HO_ctrl.ki += 0.4
    dynamic_opd_input=False
    M2C_0 =  DM_basis.construct_command_basis( basis= "Zernike", number_of_modes = 20, without_piston=True).T  
    dm_disturbance =  M2C_0[5]* 1e-1 # check dm gain for idea of scaling to opd

    # lock 
    zwfs_ns.ctrl.HO_ctrl.ki += 0.4

    """



    # #######################################################
    # #### workin code 21-1-25
    # use_pyZelda = True #False

    # wvl0=1.25e-6
    # config_ini = 'configurations/BALDR_UT_J3.ini'#'/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
    # 955,7:     # config_ini = 'configurations/BALDR_UT_J3.ini'#'/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'#zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

    # opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

    # opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

    # amp_input = 1e4 * zwfs_ns.grid.pupil_mask


    # #basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

    # # Build IM  and look at Eigenmodes! 
    
    
    # # perfect field only with internal opd aberrations 
    # # different poke methods 
    # Nmodes = 100
    # # ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]
    # basis = 'Zonal_pinned_edges' #'fourier_pinned_edges' #'Hadamard' #'Zonal_pinned_edges'
    # poke_amp = 0.05
    # Smax = 30
    # detector = zwfs_ns.detector #(4,4) # for binning , zwfs_ns.grid.N is #pixels across pupil diameter (64) therefore division 4 = 16 pixels (between CRed2 and Cred1 )
    # #M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

    # #I0 = bldr.get_I0(  opd_input  = 0 *zwfs_ns.grid.pupil_mask ,   amp_input = amp_input,\
    # #    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

    # #N0 = bldr.get_N0(  opd_input  = 0 *zwfs_ns.grid.pupil_mask  ,   amp_input = amp_input,\
    # #    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

    # # we must first define our pupil regions before building 
    # zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector , use_pyZelda = False) # For now detector is just tuple of pixels to average. useful to know is zwfs_ns.grid.N is number of pixels across pupil. # from this calculate an appropiate binning for detector 

    # zwfs_ns = bldr.build_IM( zwfs_ns,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    #     opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = poke_amp, poke_method = 'double_sided_poke',\
    #         imgs_to_mean = 1, detector=detector, use_pyZelda = use_pyZelda  )

    # # look at the eigenmodes in camera, DM and singular values
    # bldr.plot_eigenmodes( zwfs_ns , save_path = None )

    # TT_vectors = gen_basis.get_tip_tilt_vectors()

    # HO_dm_disturb = gen_basis.construct_command_basis( 'fourier_pinned_edges')


    # #zwfs_ns = bldr.construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 50, TT_vectors = TT_vectors )
    # zwfs_ns = bldr.construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 20, TT_vectors = TT_vectors )

    # #zwfs_ns = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns, TT = 'PID', HO = 'pid')
    # zwfs_ns = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns, TT = 'PID', HO = 'leaky')

    # #zwfs_ns = init_CL_simulation( zwfs_ns,  opd_internal, amp_input , basis, Nmodes, poke_amp, Smax )
      
    # dm_disturbance = 0. * TT_vectors.T[0]
    
    # #scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size= zwfs_ns.grid.N * zwfs_ns.grid.padding_factor, pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N ,r0=0.1,L0=12)
    # #scrn = phasescreens.PhaseScreenVonKarman(nx_size = zwfs_ns.grid.N * zwfs_ns.grid.padding_factor, pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N ,r0=0.1,L0=12)
    # scrn = phasescreens.PhaseScreenVonKarman(nx_size = zwfs_ns.grid.dim, pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N ,r0=0.1,L0=12)
    # dynamic_opd_input = True
    
    # #dm_disturbance = 0.1 * HO_dm_disturb.T[3]
    # #zwfs_ns.dm.current_cmd =  zwfs_ns.dm.dm_flat + disturbance_cmd 

    # zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    # zwfs_ns.ctrl.TT_ctrl.reset()
    # zwfs_ns.ctrl.HO_ctrl.reset()
    # zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
    # zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()

    # zwfs_ns.ctrl.HO_ctrl.ki = 0. * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
    # zwfs_ns.ctrl.HO_ctrl.kp = 0. * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

    # zwfs_ns.ctrl.TT_ctrl.kp = 0. * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    # zwfs_ns.ctrl.TT_ctrl.ki = 0. * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
    
    
    """
    # some prompts 
    # change dm disturbance
    dm_disturbance = 0.1 * HO_dm_disturb.T[3] + 0.1 * TT_vectors.T[0]
    # change TT 
    zwfs_ns.ctrl.TT_ctrl.kp = 1 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    zwfs_ns.ctrl.TT_ctrl.ki = 0.5 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    zwfs_ns.ctrl.HO_ctrl.ki[2:6] = 0.5 
    
    dynamic_opd_input = True
    """

    # only execute once in a given session 
    app = QtWidgets.QApplication(sys.argv)
    
    window = AOControlApp()
    window.setWindowTitle("AO Control GUI")
    window.show()
    sys.exit(app.exec_())



"""        class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Layout
        self.layout = QtWidgets.QVBoxLayout()

        # Buttons
        self.run_button = QtWidgets.QPushButton("Run")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.zero_gains_button = QtWidgets.QPushButton("Set Gains to Zero")
        self.reset_button = QtWidgets.QPushButton("Reset")
        #self.reset_button = QtWidgets.QPushButton("reset")
        
        # Visual "LED" for running state
        self.status_led = QtWidgets.QLabel()
        self.status_led.setFixedSize(20, 20)
        self.update_led(False)  # Initialize as "not running" (False)

        # Connect buttons to functions
        self.run_button.clicked.connect(self.run_loop)
        self.pause_button.clicked.connect(self.pause_loop)
        self.zero_gains_button.clicked.connect(self.set_gains_to_zero)
        self.reset_button.clicked.connect(self.reset)
        #self.reset_button.clicked.connect(self.reset)

        # Text input for user
        ## older input style
        #self.input_label = QtWidgets.QLabel("User Input:")
        #self.text_input = QtWidgets.QLineEdit()
        #self.text_input.returnPressed.connect(self.check_input)
        
        
        # Terminal-like prompt setup
        self.prompt_history = QtWidgets.QTextEdit()
        self.prompt_history.setReadOnly(True)  # History is read-only
        self.prompt_input = QtWidgets.QLineEdit()
        self.prompt_input.returnPressed.connect(self.handle_command)

        sys.stdout = OutputRedirector(self.prompt_history)  # Redirect stdout

        # Add prompt history and input to layout
        self.layout.addWidget(self.prompt_history)
        self.layout.addWidget(self.prompt_input)

        # Store history of entered commands
        self.command_history = []
        self.history_index = -1  # Track the command index

        # Set the layout

        # Add buttons and input to layout
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.pause_button)
        self.layout.addWidget(self.zero_gains_button)
        self.layout.addWidget(self.reset_button)
        
        #self.layout.addWidget(self.reset_button)
        #self.layout.addWidget(self.input_label)
        #self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.status_led)  # Add LED to the layout
        
        # PyQtGraph plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)
"""

if __name__ == "__main__":
    main()