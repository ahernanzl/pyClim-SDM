import sys
import os
import shutil
sys.path.append('../config/')
from manual_settings import *
if not os.path.isfile('../config/settings.py') or os.stat('../config/settings.py').st_size == 0:
    shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
from imports import *
from settings import *
from advanced_settings import *
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from tkinter import *
from PIL import ImageTk,Image


########################################################################################################################

def switch(*args):
    """To change the state of all objects passed as arguments"""
    for object in args:
        if object["state"] == "normal":
            object["state"] = "disabled"
        else:
            object["state"] = 'normal'

########################################################################################################################
def switch_splitMode(split_modeName, dict):
    """To activate only testing years of the active split mode"""
    for key in dict:
        switcher = dict[key][0]
        entries = dict[key][1:]

        if key not in ('all_training', 'all_testing'):
            for object in entries:
                if switcher["state"] == "active":
                    object["state"] = "normal"
                else:
                    object["state"] = 'disabled'

# ########################################################################################################################
# def switch_steps(exp, only_for_projections):
#     """To enable/disable steps depending on the experiment"""
    # for object in only_for_projections:
    #     if exp == 'PROJECTIONS':
    #         object["state"] = "normal"
    #     else:
    #         object["state"] = 'disabled'


########################################################################################################################
def switch_steps(exp, steps, steps_ordered, exp_ordered, chk_only_for_experiment):
    """To enable/disable steps depending on the experiment"""

    for i in range(len(chk_only_for_experiment)):
        object = chk_only_for_experiment[i]
        step = steps_ordered[i]
        if step in steps[exp] and exp == exp_ordered[i]:
            object["state"] = "normal"
        else:
            object["state"] = "disabled"


########################################################################################################################
def CreateToolTip(widget, text):
    """This function displays information when the mouse cursor is over the object"""

    class ToolTip(object):

        def __init__(self, widget):
            self.widget = widget
            self.tipwindow = None
            self.id = None
            self.x = self.y = 0

        def showtip(self, text):
            "Display text in tooltip window"
            self.text = text
            if self.tipwindow or not self.text:
                return
            x, y, cx, cy = self.widget.bbox("insert")
            x = x + self.widget.winfo_rootx() + 57
            y = y + cy + self.widget.winfo_rooty() + 27
            self.tipwindow = tw = Toplevel(self.widget)
            tw.wm_overrideredirect(1)
            tw.wm_geometry("+%d+%d" % (x, y))
            label = Label(tw, text=self.text, justify=LEFT,
                          background="#ffffe0", relief=SOLID, borderwidth=1,
                          font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)

        def hidetip(self):
            tw = self.tipwindow
            self.tipwindow = None
            if tw:
                tw.destroy()
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


########################################################################################################################
class welcomeMessage(ttk.Frame):
    """This function displays a welcome message which can be enabled for next runs"""

    def __init__(self):
        if showWelcomeMessage == True:
            root = tk.Tk()
            root.title("Welcome to pyClim-SDM")

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            totalW, totalH = 1280, 620
            if (screen_width < totalW) or (screen_height < totalH):
                print('Your screen resolution is too small. ')
                print('Please change your screen resolution to the minimum required resolution: w='+str(totalW)+' and h='+str(totalH)+'.')
                print(
                    'If not possible, you can use pyClim-SDM without graphical user interface by editing config/manual_settings.py')
                print('and src/manual_mode.py, and running the last one.')
                exit()

            welcomeWinW, welcomeWinH = 900, 600
            root.minsize(welcomeWinW, welcomeWinH)
            root.maxsize(welcomeWinW, welcomeWinH)
            offset = int((totalW-welcomeWinW)/2)
            root.geometry(str(welcomeWinW)+'x'+str(welcomeWinH)+'+'+str(offset)+'+50')

            # frameLogo
            frameLogo = Frame(root)
            frameLogo.grid(column=0, row=0, padx=0)

            # Logo
            Label(root, text='', borderwidth=0, background=None).grid(sticky="SE", column=0, row=0, pady=0)
            w = 800
            img = Image.open("../doc/pyClim-SDM_logo.png")
            h = int(w * img.height / img.width)
            img = img.resize((w, h), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            canvas = Canvas(frameLogo, width=w, height=h)
            canvas.create_image(0, 0, anchor=NW, image=img)
            canvas.grid(column=0, row=1, padx=0, pady=0)

            # frameMsg
            frameMsg = Frame(root)
            frameMsg.grid(sticky="W", column=0, row=1, padx=150, pady=0)

            dontShowAgain_local = tk.BooleanVar()
            l = Label(frameMsg,
                text="Welcome to pyClim-SDM. Please, create an input_data directory following the structure and \n"
                     "format indicated in the input_data_template, where example datasets have been included.\n",
            )
            l.pack(padx=20, pady=0, fill='both')
            c = Checkbutton(frameMsg, text="Do not show this dialog again", variable=dontShowAgain_local)
            c.pack(padx=10, pady=10)
            self.run = False
            def run():
                self.run = True
                root.destroy()
            b = Button(frameMsg, text="Ok", command=run)
            b.pack(padx=20, pady=10)
            b.mainloop()

            self.showWelcomeMessage_new = not dontShowAgain_local.get()
        else:
            self.run = True
            self.showWelcomeMessage_new = False

    def get(self):
        return self.run, self.showWelcomeMessage_new


########################################################################################################################
class tabSteps(ttk.Frame):

    def __init__(self, notebook, root):

        tabSteps = ttk.Frame(notebook)
        notebook.add(tabSteps, text='Experiment and Steps')
        self.chk_dict = {}
        self.rdbuts = []
        self.chk_only_for_experiment = []
        self.steps_ordered = []
        self.exp_ordered = []

        irow = 0
        ttk.Label(tabSteps, text="").grid(column=0, row=irow, padx=50)
        ttk.Label(tabSteps, text="").grid(column=1, row=irow, pady=40); irow+=1

        # Experiment
        icol = 1
        ttk.Label(tabSteps, text="Select experiment:").grid(sticky="E", column=icol, row=irow, padx=10, pady=15); icol+=1
        self.experiment = StringVar()
        experiments = {'PRECONTROL': 'Evaluation of predictors and GCMs previous to dowscaling',
                       'EVALUATION': 'Evaluate methods using a reanalysis over a historical period',
                       'PROJECTIONS': 'Apply methods to dowscale climate projections'}
        for exp in experiments:
            c = Radiobutton(tabSteps, text=exp, variable=self.experiment, value=exp,
                            command=lambda: switch_steps(self.experiment.get(), steps, self.steps_ordered,
                            self.exp_ordered, self.chk_only_for_experiment), takefocus=False)
            c.grid(sticky="W", column=icol, row=irow, padx=30); icol+=1
            CreateToolTip(c, experiments[exp])
            self.experiment.set(experiment)

        irow += 2
        icol = 1

        # Steps definition
        ttk.Label(tabSteps, text="Select steps:").grid(sticky="E", column=icol, row=irow, padx=10); irow+=1

        steps = {'PRECONTROL': {
                 'preprocess': {'text': 'Preprocess', 'info':  'Association between target points and the low \n'
                                                                   'resolution grid, calculation of derived predictors, \n'
                                                                   'standardization of predictors, training/testing split \n'
                                                                   'and weather types clustering.'},
                 'missing_data_check': {'text': 'Missing data check', 'info':  'Check for missing data in predictors by GCMs.',},
                 'predictors_correlation': {'text': 'Predictors correlation', 'info': 'Test the strength of the\n'
                                                                      'predictors/predictand relationships.'},
                 'GCMs_evaluation': {'text': 'GCMs evaluation', 'info': 'Test the reliability of GCMs in a historical period\n'
                                                                            'comparing them with a reanalysis, and also the\n'
                                                                            'uncertainty in the future.'},},

                 'EVALUATION': {
                 'preprocess': {'text': 'Preprocess', 'info':  'Association between target points and the low \n'
                                                                   'resolution grid, calculation of derived predictors, \n'
                                                                   'standardization of predictors, training/testing split \n'
                                                                   'and weather types clustering.'},
                'train_methods': {'text': 'Train methods', 'info': 'Train of all selected methods. \n'
                                                                   'If you are working in a HPC, you can assign different \n'
                                                                   'configuration (number of nodes, memory, etc) to each \n'
                                                                   'method by editing the lib/launch_jobs.py file.'},
                'downscale': {'text': 'Downscale', 'info': 'Apply all selected methods. If you are \n'
                                                           'working in a HPC, you can assign different configuration \n'
                                                           '(number of nodes, memory, etc) to each method by editing the \n'
                                                           'lib/launch_jobs.py file. Dowscaled data will be storaged in the \n'
                                                           'results/ directory.'},
                'calculate_climdex': {'text': 'Calculate climdex', 'info': 'Calculate all selected climdex.'},
                'plot_results': {'text': 'Plot results', 'info': 'Generate figures and storage them in results/figures/. \n'
                                                                 'A different set of figures will be generated depending on the \n'
                                                                 'selected experiment (EVALUATION / PROJECTIONS).'},
                 'nc2ascii': {'text': 'Convert binary files to ASCII', 'info': 'Convert binary files to ASCII.'}},

                 'PROJECTIONS': {'preprocess': {'text': 'Preprocess', 'info':  'Association between target points and the low \n'
                                                                   'resolution grid, calculation of derived predictors, \n'
                                                                   'standardization of predictors, training/testing split \n'
                                                                   'and weather types clustering.'},
                'train_methods': {'text': 'Train methods', 'info': 'Train of all selected methods. \n'
                                                                   'If you are working in a HPC, you can assign different \n'
                                                                   'configuration (number of nodes, memory, etc) to each \n'
                                                                   'method by editing the lib/launch_jobs.py file.'},
                'downscale': {'text': 'Downscale', 'info': 'Apply all selected methods. If you are \n'
                                                           'working in a HPC, you can assign different configuration \n'
                                                           '(number of nodes, memory, etc) to each method by editing the \n'
                                                           'lib/launch_jobs.py file. Dowscaled data will be storaged in the \n'
                                                           'results/ directory.'},
                'bias_correct_projections': {'text': 'Bias correct projections (optional)', 'info': 'Bias correct '
                                                                                                    'downscaled projections.'},
                'calculate_climdex': {'text': 'Calculate climdex', 'info': 'Calculate all selected climdex.'},
                'plot_results': {'text': 'Plot results', 'info': 'Generate figures and storage them in results/figures/. \n'
                                                                 'A different set of figures will be generated depending on the \n'
                                                                 'selected experiment (EVALUATION / PROJECTIONS).'},
                'nc2ascii': {'text': 'Convert binary files to ASCII', 'info': 'Convert binary files to ASCII.'}}}

        self.all_steps = steps

        # Create steps_ordered
        for exp_name in (experiments):
            for step in steps[exp_name]:
                self.steps_ordered.append(step)
                self.exp_ordered.append(exp_name)

        # Steps check buttons
        irow -= 1
        for exp_name in (experiments):
            nrows = 0
            icol +=1
            aux_dict = {}
            for step in steps[exp_name]:
                checked = tk.BooleanVar()
                c = Checkbutton(tabSteps, text=steps[exp_name][step]['text'], variable=checked, takefocus=False)
                c.grid(sticky="W", column=icol, row=irow, padx=50); irow+=1
                CreateToolTip(c, steps[exp_name][step]['info'])
                aux_dict.update({step: checked})
                self.chk_only_for_experiment.append(c)
                if exp_name == experiment:
                    c.config(state='normal')
                else:
                    c.config(state='disabled')
                nrows += 1
            self.chk_dict.update({exp_name: aux_dict})
            irow -= nrows

    def get(self):
        return self.experiment, self.chk_dict, self.all_steps


########################################################################################################################
class tabMethods(ttk.Frame):

    def __init__(self, notebook):

        # ---------------------- tab methods ---------------------------------------------------------------------------------

        self.chk_list = []
        tabMethods = ttk.Frame(notebook)
        notebook.add(tabMethods, text='Methods')

        def add_to_chk_list(chk_list, var, methodName, family, mode, fields, info, icol, irow):
            """This function adds all methods to a list. The checked variable will contain information about their status
            once the mainloop is finished"""

            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            for method_dict in methods:
                if (method_dict['var'], method_dict['methodName']) == (var, methodName):
                    checked = tk.BooleanVar(value=True)

            c = Checkbutton(tabMethods, text=methodName, variable=checked, takefocus=False)
            if var == 'tmax':
                cbuts_tmax.append(c)
            elif var == 'tmin':
                cbuts_tmin.append(c)
            elif var == 'pcp':
                cbuts_pcp.append(c)
            CreateToolTip(c, info)
            c.grid(sticky="W", column=icol, row=irow, padx=10)
            self.chk_list.append({'var': var, 'methodName': methodName, 'family': family, 'mode': mode, 'fields': fields, 'checked': checked})
            return self.chk_list

        # Functions for selecting/deselecting all
        cbuts_tmax, cbuts_tmin, cbuts_pcp = [], [], []
        buttonWidth = 8
        def select_all_tmax():
            for i in cbuts_tmax:
                i.select()
        def select_all_tmin():
            for i in cbuts_tmin:
                i.select()
        def select_all_pcp():
            for i in cbuts_pcp:
                i.select()
        def deselect_all_tmax():
            for i in cbuts_tmax:
                i.deselect()
        def deselect_all_tmin():
            for i in cbuts_tmin:
                i.deselect()
        def deselect_all_pcp():
            for i in cbuts_pcp:
                i.deselect()

        # ---------------------- tmax and tmin ----------------------
        for var in ('tmax', 'tmin', ):
            if var == 'tmax':
                icol, irow, variable = 0, 0, 'Maximum Temperature'
            elif var == 'tmin':
                icol, irow, variable = 2, 0, 'Minimum Temperature'

            ttk.Label(tabMethods, text="").grid(column=icol, row=irow, padx=40, pady=0); icol+=1; irow+=1
            ttk.Label(tabMethods, text=variable)\
                .grid(sticky="W", column=icol, row=irow, padx=30, pady=20, columnspan=4); irow+=1

            # Raw
            add_to_chk_list(self.chk_list, var, 'RAW', 'RAW', 'RAW', 'var', 'No downscaling', icol, irow); irow+=1

            # Bias correction
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow+=1
            ttk.Label(tabMethods, text="Bias Correction:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'QM', 'BC', 'MOS', 'var', 'Quantile Mapping', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'DQM', 'BC', 'MOS', 'var', 'Detrended Quantile Mapping', icol, irow); icol+=1; irow-=1
            add_to_chk_list(self.chk_list, var, 'QDM', 'BC', 'MOS', 'var', 'Quantile Delta Mapping', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'PSDM', 'BC', 'MOS', 'var', '(Parametric) Scaled Distribution Mapping', icol, irow); icol-=1; irow+=1

            # Analogs / Weather Typing
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Analogs / Weather Typing:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'ANA-MLR', 'ANA', 'PP', 'pred+saf', 'Analog Multiple Linear Regression', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'WT-MLR', 'ANA', 'PP', 'pred+saf', 'Weather Typing Multiple Linear Regression', icol, irow); irow+=1

            # Transfer function
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow+=1
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Transfer Function:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'MLR', 'TF', 'PP', 'pred', 'Multiple Linear Regression', icol, irow); icol+=1
            add_to_chk_list(self.chk_list, var, 'ANN', 'TF', 'PP', 'pred', 'Artificial Neural Network', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'SVM', 'TF', 'PP', 'pred', 'Support Vector Machine', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'LS-SVM', 'TF', 'PP', 'pred', 'Least Square Support Vector Machine', icol, irow); icol-=1; irow+=1

            # Weather Generators
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Weather Generators:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4); irow+=1
            add_to_chk_list(self.chk_list, var, 'WG-PDF', 'WG', 'WG', 'var', 'Weather generator from downscaled PDF', icol, irow); irow+=1

            # Select/deselect all
            # ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            if var == 'tmax':
                Button(tabMethods, text='Select all', command=select_all_tmax, width=buttonWidth, takefocus=False).grid(sticky="E", column=icol, row=irow); icol += 1
                Button(tabMethods, text='Deselect all', command=deselect_all_tmax, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow)
            elif var == 'tmin':
                Button(tabMethods, text='Select all', command=select_all_tmin, width=buttonWidth, takefocus=False).grid(sticky="E", column=icol, row=irow); icol += 1
                Button(tabMethods, text='Deselect all', command=deselect_all_tmin, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow)


            ttk.Label(tabMethods, text="").grid(column=icol, row=irow, padx=130, pady=25); icol+=1

        # ---------------------- pcp ----------------------
        var = 'pcp'
        icol, irow, variable = 5, 0, 'Precipitation'

        ttk.Label(tabMethods, text="").grid(column=icol, row=irow, padx=5); irow+=1; icol += 1
        ttk.Label(tabMethods, text=variable)\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1

        # Raw
        add_to_chk_list(self.chk_list, var, 'RAW', 'RAW', 'RAW', 'var', 'No downscaling', icol, irow); irow += 1

        # Bias correction
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Bias Correction:")\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_to_chk_list(self.chk_list, var, 'QM', 'BC', 'MOS', 'var', 'Quantile Mapping', icol, irow); irow += 1
        add_to_chk_list(self.chk_list, var, 'DQM', 'BC', 'MOS', 'var', 'Detrended Quantile Mapping', icol, irow); icol+=1; irow -= 1
        add_to_chk_list(self.chk_list, var, 'QDM', 'BC', 'MOS', 'var', 'Quantile Delta Mapping', icol, irow); irow += 1
        add_to_chk_list(self.chk_list, var, 'PSDM', 'BC', 'MOS', 'var', '(Parametric) Scaled Distribution Mapping', icol, irow); icol-=1; irow += 1

        # Analogs / Weather Typing
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Analogs / Weather Typing:")\
                                                .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_to_chk_list(self.chk_list, var, 'ANA-SYN-1NN', 'ANA', 'PP', 'saf', 'Nearest neighbour based on synoptic fields', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-SYN-kNN', 'ANA', 'PP', 'saf', 'k nearest neighbours based on synoptic fields', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-SYN-rand', 'ANA', 'PP', 'saf', 'Random neighbour based on synoptic fields', icol, irow);  irow-=2; icol+=1
        add_to_chk_list(self.chk_list, var, 'ANA-LOC-1NN', 'ANA', 'PP', 'pred+saf', 'Nearest neighbour based on combined synoptic and local analogies', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-LOC-kNN', 'ANA', 'PP', 'pred+saf', 'k nearest neighbours based on combined synoptic and local analogies', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-LOC-rand', 'ANA', 'PP', 'pred+saf', 'Random neighbour based on combined synoptic and local analogies', icol, irow); irow-=2; icol+=1
        add_to_chk_list(self.chk_list, var, 'ANA-PCP-1NN', 'ANA', 'PP', 'pcp', 'Nearest neighbour based on precipitation pattern', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-PCP-kNN', 'ANA', 'PP', 'pcp', 'k nearest neighbours based on precipitation pattern', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'ANA-PCP-rand', 'ANA', 'PP', 'pcp', 'Random neighbour based on precipitation pattern', icol, irow); irow+=1; icol-=2

        # Transfer function
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Transfer Function:")\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-LIN', 'TF', 'PP', 'pred', 'Generalized Linear Model (linear)', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-EXP', 'TF', 'PP', 'pred', 'Generalized Linear Model (exponential)', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-CUB', 'TF', 'PP', 'pred', 'Generalized Linear Model (cubic)', icol, irow); irow-=2; icol+=1
        add_to_chk_list(self.chk_list, var, 'ANN', 'TF', 'PP', 'pred', 'Artificial Neural Network', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'SVM', 'TF', 'PP', 'pred', 'Support Vector Machine', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'LS-SVM', 'TF', 'PP', 'pred', 'Least Square Support Vector Machine', icol, irow); irow+=1; icol-=1

        # Weather Generators
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Weather Generators:")\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_to_chk_list(self.chk_list, var, 'WG-NMM', 'WG', 'WG', 'var', 'Weather generator Non-homogeneous Markov Model', icol, irow); icol += 1
        add_to_chk_list(self.chk_list, var, 'WG-PDF', 'WG', 'WG', 'var', 'Weather generator from downscaled PDF', icol, irow); irow += 1; icol -=1

        # Select/deselect all
        # ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        Button(tabMethods, text='Select all', command=select_all_pcp, width=buttonWidth, takefocus=False).grid(sticky="E", column=icol, row=irow); icol += 1
        Button(tabMethods, text='Deselect all', command=deselect_all_pcp, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow); icol-=1; irow += 2

    def get(self):
        return self.chk_list

########################################################################################################################
class tabPredictors(ttk.Frame):

    def __init__(self, notebook):

        tabPredictors = ttk.Frame(notebook)
        notebook.add(tabPredictors, text='Predictors')

        def add_chk_bt_upperAir(chk_list, pred, block, irow, icol):
            """Check buttons for upper air predictors"""
            checked = tk.BooleanVar(value=False)
            if pred in pred_dictIn:
                checked = tk.BooleanVar(value=True)
            c = Checkbutton(tabPredictors, variable=checked, takefocus=False)
            c.grid(row=irow, padx=padx, column=icol)
            chk_list.update({block+'_'+pred: checked})
            irow += 1
            return irow, icol

        def add_chk_bt_singleLevels(chk_list, pred, block, irow, icol, nrows):
            """Check buttons for single levels predictors"""
            checked = tk.BooleanVar(value=False)
            if pred in pred_dictIn:
                checked = tk.BooleanVar(value=True)
            c = Checkbutton(tabPredictors, text=pred, variable=checked, takefocus=False)
            c.grid(sticky="W", row=irow, padx=2, column=icol, columnspan=5)
            CreateToolTip(c, singleLevelVars[pred])
            chk_list.update({block+'_'+pred: checked})
            irow += 1
            nrows += 1
            if nrows == 10:
                nrows = 0
                icol += 5
                irow -= 10
            return irow, icol, nrows

        irow = 0
        icol = 0


        Label(tabPredictors, text="").grid(sticky="W", padx=10, row=irow, column=icol); icol += 1
        ttk.Label(tabPredictors, text="").grid(sticky="W", column=icol, columnspan=100, row=irow, padx=20, pady=0); irow+=2

        # Levels
        ttk.Label(tabPredictors, text="").grid(sticky="E", column=icol, row=irow, padx=30); irow+=1
        ttk.Label(tabPredictors, text="").grid(sticky="E", column=icol, row=irow, pady=10, padx=30); irow+=1
        self.levels = [1000, 850, 700, 500, 250]
        for level in self.levels:
            Label(tabPredictors,  text=str(level) + " hPa").grid(sticky="E", padx=30,  row=irow, column=icol); irow+=1
        Label(tabPredictors, text="").grid(sticky="E", column=icol, row=irow, padx=20); irow-=6; icol+=1

        self.preds = {}
        for block in ('t', 'p', 'saf'):

            if block == 't':
                title = 'Maximum / Minimum Temperature'
                pred_dictIn = preds_dict['t']
            elif block == 'p':
                title = 'Precipitation'
                pred_dictIn = preds_dict['p']
            else:
                title = 'Synoptic Analogy Fields'
                pred_dictIn = saf_dict

            irow -= 1
            Label(tabPredictors, text=title).grid(columnspan=10, row=irow, column=icol)
            irow += 1
            upperAirVars = {'u': 'Eastward wind component',
                            'v': 'Northward wind component',
                            't': 'Temperature',
                            'z': 'Geopotential',
                            'q': 'Specific humidity',
                            'r': 'Relative humidity (derived from t and q)',
                            'td': 'Dew point (derived from q)',
                            'Dtd': 'Dew point depresion (derived from t and q)',
                            'vort': 'Vorticity (derived from u and v)',
                            'div': 'Divergence (derived from u and v)',
                            }

            for var in upperAirVars:
                c = ttk.Label(tabPredictors, text=var)
                c.grid(column=icol, row=irow, pady=10); irow += 1
                CreateToolTip(c, upperAirVars[var])
                padx = 2
                for level in self.levels:
                    irow, icol = add_chk_bt_upperAir(self.preds, str(var) + str(level), block, irow, icol)
                irow -= 6; icol += 1

            Label(tabPredictors, text="").grid(sticky="W", padx=35, row=irow, column=icol); icol += 1

            irow += 6
            icol -= 11

            singleLevelVars = {
                            'mslp': 'Mean sea level pressure',
                            'u10': 'Surface eastward wind component',
                            'v10': 'Surface northward wind component',
                            't2m': 'Surface mean temperature',
                            'tmax': 'Surface maximum temperature',
                            'tmin': 'Surface minimum temperature',
                            'pcp': 'Daily precipitation',
                            'mslp_trend': 'Mean sea level trend (from previous day)',
                            'ins': 'Theoretical insolation (hours of sun)',
                            'vtg_1000_850': 'Vertical Thermal Gradient (between 1000-850 hPa)',
                            'vtg_850_700': 'Vertical Thermal Gradient (between 850-700 hPa)',
                            'vtg_700_500': 'Vertical Thermal Gradient (between 700-500 hPa)',
                            'ugsl': 'Eastward component of the geostrophic wind at sea level (derived from t and mslp)',
                            'vgsl': 'Northward component of the geostrophic wind at sea level (derived from t and mslp)',
                            'vortgsl': 'Vorticity of the geostrophic wind at sea level (derived from t and mslp)',
                            'divgsl': 'Divergence of the geostrophic wind at sea level (derived from t and mslp)',
                            'K_index': 'K instability index',
                            'TT_index': 'Total Totals instability index',
                            'SSI_index': 'Showalter instability index',
                            'LI_index': 'Lifted instability index',
                            }

            # Label(tabPredictors, text="").grid(sticky="W", row=irow, column=icol);
            irow += 1
            nrows = 0
            for pred in singleLevelVars:
                irow, icol, nrows = add_chk_bt_singleLevels(self.preds, pred, block, irow, icol, nrows)
            irow -= 7
            icol += 11

        # reaNames and modNames
        irow, icol = 20, 2
        frameVarNames = ttk.Frame(tabPredictors)
        frameVarNames.grid(sticky="W", column=icol, row=irow, pady=20, columnspan=100)
        irow, icol = 0, 0

        ttk.Label(frameVarNames, text='').grid(column=icol, row=irow, pady=0, ); irow+=1
        ttk.Label(frameVarNames, text='Define variable names in netCDF files:')\
            .grid(column=icol, row=irow, pady=10, columnspan=13); irow += 1
        ttk.Label(frameVarNames, text='').grid(column=icol, row=irow, pady=3, padx=60); irow+=1
        ttk.Label(frameVarNames, text='Reanalysis:').grid(sticky="E", column=icol, row=irow, padx=10, ); irow+=1
        ttk.Label(frameVarNames, text='Models:').grid(sticky="E", column=icol, row=irow, padx=10,); irow+=1
        icol+=1; irow-=3

        # reaNames and modNames
        self.reaNames = {}
        self.modNames = {}
        for var in reaNames:
            if var in upperAirVars:
                info = upperAirVars[var]
            elif var in singleLevelVars:
                info = singleLevelVars[var]
            lab = ttk.Label(frameVarNames, text=var)
            CreateToolTip(lab, info)
            lab.grid(sticky="E", column=icol, row=irow); irow+=1

            reaName = reaNames[var]
            self.reaName_var = tk.StringVar()
            reaName_Entry = tk.Entry(frameVarNames, textvariable=self.reaName_var, width=8, justify='right', takefocus=False)
            reaName_Entry.insert(END, reaName)
            reaName_Entry.grid(sticky="W", column=icol, row=irow); irow+=1
            self.reaNames.update({var: self.reaName_var})

            modName = modNames[var]
            self.modName_var = tk.StringVar()
            modName_Entry = tk.Entry(frameVarNames, textvariable=self.modName_var, width=8, justify='right', takefocus=False)
            modName_Entry.insert(END, modName)
            modName_Entry.grid(sticky="W", column=icol, row=irow); icol+=1; irow-=2
            self.modNames.update({var: self.modName_var})


    def get(self):
        return self.reaNames, self.modNames, self.preds


########################################################################################################################
class tabClimdex(ttk.Frame):

    def __init__(self, notebook):

        tabClimdex = ttk.Frame(notebook)
        notebook.add(tabClimdex, text='Climdex')
        self.chk_list = []

        def add_to_chk_list(chk_list, var, climdex, info, icol, irow):

            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            if climdex in climdex_names[var]:
                checked = tk.BooleanVar(value=True)

            c = Checkbutton(tabClimdex, text=climdex, variable=checked, takefocus=False)
            if var == 'tmax':
                cbuts_tmax.append(c)
            elif var == 'tmin':
                cbuts_tmin.append(c)
            elif var == 'pcp':
                cbuts_pcp.append(c)
            CreateToolTip(c, info)
            c.grid(sticky="W", column=icol, row=irow, padx=10)
            self.chk_list.append({'var': var, 'climdex': climdex, 'checked': checked})
            return self.chk_list

        # Functions for selecting/deselecting all
        cbuts_tmax, cbuts_tmin, cbuts_pcp = [], [], []
        buttonWidth = 8
        def select_all_tmax():
            for i in cbuts_tmax:
                i.select()
        def select_all_tmin():
            for i in cbuts_tmin:
                i.select()
        def select_all_pcp():
            for i in cbuts_pcp:
                i.select()
        def deselect_all_tmax():
            for i in cbuts_tmax:
                i.deselect()
        def deselect_all_tmin():
            for i in cbuts_tmin:
                i.deselect()
        def deselect_all_pcp():
            for i in cbuts_pcp:
                i.deselect()

        climdex_dict = {
            'tmax': {
                'TXm': 'Mean value of daily maximum temperature',
                'TXx': 'Maximum value of daily maximum temperature',
                'TXn': 'Minimum value of daily maximum temperature',
                'TX90p': 'Percentage of days when TX > 90th percentile',
                'TX10p': 'Percentage of days when TX < 10th percentile',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                'SU': 'Number of summer days',
                'ID': 'Number of icing days',
                'WSDI': 'Warm spell duration index',
            },
            'tmin': {
                'TNm': 'Mean value of daily minimum temperature',
                'TNx': 'Maximum value of daily minimum temperature',
                'TNn': 'Minimum value of daily minimum temperature',
                'TN90p': 'Percentage of days when TN > 90th percentile',
                'TN10p': 'Percentage of days when TN < 10th percentile',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                'FD': 'Number of frost days',
                'TR': 'Number of tropical nights',
                'CSDI': 'Cold spell duration index',
            },
            'pcp': {
                'Pm': 'Mean precipitation amount',
                'PRCPTOT': 'Total precipitation on wet days',
                'R01': 'Number of days when PRCP ≥ 1 mm (wet days)',
                'SDII': 'Simple precipitation intensity index (Mean precipitation on wet days)',
                'Rx1day': 'Maximum 1-day precipitation',
                'Rx5day': 'Maximum consecutive 5-day precipitation',
                'R10mm': 'Number of days when PRCP ≥ 10mm',
                'R20mm': 'Number of days when PRCP ≥ 20mm',
                'p95': '95th percentile',
                'R95p': 'Total PRCP when RR > 95th percentile (total precipitation on very wet days)',
                'R95pFRAC': 'Fraction of total PRCP when RR > 95th percentile (fraction of total precipitation on very wet days)',
                'p99': '99th percentile',
                'R99p': 'Total PRCP when RR > 99th percentile (total precipitation on very wet days)',
                'R99pFRAC': 'Fraction of total PRCP when RR > 99th percentile (fraction of total precipitation on very wet days)',
                'CDD': 'Maximum length of dry spell',
                'CWD': 'Maximum length of wet spell',
            }
        }


        irow, icol = 0, 0
        ttk.Label(tabClimdex, text="").grid(sticky="W", column=icol, row=irow, padx=80, pady=10)

        for (var, title) in [('tmax', 'Maximum Temperature'), ('tmin', 'Minimum Temperature'), ('pcp', 'Precipitation')]:
            irow = 1
            Label(tabClimdex, text='').grid(column=icol, row=irow, padx=50, pady=30); icol+=1
            Label(tabClimdex, text=title).grid(sticky="W", column=icol, row=irow, padx=30, pady=30, columnspan=3); irow+=1
            nrows = 1
            colJumps = 0
            for climdex in climdex_dict[var]:
                add_to_chk_list(self.chk_list, var, climdex, climdex_dict[var][climdex], icol, irow); irow+=1; nrows+=1
                if nrows==9:
                    icol+=1; irow-=nrows; nrows=1; irow+=1; colJumps+=1

            irow = 10; icol-=colJumps
            # Select/deselect all
            ttk.Label(tabClimdex, text="").grid(sticky="W", column=icol, row=irow, pady=30); irow += 1
            if var == 'tmax':
                Button(tabClimdex, text='Select all', command=select_all_tmax, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
                Button(tabClimdex, text='Deselect all', command=deselect_all_tmax, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
            elif var == 'tmin':
                Button(tabClimdex, text='Select all', command=select_all_tmin, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
                Button(tabClimdex, text='Deselect all', command=deselect_all_tmin, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
            elif var == 'pcp':
                Button(tabClimdex, text='Select all', command=select_all_pcp, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
                Button(tabClimdex, text='Deselect all', command=deselect_all_pcp, width=buttonWidth, takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1

            icol+=3


    def get(self):
        return self.chk_list



########################################################################################################################
class tabModelsAndScenes(ttk.Frame):

    def __init__(self, notebook):
        tabModelsAndScenes = ttk.Frame(notebook)
        notebook.add(tabModelsAndScenes, text='Models and Scenarios')
        self.chk_dict_models = {}

        def add_to_chk_list(name, list, icol, irow, obj=None, affectedBySelectAll=False):
            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            if name in list:
                checked = tk.BooleanVar(value=True)
            if obj == None:
                c = Checkbutton(tabModelsAndScenes, text=name.split('_')[0], variable=checked, takefocus=False)
            else:
                c = Checkbutton(tabModelsAndScenes, text=name.split('_')[0], variable=checked, command=lambda: switch(obj), takefocus=False)
            if affectedBySelectAll == True:
                cbuts.append(c)
            c.grid(sticky="W", column=icol, row=irow, padx=30)
            return {name: checked}

        # Functions for selecting/deselecting all
        cbuts = []
        buttonWidth = 8
        def select_all():
            for i in cbuts:
                i.select()
        def deselect_all():
            for i in cbuts:
                i.deselect()


        irow, icol = 0, 0
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, padx=20, pady=0); icol+=1; irow+=1
        Label(tabModelsAndScenes, text="Select models from the list to include their r1i1p1f1 run:")\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=100);
        Label(tabModelsAndScenes, text="Select scenarios:").grid(sticky="W", column=icol+4, row=irow, padx=100,
                                                          columnspan=100);
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, pady=10); irow+=1


        # Models
        all_models = ('ACCESS-CM2_r1i1p1f1', 'ACCESS-ESM1-5_r1i1p1f1', 'AWI-CM-1-1-MR_r1i1p1f1', 'AWI-CM-1-1-LR_r1i1p1f1', 'BCC-CSM2-MR_r1i1p1f1', 'BCC-ESM1_r1i1p1f1',
                      'CAMS-CSM1-0_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CanESM5-CanOE_r1i1p1f1',
                      'CESM2_r1i1p1f1_r1i1p1f1', 'CESM2-FV2_r1i1p1f1', 'CESM2-WACCM_r1i1p1f1', 'CESM2-WACCM-FV2_r1i1p1f1', 'CIESM_r1i1p1f1',
                      'CMCC-CM2-HR4_r1i1p1f1', 'CMCC-CM2-SR5_r1i1p1f1', 'CMCC-ESM2_r1i1p1f1', 'CNRM-CM6-1_r1i1p1f1', 'CNRM-CM6-1-HR_r1i1p1f1', 'CNRM-ESM2-1_r1i1p1f1',
                      'E3SM-1-0_r1i1p1f1', 'E3SM-1-1_r1i1p1f1', 'E3SM-1-1-ECA_r1i1p1f1',
                      'EC-Earth3_r1i1p1f1', 'EC-Earth3-AerChem_r1i1p1f1', 'EC-Earth3-CC_r1i1p1f1', 'EC-Earth3-Veg_r1i1p1f1',
                      'EC-Earth3-Veg-LR_r1i1p1f1', 'FGOALS-f3-L_r1i1p1f1', 'FGOALS-g3_r1i1p1f1', 'FIO-ESM-2-0_r1i1p1f1',
                      'GFDL-CM4_r1i1p1f1', 'GFDL-ESM4_r1i1p1f1', 'GISS-E2-1-G_r1i1p1f1', 'GISS-E2-1-H_r1i1p1f1', 'HadGEM3-GC31-LL_r1i1p1f1',
                      'HadGEM3-GC31-MM_r1i1p1f1', 'IITM-ESM_r1i1p1f1', 'INM-CM4-8_r1i1p1f1', 'INM-CM5-0_r1i1p1f1', 'IPSL-CM5A2-INCA_r1i1p1f1', 'IPSL-CM6A-LR_r1i1p1f1',
                      'KACE-1-0-G_r1i1p1f1', 'KIOST-ESM_r1i1p1f1', 'MCM-UA-1-0_r1i1p1f1', 'MIROC-ES2H_r1i1p1f1', 'MIROC-ES2L_r1i1p1f1', 'MIROC6_r1i1p1f1', 'MPI-ESM-1-2-HAM_r1i1p1f1',
                      'MPI-ESM1-2-HR_r1i1p1f1', 'MPI-ESM1-2-LR_r1i1p1f1', 'MRI-ESM2-0_r1i1p1f1', 'NESM3_r1i1p1f1', 'NorCPM1_r1i1p1f1', 'NorESM2-LM_r1i1p1f1', 'NorESM2-MM_r1i1p1f1',
                      'SAM0-UNICON_r1i1p1f1', 'TaiESM1_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f1')

        maxRows = 15
        ncols = 0
        nrows = maxRows
        for model in all_models:
            self.chk_dict_models.update(add_to_chk_list(model, model_names_list, icol, irow, affectedBySelectAll=True))
            irow += 1; nrows-=1
            if nrows == 0:
                ncols+=1; nrows = maxRows; icol+=1; irow-=maxRows

        # Select all models
        irow+=1; icol-=ncols-1
        Button(tabModelsAndScenes, text='Select all', command=select_all, takefocus=False).grid(column=icol, row=irow, pady=10); icol += 1
        Button(tabModelsAndScenes, text='Deselect all', command=deselect_all, takefocus=False).grid(column=icol, row=irow)


        # Other models
        irow+=maxRows
        icol -=2

        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, padx=30, pady=5, columnspan=4); irow+=1
        Label(tabModelsAndScenes, text="In order to include other models and/or runs introduce them here "
                                       "separated by ';'")\
                                    .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4); irow+=1
        Label(tabModelsAndScenes, text="Example: ACCESS-CM2_r1i1p1f3; EC-Earth3_r2i1p1f1")\
                                    .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4)
        otherModels_list = []
        for model in model_names_list:
            if model.split('_')[1] != 'r1i1p1f1':
                otherModels_list.append(model)
        otherModels_list = '; '.join((otherModels_list))

        icol += 1
        self.otherModels_var = tk.StringVar()
        self.otherModels_Entry = tk.Entry(tabModelsAndScenes, textvariable=self.otherModels_var, width=45,
                                          justify='left', state='normal', takefocus=False)
        self.otherModels_Entry.insert(END, otherModels_list)
        self.otherModels_Entry.grid(sticky="E", column=icol, row=irow, columnspan=3)
        icol += 1


        # Scenes
        irow, icol = 2, 5
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, padx=30); icol+=1
        all_scenes = ['HISTORICAL', 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
        self.chk_dict_scenes = {}
        for scene in all_scenes:
            self.chk_dict_scenes.update(add_to_chk_list(scene, scene_names_list, icol, irow)); irow += 1

        # Other models
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow)
        self.otherScenes_var = tk.StringVar()
        self.otherScenes_Entry = tk.Entry(tabModelsAndScenes, textvariable=self.otherScenes_var, width=15, justify='right', state='disabled')
        self.otherScenes_Entry.grid(sticky="E", column=icol, row=irow, padx=100)
        CreateToolTip(self.otherScenes_Entry, "Enter scenario names separated by ';'")
        self.chk_dict_scenes.update(add_to_chk_list('Others:', scene_names_list, icol, irow, obj=self.otherScenes_Entry)); irow += 1

    def get(self):
        return self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var



########################################################################################################################
class tabDatesAndDomain(ttk.Frame):

    def __init__(self, notebook):
        tabDatesAndDomain = ttk.Frame(notebook)
        notebook.add(tabDatesAndDomain, text='Dates and Domain')

        # frameDates
        frameDates = Frame(tabDatesAndDomain)
        frameDates.grid(sticky="W", column=0, row=0, padx=60, pady=20)

        # frameSplitMode
        frameSplitMode = Frame(tabDatesAndDomain)
        frameSplitMode.grid(sticky="W", column=0, row=1, padx=60, pady=20)

        # framePeriodFilenames
        framePeriodFilenames = Frame(tabDatesAndDomain)
        framePeriodFilenames.grid(sticky="W", column=1, row=0, padx=60, pady=20)

        # frameDomain
        frameDomain = Frame(tabDatesAndDomain)
        frameDomain.grid(sticky="W", column=1, row=1, padx=60, pady=20)


        icol, irow = 0, 0
        ttk.Label(frameDates, text="Select dates:").grid(sticky="W", column=icol, row=irow, padx=30, pady=10, columnspan=100); irow+=1
        ttk.Label(frameDates, text="").grid(sticky="W", column=icol, row=irow, padx=30); icol += 1

        # Years
        for (text, var, info) in [
            ('Calibration:', calibration_years, 'Longest period available by both reanalysis and hres data, \n'
                                                'which then can be split for training and testing'),
            ('Reference:', reference_years, 'For standardization and as reference climatology. \n'
                                            'The choice of the reference period is constrained by availability of \n'
                                            'reanalysis, historical GCMs and hres data.'),
            ('Historical:', historical_years, 'For historical projections'),
            ('SSPs:', ssp_years, 'For future projections'),
            ('Bias correction:', biasCorr_years, 'For bias correction of projections\n'
                                            'The choice of the bias correction period is constrained by availability of \n'
                                            'historical GCMs and hres data.'),
            ]:

            lab = ttk.Label(frameDates, text=text)
            CreateToolTip(lab, info)
            lab.grid(sticky="E", column=icol, row=irow, padx=8); icol+=1

            firstYear, lastYear = var[0], var[1]
            firstYear_var = tk.StringVar()
            firstYear_Entry = tk.Entry(frameDates, textvariable=firstYear_var, width=6, justify='right', takefocus=False)
            firstYear_Entry.insert(END, firstYear)
            firstYear_Entry.grid(sticky="W", column=icol, row=irow); icol+=1
            Label(frameDates, text='-').grid(sticky="E", column=icol, row=irow, padx=3); icol+=1
            lastYear_var = tk.StringVar()
            lastYear_Entry = tk.Entry(frameDates, textvariable=lastYear_var, width=6, justify='right', takefocus=False)
            lastYear_Entry.insert(END, lastYear)
            lastYear_Entry.grid(sticky="W", column=icol, row=irow)

            # bc_method
            if text == 'Bias correction:':
                icol += 1
                self.bc_method = StringVar()
                bc_methods = {'None': 'Do not perform bias correction over projections', 'QM': 'Quantile Mapping',
                              'DQM': 'Detrended Quantile Mapping', 'QDM': 'Quantile Delta Mapping',
                              'PSDM': '(Parametric) Scaled Distribution Mapping'}
                for bc_meth in bc_methods:
                    c = Radiobutton(frameDates, text=str(bc_meth), variable=self.bc_method, value=bc_meth, takefocus=False)
                    c.grid(sticky="W", column=icol, row=irow);
                    icol += 1
                    CreateToolTip(c, bc_methods[bc_meth])
                    self.bc_method.set(bc_method)
                icol-=4; irow+=1

            irow+=1

            if text == 'Calibration:':
                self.calibration_years = (firstYear_var, lastYear_var)
            elif text == 'Reference:':
                self.reference_years = (firstYear_var, lastYear_var)
            elif text == 'Historical:':
                self.historical_years = (firstYear_var, lastYear_var)
            elif text == 'SSPs:':
                self.ssp_years = (firstYear_var, lastYear_var)
            elif text == 'Bias correction:':
                self.biasCorr_years = (firstYear_var, lastYear_var)
            icol -= 3


        # Train/test split
        self.split_mode = StringVar()
        irow, icol = 0, 0
        Label(frameSplitMode, text='Define how to split the calibration period for training/testing:')\
            .grid(sticky="W", column=icol, row=irow, columnspan=10, padx=30, pady=10); irow += 1; icol+=2
        Label(frameSplitMode, text='Testing years:')\
            .grid(column=icol, row=irow, columnspan=4);
        irow += 1; icol-=1


        def add_splitMode_button_and_years(text, split_modeName, years, info, irow, icol):


            c = Radiobutton(frameSplitMode, text=str(text), variable=self.split_mode, value=split_modeName,
                            command=lambda: switch_splitMode(split_modeName, self.dict_buttons), takefocus=False)

            c.grid(sticky="W", column=icol, row=irow)
            CreateToolTip(c, info)
            self.split_mode.set(split_mode)

            if split_modeName in ('single_split', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5'):
                icol+=2
                firstYear, lastYear = years[0], years[1]
                firstYear_var = tk.StringVar()
                firstYearTesting_Entry = tk.Entry(frameSplitMode, textvariable=firstYear_var, width=6, justify='right', takefocus=False)
                firstYearTesting_Entry.insert(END, firstYear)
                if split_modeName != split_mode:
                    firstYearTesting_Entry.config(state='disabled')
                firstYearTesting_Entry.grid(sticky="E", column=icol, row=irow);
                icol += 1
                Label(frameSplitMode, text='-').grid(column=icol, row=irow);
                icol += 1
                lastYear_var = tk.StringVar()
                lastYearTesting_Entry = tk.Entry(frameSplitMode, textvariable=lastYear_var, width=6, justify='right', takefocus=False)
                lastYearTesting_Entry.insert(END, lastYear)
                if split_modeName != split_mode:
                    lastYearTesting_Entry.config(state='disabled')
                lastYearTesting_Entry.grid(sticky="W", column=icol, row=irow)
                self.testing_years_dict.update({split_modeName: (firstYear_var, lastYear_var)})
                self.dict_buttons.update({split_modeName: [c, firstYearTesting_Entry, lastYearTesting_Entry]})
                icol -= 4
            else:
                self.dict_buttons.update({split_modeName: [c, None, None]})


        self.testing_years_dict = {}

        self.dict_buttons = {}

        for (text, split_modeName, years, info) in [
            ('All training', 'all_training', None, 'The whole calibration period is used for training'),
            ('All testing', 'all_testing', None, 'The whole calibration period is used for testing'),
            ('Single train/test split', 'single_split', single_split_testing_years, 'Single train/test split'),
            ('k-fold 1/5', 'fold1', fold1_testing_years, 'When downscaling k-fold 5/5, the five k-folds will be authomatically joined'),
            ('k-fold 2/5', 'fold2', fold2_testing_years, 'When downscaling k-fold 5/5, the five k-folds will be authomatically joined'),
            ('k-fold 3/5', 'fold3', fold3_testing_years, 'When downscaling k-fold 5/5, the five k-folds will be authomatically joined'),
            ('k-fold 4/5', 'fold4', fold4_testing_years, 'When downscaling k-fold 5/5, the five k-folds will be authomatically joined'),
            ('k-fold 5/5', 'fold5', fold5_testing_years, 'When downscaling k-fold 5/5, the five k-folds will be authomatically joined'),
            ]:

            add_splitMode_button_and_years(text, split_modeName, years, info, irow, icol)

            irow+=1


        icol = 0
        irow = 0
        Label(framePeriodFilenames, text='Define the following fields:').grid(sticky="W", column=icol, row=irow, padx=10, pady=10, columnspan=10); irow += 1

        entriesW = 17

        # hresPeriodFilename_t
        self.hresPeriodFilename_var_t = tk.StringVar()
        Label(framePeriodFilenames, text='Hres temperature period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        hresPeriodFilename_Entry_t = tk.Entry(framePeriodFilenames, textvariable=self.hresPeriodFilename_var_t, width=entriesW, justify='right', takefocus=False)
        hresPeriodFilename_Entry_t.insert(END, hresPeriodFilename['t'])
        hresPeriodFilename_Entry_t.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

        # hresPeriodFilename_p
        self.hresPeriodFilename_var_p = tk.StringVar()
        Label(framePeriodFilenames, text='Hres precipitation period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        hresPeriodFilename_Entry_p = tk.Entry(framePeriodFilenames, textvariable=self.hresPeriodFilename_var_p, width=entriesW, justify='right', takefocus=False)
        hresPeriodFilename_Entry_p.insert(END, hresPeriodFilename['p'])
        hresPeriodFilename_Entry_p.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

        # reanalysisName
        self.reanalysisName_var = tk.StringVar()
        Label(framePeriodFilenames, text='Reanalysis name:').grid(sticky="W", column=icol, row=irow, padx=10); icol += 1
        reanalysisName_Entry = tk.Entry(framePeriodFilenames, textvariable=self.reanalysisName_var, width=entriesW, justify='right', takefocus=False)
        reanalysisName_Entry.insert(END, reanalysisName)
        reanalysisName_Entry.grid(sticky="W", column=icol, row=irow); icol -= 1; irow+=1

        # reanalysisPeriodFilename
        self.reanalysisPeriodFilename_var = tk.StringVar()
        Label(framePeriodFilenames, text='Reanalysis period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        reanalysisPeriodFilename_Entry = tk.Entry(framePeriodFilenames, textvariable=self.reanalysisPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        reanalysisPeriodFilename_Entry.insert(END, reanalysisPeriodFilename)
        reanalysisPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

        # historicalPeriodFilename
        self.historicalPeriodFilename_var = tk.StringVar()
        Label(framePeriodFilenames, text='Historical period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        historicalPeriodFilename_Entry = tk.Entry(framePeriodFilenames, textvariable=self.historicalPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        historicalPeriodFilename_Entry.insert(END, historicalPeriodFilename)
        historicalPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

        # sspPeriodFilename
        self.sspPeriodFilename_var = tk.StringVar()
        Label(framePeriodFilenames, text='SSP period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        sspPeriodFilename_Entry = tk.Entry(framePeriodFilenames, textvariable=self.sspPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        sspPeriodFilename_Entry.insert(END, sspPeriodFilename)
        sspPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

        icol, irow = 0, 0
        ttk.Label(frameDomain, text="Define the following spatial information:").grid(sticky="W", column=icol, row=irow, padx=10, pady=30, columnspan=100); irow+=1

        # grid_res
        self.grid_res_var = tk.StringVar()
        lab = Label(frameDomain, text="Grid resolution:")
        lab.grid(sticky="W", column=icol, row=irow, padx=10, columnspan=5); icol+=2
        grid_resTesting_Entry = tk.Entry(frameDomain, textvariable=self.grid_res_var, width=4, justify='right', takefocus=False)
        grid_resTesting_Entry.insert(END, grid_res)
        CreateToolTip(lab, 'Grid resolution')
        grid_resTesting_Entry.grid(sticky="W", column=icol+1, row=irow)
        irow+=1; icol-=2

        # safGrid
        ttk.Label(frameDomain, text="").grid(sticky="W", column=icol, row=irow, padx=10, pady=2, columnspan=100); irow+=1
        padx, pady, width = 2, 2, 5
        lab = Label(frameDomain, text='Domain for synoptic analogy fields \n'
                                     '(lat up, lat down, lon left and long right):', justify=LEFT)
        lab.grid(sticky="W", column=icol, row=irow, padx=10, pady=2, columnspan=20); irow+=1

        ttk.Label(frameDomain, text="").grid(sticky="W", column=icol, row=irow, padx=10, pady=10, columnspan=100); icol+=1


        self.saf_lat_up_var = tk.StringVar()
        saf_lat_upTesting_Entry = tk.Entry(frameDomain, textvariable=self.saf_lat_up_var, width=width, justify='right', takefocus=False)
        saf_lat_upTesting_Entry.insert(END, saf_lat_up)
        CreateToolTip(saf_lat_upTesting_Entry, 'lat up')
        saf_lat_upTesting_Entry.grid(sticky="W", column=icol+1, row=irow, padx=padx, pady=pady)
        self.saf_lon_left_var = tk.StringVar()
        saf_lon_leftTesting_Entry = tk.Entry(frameDomain, textvariable=self.saf_lon_left_var, width=width, justify='right', takefocus=False)
        saf_lon_leftTesting_Entry.insert(END, saf_lon_left)
        CreateToolTip(saf_lon_leftTesting_Entry, 'lon left')
        saf_lon_leftTesting_Entry.grid(sticky="W", column=icol, row=irow+1, padx=padx, pady=pady)
        self.saf_lon_right_var = tk.StringVar()
        saf_lon_rightTesting_Entry = tk.Entry(frameDomain, textvariable=self.saf_lon_right_var, width=width, justify='right', takefocus=False)
        saf_lon_rightTesting_Entry.insert(END, saf_lon_right)
        CreateToolTip(saf_lon_rightTesting_Entry, 'lon right')
        saf_lon_rightTesting_Entry.grid(sticky="W", column=icol+2, row=irow+1, padx=padx, pady=pady)
        self.saf_lat_down_var = tk.StringVar()
        saf_lat_downTesting_Entry = tk.Entry(frameDomain, textvariable=self.saf_lat_down_var, width=width, justify='right', takefocus=False)
        saf_lat_downTesting_Entry.insert(END, saf_lat_down)
        CreateToolTip(saf_lat_downTesting_Entry, 'lat down')
        saf_lat_downTesting_Entry.grid(sticky="W", column=icol+1, row=irow+2, padx=padx, pady=pady); irow+=3
        lab = Label(frameDomain, text='Make sure all files (reanalysis and models) contain, at least, \n'
                                     'this domain plus a border of one grid box width.', justify=LEFT)
        lab.grid(sticky="W", column=icol, row=irow, padx=10, pady=2, columnspan=20); irow+=1


    def get(self):
        return self.calibration_years, self.reference_years, self.historical_years, self.ssp_years, self.biasCorr_years, \
               self.bc_method, self.testing_years_dict, self.hresPeriodFilename_var_t, self.hresPeriodFilename_var_p, \
               self.reanalysisName_var, self.reanalysisPeriodFilename_var, self.historicalPeriodFilename_var, \
               self.sspPeriodFilename_var, self.split_mode, self.grid_res_var, self.saf_lat_up_var, \
               self.saf_lon_left_var, self.saf_lon_right_var, self.saf_lat_down_var


########################################################################################################################
class tabVisualization(ttk.Frame):


    def __init__(self, notebook):
        tabVisualization = ttk.Frame(notebook)
        notebook.add(tabVisualization, text='Visualization')

        def open_figure(imgs):

            w = 600
            filename = '_'.join((self.fields[0], self.fields[1], self.fields[2], self.fields[3], self.fields[4],
                                 self.fields[5])) + '.png'
            try:
                text = '\n' + self.descriptions['_'.join((self.fields[0], self.fields[1]))] + '\n'
            except:
                text = ''

            if os.path.isfile("../results/Figures/" + filename):

                imgs.append(Image.open("../results/Figures/" + filename))
                h = int(w * imgs[-1].height / imgs[-1].width)
                imgs[-1] = imgs[-1].resize((w, h), Image.ANTIALIAS)
                imgs[-1] = ImageTk.PhotoImage(imgs[-1])

                rootIm = tk.Toplevel()
                rootIm.title(filename.replace('.png', '').replace('_', ' '))

                canvas = Canvas(rootIm, width=w, height=h)
                canvas.create_image(0, 0, anchor=NW, image=imgs[-1])
                canvas.grid(column=0, row=0, padx=0, pady=0)
                l = Label(rootIm, text=text, borderwidth=0, background=None, wraplength=w)
                l.grid(column=0, row=1)

                rootIm.resizable(width=False, height=False)

            else:
                tk.messagebox.showerror("pyClim-SDM",  "No figure has been generated matching the selection:\n" + filename)


        # frameFigSelection
        frameFigSelection = Frame(tabVisualization, height=510, width=1140)
        frameFigSelection.grid(sticky="W", column=0, row=0, padx=0, pady=0)
        frameFigSelection.grid_propagate(False)

        irow, icol = 0, 0

        Label(frameFigSelection, text='').grid(sticky="W", column=icol, row=5, padx=10, pady=10); irow+=1
        Label(frameFigSelection, text='Make your selection in order to visualize existing figures:')\
            .grid(sticky="W", column=icol, row=irow, padx=30, pady=30, columnspan=100); irow+=1

        Label(frameFigSelection, text="").grid(sticky="W", column=icol, row=irow, padx=10, pady=10); irow+=1; icol+=1

        Label(frameFigSelection, text="Select experiment:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol+=1
        Label(frameFigSelection, text="Select figure type:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol+=1
        Label(frameFigSelection, text="Select variable:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol+=1
        Label(frameFigSelection, text="Select climdex/predictor:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol+=1
        Label(frameFigSelection, text="Select method/model/scene:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol+=1
        Label(frameFigSelection, text="Select season:").grid(sticky="W", column=icol, row=irow, padx=0, pady=10); icol-=5; irow+=1

        self.fields = 6 * ['None']
        self.last_defined_field = 0
        self.l = Label(frameFigSelection, text='', anchor="e", justify=LEFT, wraplength=600)


        def clear_comboboxes_from(icol, first_time=False):
            """delete fields and clear combobox"""
            ncols = 6
            irow = 4
            if self.last_defined_field > icol or first_time == True:
                for i in range(icol, ncols):
                    self.fields[i] = 'None'
                    combobox = ttk.Combobox(frameFigSelection, state='disabled')
                    combobox.grid(sticky="W", column=i+1, row=irow, padx=2 , pady=0)
            self.last_defined_field = icol

        # Clear combobox
        clear_comboboxes_from(0, first_time=True)

        self.descriptions = {
            'PRECONTROL_correlationMap': 'Correlation of the temporal daily series between one predictor and one '
                                         'predictand (Pearson coefficient for tmax/tmin and Spearman for pcp).',
            'PRECONTROL_correlationBoxplot': 'Correlation of the temporal daily series between all predictors and one '
                                             'predictand (Pearson coefficient for tmax/tmin and Spearman for pcp). Each '
                                             'box contains one value per grid point.',
            'PRECONTROL_nansMap': 'Map with percentage of missing data for one predictor, model and scene.',
            'PRECONTROL_nansMatrix': 'Percentage of missing data (spatially averaged) for one scene (all predictors and '
                                     'models).',
            'PRECONTROL_biasBoxplot': 'Bias of all models compared to the reanalysis (in the mean value) in a historical '
                                      'period. For tmax/tmin absolute bias, for pcp relative bias and the rest '
                                      'standardized and absolute bias. Each box contains one value per grid point.',
            'PRECONTROL_evolSpaghetti': 'Evolution of one predictor by all models in the form of anomaly with respect to the '
                                        'reference period (absolute anomaly for tmax/tmin, relative anomaly for pcp and '
                                        'absolute anomaly of the standardized variables for the rest).',
            'PRECONTROL_qqPlot': 'QQ-plot for one variable by one model in historical vs. reanalysis.',
            'PRECONTROL_annualCycle': 'Annual cycle for one variable by all models in historical and reanalysis '
                                      '(monthly means for tmax/tmin, monthly accumulations for pcp and monthly means of '
                                      'the standardized variable for the rest). ',
            'PRECONTROL_evolTube': 'Evolution graph for one variable by multimodel ensemble (the central line represents'
                                   ' 50th percentile and the shaded area represents IQR), in the form of anomaly with '
                                   'respect to the reference period (absolute anomaly for tmax/tmin, relative anomaly '
                                   'for pcp and absolute anomaly of the standardized variables for the rest).',
            'EVALUATION_annualCycle': 'Annual cycle for one variable, downscaled by all methods vs. observation '
                                      '(monthly means for tmax/tmin and monthly accumulations for pcp).',
            'EVALUATION_correlationBoxplot': 'Correlation (Pearson for temperature and Spearman for precipitation) of '
                                             'the daily series (downscaled and observed) by all methods. Boxes contain '
                                             'one value per grid point.',
            'EVALUATION_varianceBoxplot': 'Bias (relative, %) in the variance of the daily series (downscaled and '
                                          'observed) by all methods. Boxes contain one value per grid point.',
            'EVALUATION_qqPlot': 'QQ-plot for one variable by one method vs. observations.',
            'EVALUATION_r2Map': 'R2 score of the daily series (coefficient of determination) by one method.',
            'EVALUATION_accuracyMap': 'AAccuracy score for the daily series (only for wet/dry classification. '
                                      'Acc=corrects/total) by one method.',
            'EVALUATION_correlationMapMonthly': 'Correlation for the monthly (mean for tmax/tmin and accumulated for '
                                                'pcp) series by one method with observations.  Pearson coefficient for '
                                                'tmax/tmin and Spearman for pcp.',
            'EVALUATION_r2MapMonthly': 'R2 score (coefficient of determination)  for the monthly (mean for tmax/tmin '
                                       'and accumulated for pcp) series by one method with observations. ',
            'EVALUATION_biasClimdexBoxplot': 'Bias (absolute/relative) for the mean climdex in the whole testing period '
                                             'by all methods. Boxes contain one value per grid point.',
            'EVALUATION_obsMap': 'Mean observed values in the whole period.',
            'EVALUATION_estMap': 'Mean estimated (downscaled) values in the whole period by one method.',
            'EVALUATION_biasMap': 'Bias (absolute/relative) in the whole period by one method.',
            'EVALUATION_scatterPlot': 'Downscaled vs. observed climdex in the whole period  each scatter point '
                                      'corresponds to a grid point.',
            'PROJECTIONS_evolSpaghetti': 'Evolution of one variable by all models in the form of anomaly with respect '
                                         'to the reference period (absolute anomaly for tmax/tmin and relative anomaly '
                                         'for pcp).',
            'PROJECTIONS_evolTube': 'Evolution graph of one variable by the multimodel ensemble (the central line '
                                    'represents 50th percentile and the shaded area represents IQR), in the form of '
                                    'anomaly with respect to the reference period (absolute anomaly for tmax/tmin and '
                                    'relative anomaly for pcp).',
            'PROJECTIONS_meanChangeMap': 'Anomaly in a future period with respect to a reference period given by the '
                                         'multimodel ensemble mean (mean change). Absolute anomaly for tmax/tmin and '
                                         'relative anomaly for pcp.',
            'PROJECTIONS_stdChangeMap': 'Standard deviation in the anomaly given by the multimodel ensemble (spread).',
            'PROJECTIONS_evolTrendRaw': 'Evolution graph, by one method vs. raw models, of one variable by the '
                                        'multimodel ensemble (the central line represents 50th percentile and the '
                                        'shaded area represents IQR), in the form of anomaly with respect to the '
                                        'reference period (absolute anomaly for tmax/tmin and relative anomaly for pcp).',
        }

        def callback_experiment(event):
            clear_comboboxes_from(1)
            self.fields[0] = self.experimentVar.get()

            def callback_figType(event):
                clear_comboboxes_from(2)
                self.fields[1] = self.figTypeVar.get()

                # Create label with description
                Label(frameFigSelection, text='').grid(sticky="W", column=1, row=5, padx=10, pady=20)
                text = 'Your current selection corresponds to: \n\n' + self.descriptions['_'.join((self.fields[0], self.fields[1]))]
                self.l.destroy()
                self.l = Label(frameFigSelection, text=text, anchor="e", justify=LEFT, wraplength=600)
                self.l.grid(sticky="W", column=2, row=6, padx=10, pady=10, columnspan=100)

                def callback_var(event):
                    clear_comboboxes_from(3)
                    self.fields[2] = self.varVar.get()

                    def callback_climdex_pred(event):
                        clear_comboboxes_from(4)
                        self.fields[3] = self.climdex_predVar.get()

                        def callback_method_model_scene(event):
                            clear_comboboxes_from(5)
                            self.fields[4] = self.method_model_sceneVar.get()

                            def callback_season(event):
                                self.fields[5] = self.seasonVar.get()

                            # season
                            seasons = []
                            for file in os.listdir('../results/Figures/'):
                                if file.endswith(".png") and \
                                        file.split('_')[0] == self.fields[0] \
                                        and file.split('_')[1] == self.fields[1] \
                                        and file.split('_')[2] == self.fields[2] \
                                        and file.split('_')[3] == self.fields[3] \
                                        and file.split('_')[4] == self.fields[4] \
                                        and file.split('_')[5].replace('.png', '') not in seasons:
                                    seasons.append(file.split('_')[5].replace('.png', ''))

                            # sort seasons
                            ordered_seasons = ['ANNUAL', 'DJF', 'MAM', 'JJA', 'SON', 'None']
                            aux = ['ANNUAL', 'DJF', 'MAM', 'JJA', 'SON', 'None']
                            for sea in ordered_seasons:
                                if sea not in seasons:
                                    aux.remove(sea)
                            seasons = aux
                            self.seasonVar = tk.StringVar()
                            combobox = ttk.Combobox(frameFigSelection, textvariable=self.seasonVar)
                            combobox['values'] = seasons
                            combobox['state'] = 'readonly'
                            combobox.grid(sticky="W", column=6, row=4, padx=2, pady=0)
                            combobox.bind('<<ComboboxSelected>>', callback_season)
                            self.fields[5] = self.seasonVar.get()

                        # method_model_scene
                        method_model_scenes = []
                        for file in os.listdir('../results/Figures/'):
                            if file.endswith(".png") and \
                                    file.split('_')[0] == self.fields[0] \
                                    and file.split('_')[1] == self.fields[1] \
                                    and file.split('_')[2] == self.fields[2] \
                                    and file.split('_')[3] == self.fields[3] \
                                    and file.split('_')[4] not in method_model_scenes:
                                method_model_scenes.append(file.split('_')[4])

                        self.method_model_sceneVar = tk.StringVar()
                        combobox = ttk.Combobox(frameFigSelection, textvariable=self.method_model_sceneVar)
                        combobox['values'] = method_model_scenes
                        combobox['state'] = 'readonly'
                        combobox.grid(sticky="W", column=5, row=4, padx=2, pady=0)
                        combobox.bind('<<ComboboxSelected>>', callback_method_model_scene)
                        self.fields[4] = self.method_model_sceneVar.get()


                    # climdex_pred
                    climdex_preds = []
                    for file in os.listdir('../results/Figures/'):
                        if file.endswith(".png") and file.split('_')[0] == self.fields[0] and \
                                file.split('_')[1] == self.fields[1] and \
                                file.split('_')[2] == self.fields[2] and file.split('_')[3] not in climdex_preds:
                            climdex_preds.append(file.split('_')[3])

                    self.climdex_predVar = tk.StringVar()
                    combobox = ttk.Combobox(frameFigSelection, textvariable=self.climdex_predVar)
                    combobox['values'] = climdex_preds
                    combobox['state'] = 'readonly'
                    combobox.grid(sticky="W", column=4, row=4, padx=2, pady=0)
                    combobox.bind('<<ComboboxSelected>>', callback_climdex_pred)
                    self.fields[3] = self.climdex_predVar.get()

                # var
                vars = []
                for file in os.listdir('../results/Figures/'):
                    if file.endswith(".png") and file.split('_')[0] == self.fields[0] and \
                            file.split('_')[1] == self.fields[1] and file.split('_')[2] not in vars:
                        vars.append(file.split('_')[2])

                self.varVar = tk.StringVar()
                combobox = ttk.Combobox(frameFigSelection, textvariable=self.varVar)
                combobox['values'] = vars
                combobox['state'] = 'readonly'
                combobox.grid(sticky="W", column=3, row=4, padx=2, pady=0)
                combobox.bind('<<ComboboxSelected>>', callback_var)
                self.fields[2] = self.varVar.get()

            # figType
            figTypes = []
            for file in os.listdir('../results/Figures/'):
                if file.endswith(".png") and file.split('_')[0] == self.fields[0] and file.split('_')[1] not in figTypes:
                    figTypes.append(file.split('_')[1])

            self.figTypeVar = tk.StringVar()
            combobox = ttk.Combobox(frameFigSelection, textvariable=self.figTypeVar)
            combobox['values'] = figTypes
            combobox['state'] = 'readonly'
            combobox.grid(sticky="W", column=2, row=4, padx=2, pady=0)
            combobox.bind('<<ComboboxSelected>>', callback_figType)
            self.fields[1] = self.figTypeVar.get()


        # experiment
        experiments = []
        for file in os.listdir('../results/Figures/'):
            if file.endswith(".png") and file.split('_')[0] not in experiments:
                experiments.append(file.split('_')[0])

        # sort experiments
        ordered_experiments = ['PRECONTROL', 'EVALUATION', 'PROJECTIONS']
        for exp in ordered_experiments:
            if exp not in experiments:
                ordered_experiments.remove(exp)
        experiments = ordered_experiments

        self.experimentVar = tk.StringVar()
        combobox = ttk.Combobox(frameFigSelection, textvariable=self.experimentVar)
        combobox['values'] = experiments
        combobox['state'] = 'readonly'
        combobox.grid(sticky="W", column=1, row=4, padx=2, pady=10)
        combobox.bind('<<ComboboxSelected>>', callback_experiment)
        self.fields[0] = self.experimentVar.get()



        global imgs
        imgs = []
        Button(tabVisualization, text="Open figure", width=10,
               command=lambda: open_figure(imgs), takefocus=False).grid(sticky="SE", column=1, row=1)




########################################################################################################################
class selectionWindow():

    def __init__(self):

        # Welcome message
        run, self.showWelcomeMessage = welcomeMessage().get()
        if run == False:
            exit()

        # Root menu
        root = tk.Tk()
        root.title("pyClim-SDM")
        rootW, rootH = 1280, 620
        root.minsize(rootW, rootH )
        root.maxsize(rootW, rootH )

        # Notebook (frame for tabs)
        notebook = ttk.Notebook(root, width=rootW, height=rootH)
        notebook.pack(expand=1, fill="both")


        # Tab: run
        self.experiment_chk, self.steps_dict, self.all_steps = tabSteps(notebook, root).get()

        # Tab: methods
        self.methods_chk = tabMethods(notebook).get()

        # Tab: predictors
        self.reaNames_chk, self.modNames_chk, self.preds = tabPredictors(notebook).get()

        # Tab: climdex
        self.climdex_dict_chk = tabClimdex(notebook).get()

        # Tab: models
        self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var = \
            tabModelsAndScenes(notebook).get()

        # Tab: dates and Domain
        self.calibration_years_chk, self.reference_years_chk, self.historical_years_chk, self.ssp_years_chk, self.biasCorr_years_chk, \
            self.bc_method_chk, self.testing_years_dict_chk, self.hresPeriodFilename_var_t_chk, self.hresPeriodFilename_var_p_chk, \
            self.reanalysisName_var_chk, self.reanalysisPeriodFilename_var_chk, self.historicalPeriodFilename_var_chk, \
            self.sspPeriodFilename_var_chk, self.split_mode_chk, self.grid_res_var_chk, self.saf_lat_up_var_chk, self.saf_lon_left_var_chk, \
            self.saf_lon_right_var_chk, self.saf_lat_down_var_chk = tabDatesAndDomain(notebook).get()

        # Tab: visualization
        tabVisualization(notebook)

        # Logo
        w = 120
        img = Image.open("../doc/pyClim-SDM_logo.png")
        h = int(w * img.height / img.width)
        img = img.resize((w, h), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        canvas = Canvas(notebook, width=w, height=h)
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.grid(sticky="W", column=0, row=1, padx=10)

        self.run = False
        def run():

            self.all_checks_ok = False

            # Read experiment and steps
            self.exp = self.experiment_chk.get()
            self.steps = []
            for step in self.all_steps[self.exp]:
                if self.steps_dict[self.exp][step].get() == True:
                    self.steps.append(step)

            # Read predictors and saf
            self.selected_all_preds = []
            for x in self.preds:
                if self.preds[x].get() == True and x.split('_')[0] not in self.selected_all_preds:
                    self.selected_all_preds.append(x.split('_')[0])
            self.target_vars0 = [x for x in self.selected_all_preds if x != 'saf']

            # Read methods
            self.target_vars = []
            for meth in self.methods_chk:
                if meth['checked'].get() == True and meth['var'] not in self.target_vars:
                    self.target_vars.append(meth['var'])

            # Force to select at least one saf
            if 'saf' not in self.selected_all_preds:
                self.all_checks_ok = False
                tk.messagebox.showerror("pyClim-SDM",  "At least one Synoptic Analogy Field must be selected (Predictors tab)")
            else:
                self.all_checks_ok = True

            # Force to select at least one pred
            if self.all_checks_ok == True:
                if len(self.target_vars0) == 0:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "At least one predictor (for temperature and/or precipitation) must be selected (Predictors tab)")
                else:
                    self.all_checks_ok = True

            # Force consistency between target_vars and target_vars0
            for var in ('tmax', 'tmin', 'pcp',):
                if self.all_checks_ok == True:
                    if (var in self.target_vars) and (var[0] not in self.target_vars0) and (self.exp != 'PRECONTROL'):
                        self.all_checks_ok = False
                        tk.messagebox.showerror("pyClim-SDM",  'Your selection includes some methods for ' + var + ' but no predictor has been selected')
                    else:
                        self.all_checks_ok = True

            # Force consistency between methods and experiment
            if self.all_checks_ok == True:
                if len(self.target_vars) == 0 and self.exp != 'PRECONTROL':
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  'For ' + self.exp + ' experiment, at least one method must be selected')
                else:
                    self.all_checks_ok = True

            # Force at least one step
            if self.all_checks_ok == True:
                if len(self.steps) == 0:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "At least one step must be selected (Experiments and Steps tab)")
                else:
                    self.all_checks_ok = True

            # Years
            self.aux_calibration_years = (int(self.calibration_years_chk[0].get()), int(self.calibration_years_chk[1].get()))
            self.aux_reference_years = (int(self.reference_years_chk[0].get()), int(self.reference_years_chk[1].get()))
            self.aux_historical_years = (int(self.historical_years_chk[0].get()), int(self.historical_years_chk[1].get()))
            self.aux_ssp_years = (int(self.ssp_years_chk[0].get()), int(self.ssp_years_chk[1].get()))
            self.aux_biasCorr_years = (int(self.biasCorr_years_chk[0].get()), int(self.biasCorr_years_chk[1].get()))
            self.all_years_hres = [x for x in range(max(int(self.hresPeriodFilename_var_t_chk.get().split('-')[0][:4]),
                                                        int(self.hresPeriodFilename_var_p_chk.get().split('-')[0][:4])),
                                                    min(int(self.hresPeriodFilename_var_t_chk.get().split('-')[1][:4]),
                                                        int(self.hresPeriodFilename_var_p_chk.get().split('-')[1][:4]))
                                                        + 1)]
            self.all_years_reanalysis = [x for x in range(int(self.reanalysisPeriodFilename_var_chk.get().split('-')[0][:4]),
                                                    int(self.reanalysisPeriodFilename_var_chk.get().split('-')[1][:4])+1)]
            self.all_years_historical = [x for x in range(int(self.historicalPeriodFilename_var_chk.get().split('-')[0][:4]),
                                                    int(self.historicalPeriodFilename_var_chk.get().split('-')[1][:4])+1)]
            self.all_years_ssp = [x for x in range(int(self.sspPeriodFilename_var_chk.get().split('-')[0][:4]),
                                                    int(self.sspPeriodFilename_var_chk.get().split('-')[1][:4])+1)]

            # Force calibration years
            if self.all_checks_ok == True:
                year = self.aux_calibration_years[0]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Calibration years not available by reanalysis or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
            if self.all_checks_ok == True:
                year = self.aux_calibration_years[1]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Calibration years not available by reanalysis or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force reference years
            if self.all_checks_ok == True:
                year = self.aux_reference_years[0]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Reference years not available by reanalysis, historical GCMs or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
            if self.all_checks_ok == True:
                year = self.aux_reference_years[1]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Reference years not available by reanalysis, historical GCMs or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force historical years
            if self.all_checks_ok == True:
                year = self.aux_historical_years[0]
                if year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Historical years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
                year = self.aux_historical_years[1]
                if year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Historical years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force ssp years
            if self.all_checks_ok == True:
                year = self.aux_ssp_years[0]
                if year not in self.all_years_ssp:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM", "SSP years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
                year = self.aux_ssp_years[1]
                if year not in self.all_years_ssp:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM", "SSP years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force bias correction years
            if self.all_checks_ok == True:
                year = self.aux_biasCorr_years[0]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM", "Bias correction years not available by historical GCMs or hres data.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
                year = self.aux_biasCorr_years[1]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    tk.messagebox.showerror("pyClim-SDM",  "Bias correction years not available by historical GCMs or hres data.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force all checks ok
            if self.all_checks_ok == True:
                self.run = True
                # root.destroy()

                # Experiment
                self.experiment = self.experiment_chk.get()

                # Steps
                self.steps = []
                for step in self.all_steps[self.experiment]:
                    if self.steps_dict[self.experiment][step].get() == True:
                        self.steps.append(step)

                # Methods
                self.methods = []
                for meth in self.methods_chk:
                    if meth['checked'].get() == True:
                        aux = {}
                        for key in meth:
                            if key != 'checked':
                                aux.update({key: meth[key]})
                        self.methods.append(aux)

                # reaNames and modNames
                self.reaNames = {}
                self.modNames = {}
                for var in self.reaNames_chk:
                    self.reaNames.update({var: self.reaNames_chk[var].get()})
                    self.modNames.update({var: self.modNames_chk[var].get()})

                # Predictors
                self.preds_t_list = []
                self.preds_p_list = []
                self.saf_list = []
                for predLong in self.preds:
                    block = predLong.split('_')[0]
                    pred = predLong.split('_')[1]
                    if self.preds[predLong].get() == True:
                        if block == 't':
                            self.preds_t_list.append(pred)
                        elif block == 'p':
                            self.preds_p_list.append(pred)
                        elif block == 'saf':
                            self.saf_list.append(pred)

                # Climdex
                self.climdex_names = {}
                for var in ('tmax', 'tmin', 'pcp'):
                    tmp_climdex_list = []
                    for climdex_dict in self.climdex_dict_chk:
                        if climdex_dict['checked'].get() == True and climdex_dict['var'] == var:
                            tmp_climdex_list.append(climdex_dict['climdex'])
                    self.climdex_names.update({var: tmp_climdex_list})
                    del tmp_climdex_list

                # Years
                self.calibration_years = (self.calibration_years_chk[0].get(), self.calibration_years_chk[1].get())
                self.reference_years = (self.reference_years_chk[0].get(), self.reference_years_chk[1].get())
                self.historical_years = (self.historical_years_chk[0].get(), self.historical_years_chk[1].get())
                self.ssp_years = (self.ssp_years_chk[0].get(), self.ssp_years_chk[1].get())
                self.biasCorr_years = (self.biasCorr_years_chk[0].get(), self.biasCorr_years_chk[1].get())

            # bc_method
            self.bc_method = self.bc_method_chk.get()
            if self.bc_method == 'None':
                self.bc_method = None

            # split_mode and testing years
            self.split_mode = self.split_mode_chk.get()
            self.single_split_testing_years = (
                self.testing_years_dict_chk['single_split'][0].get(), self.testing_years_dict_chk['single_split'][1].get())
            self.fold1_testing_years = (
                self.testing_years_dict_chk['fold1'][0].get(), self.testing_years_dict_chk['fold1'][1].get())
            self.fold2_testing_years = (
                self.testing_years_dict_chk['fold2'][0].get(), self.testing_years_dict_chk['fold2'][1].get())
            self.fold3_testing_years = (
                self.testing_years_dict_chk['fold3'][0].get(), self.testing_years_dict_chk['fold3'][1].get())
            self.fold4_testing_years = (
                self.testing_years_dict_chk['fold4'][0].get(), self.testing_years_dict_chk['fold4'][1].get())
            self.fold5_testing_years = (
                self.testing_years_dict_chk['fold5'][0].get(), self.testing_years_dict_chk['fold5'][1].get())

            # period filenames
            self.hresPeriodFilename_t = self.hresPeriodFilename_var_t_chk.get()
            self.hresPeriodFilename_p = self.hresPeriodFilename_var_p_chk.get()
            self.reanalysisName = self.reanalysisName_var_chk.get()
            self.reanalysisPeriodFilename = self.reanalysisPeriodFilename_var_chk.get()
            self.historicalPeriodFilename = self.historicalPeriodFilename_var_chk.get()
            self.sspPeriodFilename = self.sspPeriodFilename_var_chk.get()

            # grid_res
            self.grid_res = self.grid_res_var_chk.get()
            self.saf_lat_up = self.saf_lat_up_var_chk.get()
            self.saf_lon_left = self.saf_lon_left_var_chk.get()
            self.saf_lon_right = self.saf_lon_right_var_chk.get()
            self.saf_lat_down = self.saf_lat_down_var_chk.get()

            # Models
            self.model_names_list = []
            for model in self.chk_dict_models:
                if self.chk_dict_models[model].get() == True:
                    self.model_names_list.append(model)
            otherModels = self.otherModels_var.get()
            if otherModels != '':
                while ' ' in otherModels:
                    otherModels = otherModels.replace(' ', '')
                for model in otherModels.split(';'):
                    self.model_names_list.append(model)

            # Scenes
            self.scene_names_list = []
            for scene in self.chk_dict_scenes:
                if self.chk_dict_scenes[scene].get() == True:
                    self.scene_names_list.append(scene)
            if 'Others:' in self.scene_names_list:
                self.scene_names_list.remove('Others:')
                otherScenes = self.otherScenes_var.get()
                while ' ' in otherScenes:
                    otherScenes = otherScenes.replace(' ', '')
                for scene in otherScenes.split(';'):
                    self.scene_names_list.append(scene)

            # Write settings file
            write_settings_file(self.showWelcomeMessage, self.experiment, self.steps, self.methods,
                                self.reaNames, self.modNames, self.preds_t_list, self.preds_p_list, self.saf_list,
                                self.climdex_names, self.calibration_years, self.reference_years, self.historical_years,
                                self.ssp_years, self.biasCorr_years, self.bc_method, self.single_split_testing_years,
                                self.fold1_testing_years, self.fold2_testing_years, self.fold3_testing_years,
                                self.fold4_testing_years, self.fold5_testing_years, self.hresPeriodFilename_t,
                                self.hresPeriodFilename_p, self.reanalysisName, self.reanalysisPeriodFilename,
                                self.historicalPeriodFilename, self.sspPeriodFilename, self.split_mode,
                                self.grid_res, self.saf_lat_up, self.saf_lon_left, self.saf_lon_right,
                                self.saf_lat_down, self.model_names_list, self.scene_names_list)

            # Write tmp_main file
            write_tmpMain_file(self.steps)

            # Run .tmp_main
            # import platform
            # if platform.system() == 'Linux':
            subprocess.call(['xterm', '-e', 'python .tmp_main.py'])

            # Delete tmp_main
            os.remove('.tmp_main.py')

        # Run butnon
        frame = Frame(notebook)
        frame.grid(sticky="SE", column=0, row=0, padx=560, pady=272)
        Button(notebook, text="Run", width=10, command=run).grid(sticky="W", column=2, row=1, padx=20, pady=0)

        # Mainloop
        root.mainloop()



########################################################################################################################
def write_settings_file(showWelcomeMessage, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list,
                        saf_list, climdex_names, calibration_years, reference_years, historical_years, ssp_years, biasCorr_years,
                        bc_method, single_split_testing_years, fold1_testing_years, fold2_testing_years,
                        fold3_testing_years, fold4_testing_years, fold5_testing_years, hresPeriodFilename_t, hresPeriodFilename_p,
                        reanalysisName, reanalysisPeriodFilename, historicalPeriodFilename,
                        sspPeriodFilename, split_mode, grid_res, saf_lat_up, saf_lon_left, saf_lon_right,
                        saf_lat_down, model_names_list, scene_names_list):

    """This function prepares a new settings file with the user selected options"""

    # Open f for writing
    f = open('../config/settings.py', "w")

    # Write new settings
    f.write("showWelcomeMessage = " + str(showWelcomeMessage) + "\n")
    f.write("experiment = '" + str(experiment) + "'\n")
    f.write("methods = " + str(methods) + "\n")
    f.write("reaNames = " + str(reaNames) + "\n")
    f.write("modNames = " + str(modNames) + "\n")
    f.write("preds_t_list = " + str(preds_t_list) + "\n")
    f.write("preds_p_list = " + str(preds_p_list) + "\n")
    f.write("saf_list = " + str(saf_list) + "\n")
    f.write("calibration_years = (" + str(calibration_years[0]) + ", " + str(calibration_years[1]) + ")\n")
    f.write("reference_years = (" + str(reference_years[0]) + ", " + str(reference_years[1]) + ")\n")
    f.write("historical_years = (" + str(historical_years[0]) + ", " + str(historical_years[1]) + ")\n")
    f.write("ssp_years = (" + str(ssp_years[0]) + ", " + str(ssp_years[1]) + ")\n")
    f.write("biasCorr_years = (" + str(biasCorr_years[0]) + ", " + str(biasCorr_years[1]) + ")\n")
    if bc_method == None:
        f.write("bc_method = " + str(bc_method) + "\n")
    else:
        f.write("bc_method = '" + str(bc_method) + "'\n")

    f.write("single_split_testing_years = (" + str(single_split_testing_years[0]) + ", " + str(single_split_testing_years[1]) + ")\n")
    f.write("fold1_testing_years = (" + str(fold1_testing_years[0]) + ", " + str(fold1_testing_years[1]) + ")\n")
    f.write("fold2_testing_years = (" + str(fold2_testing_years[0]) + ", " + str(fold2_testing_years[1]) + ")\n")
    f.write("fold3_testing_years = (" + str(fold3_testing_years[0]) + ", " + str(fold3_testing_years[1]) + ")\n")
    f.write("fold4_testing_years = (" + str(fold4_testing_years[0]) + ", " + str(fold4_testing_years[1]) + ")\n")
    f.write("fold5_testing_years = (" + str(fold5_testing_years[0]) + ", " + str(fold5_testing_years[1]) + ")\n")


    f.write("hresPeriodFilename = {}\n")
    f.write("hresPeriodFilename.update({'t': '" + str(hresPeriodFilename_t) + "'})\n")
    f.write("hresPeriodFilename.update({'p': '" + str(hresPeriodFilename_p) + "'})\n")
    f.write("reanalysisName = '" + str(reanalysisName) + "'\n")
    f.write("reanalysisPeriodFilename = '" + str(reanalysisPeriodFilename) + "'\n")
    f.write("historicalPeriodFilename = '" + str(historicalPeriodFilename) + "'\n")
    f.write("sspPeriodFilename = '" + str(sspPeriodFilename) + "'\n")
    f.write("split_mode = '" + str(split_mode) + "'\n")
    f.write("grid_res = " + str(grid_res) + "\n")
    f.write("saf_lat_up = " + str(saf_lat_up) + "\n")
    f.write("saf_lon_left = " + str(saf_lon_left) + "\n")
    f.write("saf_lon_right = " + str(saf_lon_right) + "\n")
    f.write("saf_lat_down = " + str(saf_lat_down) + "\n")


    f.write("model_names_list = " + str(model_names_list) + "\n")
    f.write("scene_names_list = " + str(scene_names_list) + "\n")
    f.write("climdex_names = " + str(climdex_names) + "\n")

    # Close f
    f.close()


########################################################################################################################
def write_tmpMain_file(steps):

    """This function prepares a tmp main file with the user selected options"""

    # Open f for writing
    f = open('.tmp_main.py', "w")


    f.write("import sys\n")
    f.write("sys.path.append('../config/')\n")
    f.write("from imports import *\n")
    f.write("from settings import *\n")
    f.write("from advanced_settings import *\n")

    f.write("\n")
    f.write("def main():\n")

    # Steps
    noSteps = True
    if len(steps) > 0:
        f.write("    aux_lib.initial_checks()\n")
    if 'preprocess' in steps:
        noSteps = False
        f.write("    preprocess.preprocess()\n")
    if 'missing_data_check' in steps:
        noSteps = False
        f.write("    precontrol.missing_data_check()\n")
    if 'predictors_correlation' in steps:
        noSteps = False
        f.write("    precontrol.predictors_correlation()\n")
    if 'GCMs_evaluation' in steps:
        noSteps = False
        f.write("    precontrol.GCMs_evaluation()\n")
    if 'train_methods' in steps:
        noSteps = False
        f.write("    preprocess.train_methods()\n")
    if 'downscale' in steps:
        noSteps = False
        f.write("    process.downscale()\n")
    if 'calculate_climdex' in steps:
        noSteps = False
        f.write("    postprocess.get_climdex()\n")
    if 'plot_results' in steps:
        noSteps = False
        f.write("    postprocess.plot_results()\n")
    if 'bias_correct_projections' in steps:
        noSteps = False
        f.write("    postprocess.bias_correction_projections()\n")
    if 'nc2ascii' in steps:
        noSteps = False
        f.write("    postprocess.nc2ascii()\n")
    if noSteps == True:
        print('-----------------------------------------------')
        print('At least one step must be selected.')
        print('-----------------------------------------------')
        exit()

    f.write("\n")
    f.write("if __name__ == '__main__':\n")
    f.write("    start = datetime.datetime.now()\n")
    f.write("    main()\n")
    f.write("    end = datetime.datetime.now()\n")
    f.write("    print('Elapsed time: ' + str(end - start))\n")
    f.write("    input('')\n")

    # Close f
    f.close()


########################################################################################################################
def main():
    """
    This function shows a graphical dialog to select settings and launch the main program.
    """

    # Seletcion window
    selectionWindow()

if __name__=="__main__":
    main()
