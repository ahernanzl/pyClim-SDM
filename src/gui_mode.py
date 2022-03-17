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
            Label(root, text='', borderwidth=0).grid(sticky="SE", column=0, row=0, pady=0)
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
                # justify="center", font=("Arial", 12)
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

        fontSize = 10

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
        ttk.Label(tabSteps, text="Select experiment:", font=("Arial", fontSize)).grid(sticky="E", column=icol, row=irow, padx=10, pady=15); icol+=1
        self.experiment = StringVar()
        experiments = {'PRECONTROL': 'Evaluation of predictors and GCMs previous to dowscaling',
                       'EVALUATION': 'Evaluate methods using a reanalysis over a historical period',
                       'PROJECTIONS': 'Apply methods to dowscale climate projections'}
        for exp in experiments:
            c = Radiobutton(tabSteps, text=exp, font=("Arial", fontSize), variable=self.experiment, value=exp,
                            command=lambda: switch_steps(self.experiment.get(), steps, self.steps_ordered,
                            self.exp_ordered, self.chk_only_for_experiment), takefocus=False)
            c.grid(sticky="W", column=icol, row=irow, padx=30); icol+=1
            CreateToolTip(c, experiments[exp])
            self.experiment.set(experiment)

        irow += 2
        icol = 1

        # Steps definition
        ttk.Label(tabSteps, text="Select steps:", font=("Arial", fontSize)).grid(sticky="E", column=icol, row=irow, padx=10); irow+=1

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
                c = Checkbutton(tabSteps, text=steps[exp_name][step]['text'], font=("Arial", fontSize), variable=checked, takefocus=False)
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

        familyFontSize = 10
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
            ttk.Label(tabMethods, text=variable, font=("Arial", 12, "bold"))\
                .grid(sticky="W", column=icol, row=irow, padx=30, pady=20, columnspan=4); irow+=1

            # Raw
            add_to_chk_list(self.chk_list, var, 'RAW', 'RAW', 'RAW', 'var', 'No downscaling', icol, irow); irow+=1

            # Bias correction
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow+=1
            ttk.Label(tabMethods, text="Bias Correction:", font=("Arial", familyFontSize, "bold")).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'QM', 'BC', 'MOS', 'var', 'Quantile Mapping', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'DQM', 'BC', 'MOS', 'var', 'Detrended Quantile Mapping', icol, irow); icol+=1; irow-=1
            add_to_chk_list(self.chk_list, var, 'QDM', 'BC', 'MOS', 'var', 'Quantile Delta Mapping', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'PSDM', 'BC', 'MOS', 'var', '(Parametric) Scaled Distribution Mapping', icol, irow); icol-=1; irow+=1

            # Analogs / Weather Typing
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Analogs / Weather Typing:", font=("Arial", familyFontSize, "bold")).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'ANA-MLR', 'ANA', 'PP', 'pred+saf', 'Analog Multiple Linear Regression', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'WT-MLR', 'ANA', 'PP', 'pred+saf', 'Weather Typing Multiple Linear Regression', icol, irow); irow+=1

            # Transfer function
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow+=1
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Transfer Function:", font=("Arial", familyFontSize, "bold")).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
            add_to_chk_list(self.chk_list, var, 'MLR', 'TF', 'PP', 'pred', 'Multiple Linear Regression', icol, irow); icol+=1
            add_to_chk_list(self.chk_list, var, 'ANN', 'TF', 'PP', 'pred', 'Artificial Neural Network', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'SVM', 'TF', 'PP', 'pred', 'Support Vector Machine', icol, irow); irow+=1
            add_to_chk_list(self.chk_list, var, 'LS-SVM', 'TF', 'PP', 'pred', 'Least Square Support Vector Machine', icol, irow); icol-=1; irow+=1

            # Weather Generators
            ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
            ttk.Label(tabMethods, text="Weather Generators:", font=("Arial", familyFontSize, "bold")).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4); irow+=1
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
        ttk.Label(tabMethods, text=variable, font=("Arial", 12, "bold"))\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1

        # Raw
        add_to_chk_list(self.chk_list, var, 'RAW', 'RAW', 'RAW', 'var', 'No downscaling', icol, irow); irow += 1

        # Bias correction
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Bias Correction:", font=("Arial", familyFontSize, "bold"))\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_to_chk_list(self.chk_list, var, 'QM', 'BC', 'MOS', 'var', 'Quantile Mapping', icol, irow); irow += 1
        add_to_chk_list(self.chk_list, var, 'DQM', 'BC', 'MOS', 'var', 'Detrended Quantile Mapping', icol, irow); icol+=1; irow -= 1
        add_to_chk_list(self.chk_list, var, 'QDM', 'BC', 'MOS', 'var', 'Quantile Delta Mapping', icol, irow); irow += 1
        add_to_chk_list(self.chk_list, var, 'PSDM', 'BC', 'MOS', 'var', '(Parametric) Scaled Distribution Mapping', icol, irow); icol-=1; irow += 1

        # Analogs / Weather Typing
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Analogs / Weather Typing:", font=("Arial", familyFontSize, "bold"))\
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
        ttk.Label(tabMethods, text="Transfer Function:", font=("Arial", familyFontSize, "bold"))\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-LIN', 'TF', 'PP', 'pred', 'Generalized Linear Model (linear)', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-EXP', 'TF', 'PP', 'pred', 'Generalized Linear Model (exponential)', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'GLM-CUB', 'TF', 'PP', 'pred', 'Generalized Linear Model (cubic)', icol, irow); irow-=2; icol+=1
        add_to_chk_list(self.chk_list, var, 'ANN', 'TF', 'PP', 'pred', 'Artificial Neural Network', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'SVM', 'TF', 'PP', 'pred', 'Support Vector Machine', icol, irow); irow+=1
        add_to_chk_list(self.chk_list, var, 'LS-SVM', 'TF', 'PP', 'pred', 'Least Square Support Vector Machine', icol, irow); irow+=1; icol-=1

        # Weather Generators
        ttk.Label(tabMethods, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(tabMethods, text="Weather Generators:", font=("Arial", familyFontSize, "bold"))\
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
            if nrows == 8:
                nrows = 0
                icol += 5
                irow -= 8
            return irow, icol, nrows

        irow = 0
        icol = 0


        Label(tabPredictors, text="").grid(sticky="W", padx=10, row=irow, column=icol); icol += 1
        ttk.Label(tabPredictors, text="").grid(sticky="W", column=icol, columnspan=100, row=irow, padx=30, pady=10); irow+=2

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
            Label(tabPredictors, text=title, font=("Arial", 12, "bold")).grid(columnspan=10, row=irow, column=icol)
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
            Label(tabClimdex, text=title, font=("Arial", 12, "bold")).grid(sticky="W", column=icol, row=irow, padx=30, pady=30, columnspan=3); irow+=1
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
                c = Checkbutton(tabModelsAndScenes, text=name, variable=checked, takefocus=False)
            else:
                c = Checkbutton(tabModelsAndScenes, text=name, variable=checked, command=lambda: switch(obj), takefocus=False)
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
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, padx=50); icol+=1
        Label(tabModelsAndScenes, text="Select models and scenarios:").grid(sticky="W", column=icol, row=irow, padx=30, pady=30,
                                                          columnspan=100);
        irow += 1


        # Models
        all_models = ('ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5', 'CESM2',
                      'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1',
                      'EC-Earth3', 'EC-Earth3-AerChem', 'EC-Earth3-CC', 'EC-Earth3-Veg',
                      'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-CM4', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL',
                      'HadGEM3-GC31-MM', 'IITM-ESM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM5A2-INCA', 'IPSL-CM6A-LR',
                      'KACE-1-0-G', 'KIOST-ESM', 'MIROC-ES2L', 'MIROC6', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR',
                      'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1', 'UKESM1-0-LL')
        maxRows = 15
        ncols = 0
        nrows = maxRows
        for model in all_models:
            self.chk_dict_models.update(add_to_chk_list(model, model_names_list, icol, irow, affectedBySelectAll=True))
            irow += 1; nrows-=1
            if nrows == 0:
                ncols+=1; nrows = maxRows; icol+=1; irow-=maxRows

        # Other models
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow); irow+=1
        self.otherModels_var = tk.StringVar()
        self.otherModels_Entry = tk.Entry(tabModelsAndScenes, textvariable=self.otherModels_var, width=15, justify='right', state='disabled')
        self.otherModels_Entry.grid(sticky="E", column=icol, row=irow, padx=100)
        CreateToolTip(self.otherModels_Entry, "Enter model names separated by ';'")
        self.chk_dict_models.update(add_to_chk_list('Others:', model_names_list, icol, irow, obj=self.otherModels_Entry)); irow += 1

        # Select all models
        irow+=maxRows; icol-=ncols
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, pady=10); irow+=1
        Button(tabModelsAndScenes, text='Select all', command=select_all, takefocus=False).grid(sticky="E", column=icol, row=irow); icol += 1
        Button(tabModelsAndScenes, text='Deselect all', command=deselect_all, takefocus=False).grid(sticky="E", column=icol, row=irow)

        # Scenes
        irow, icol = 1, 6
        Label(tabModelsAndScenes, text="").grid(sticky="W", column=icol, row=irow, padx=20); icol+=1
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

        # modelRealizationFilename
        irow+=3
        Label(tabModelsAndScenes, text="Realization:")\
            .grid(sticky="W", column=icol, row=irow, padx=3)#; irow+=1
        self.modelRealizationFilename_var = tk.StringVar()
        self.modelRealizationFilename_Entry = tk.Entry(tabModelsAndScenes, textvariable=self.modelRealizationFilename_var, width=8, justify='right', takefocus=False)
        self.modelRealizationFilename_Entry.grid(sticky="W", column=icol, row=irow, padx=80)
        self.modelRealizationFilename_Entry.insert(END, modelRealizationFilename)


    def get(self):
        return self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var, \
               self.modelRealizationFilename_var



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
            ('Calibration:', calibration_years, 'Longest period available, which then can be split for training and testing'),
            ('Reference:', reference_years, 'For standardization and as reference climatology'),
            ('Historical:', historical_years, 'For historical projections'),
            ('SSPs:', ssp_years, 'For future projections'),
            ('Bias correction:', biasCorr_years, 'For bias correction of projections'),
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

        # hresPeriodFilename
        self.hresPeriodFilename_var = tk.StringVar()
        Label(framePeriodFilenames, text='Predictands period filename:').grid(sticky="W", column=icol, row=irow, padx=10); icol+=1
        hresPeriodFilename_Entry = tk.Entry(framePeriodFilenames, textvariable=self.hresPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        hresPeriodFilename_Entry.insert(END, hresPeriodFilename)
        hresPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

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
               self.bc_method, self.testing_years_dict, self.hresPeriodFilename_var, self.reanalysisName_var, \
               self.reanalysisPeriodFilename_var, self.historicalPeriodFilename_var, self.sspPeriodFilename_var, \
               self.split_mode, self.grid_res_var, \
               self.saf_lat_up_var, self.saf_lon_left_var, self.saf_lon_right_var, self.saf_lat_down_var



########################################################################################################################
class selectionWindow():

    def __init__(self):

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
        self.experiment, self.steps_dict, self.all_steps = tabSteps(notebook, root).get()

        # Tab: methods
        self.methods_chk = tabMethods(notebook).get()

        # Tab: predictors
        self.reaNames, self.modNames, self.preds = tabPredictors(notebook).get()

        # Tab: climdex
        self.climdex_dict_chk = tabClimdex(notebook).get()

        # Tab: models
        self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var, \
                self.modelRealizationFilename_var = tabModelsAndScenes(notebook).get()

        # Tab: dates and Domain
        self.calibration_years, self.reference_years, self.historical_years, self.ssp_years, self.biasCorr_years, \
            self.bc_method, self.testing_years_dict, self.hresPeriodFilename_var, self.reanalysisName_var, \
               self.reanalysisPeriodFilename_var, self.historicalPeriodFilename_var, self.sspPeriodFilename_var, \
                self.split_mode, self.grid_res_var, \
                   self.saf_lat_up_var, self.saf_lon_left_var, self.saf_lon_right_var, self.saf_lat_down_var\
                = tabDatesAndDomain(notebook).get()

        # Logo
        Label(notebook, text='', borderwidth=0).grid(sticky="SE", column=0, row=0, pady=265)
        w = 120
        img = Image.open("../doc/pyClim-SDM_logo.png")
        h = int(w * img.height / img.width)
        img = img.resize((w, h), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        canvas = Canvas(notebook, width=w, height=h)
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.grid(sticky="W", column=0, row=1, padx=10)

        # Run butnon
        Label(notebook, text='', borderwidth=0).grid(sticky="SE", column=1, row=0, padx=480)
        self.run = False
        def run():
            self.run = True
            root.destroy()
        Button(notebook, text="Run", width=10, command=run).grid(sticky="W", column=2, row=1, padx=20, pady=0)

        # Mainloop
        root.mainloop()


    def get(self):

        # Experiment
        self.experiment = self.experiment.get()

        # Steps
        self.steps = []
        for step in self.all_steps[self.experiment]:
            if self.steps_dict[self.experiment][step].get() == True:
                self.steps.append(step)

        # Methods
        self.methods = []
        for meth in self.methods_chk:
            if meth['checked'].get() == True:
                del meth['checked']
                self.methods.append(meth)

        # reaNames and modNames
        for var in self.reaNames:
            self.reaNames.update({var: self.reaNames[var].get()})
            self.modNames.update({var: self.modNames[var].get()})

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
        self.calibration_years = (self.calibration_years[0].get() ,self.calibration_years[1].get())
        self.reference_years = (self.reference_years[0].get() ,self.reference_years[1].get())
        self.historical_years = (self.historical_years[0].get() ,self.historical_years[1].get())
        self.ssp_years = (self.ssp_years[0].get() ,self.ssp_years[1].get())
        self.biasCorr_years = (self.biasCorr_years[0].get() ,self.biasCorr_years[1].get())

        # bc_method
        self.bc_method = self.bc_method.get()
        if self.bc_method == 'None':
            self.bc_method = None

        # split_mode and testing years
        self.split_mode = self.split_mode.get()
        self.single_split_testing_years = (self.testing_years_dict['single_split'][0].get(), self.testing_years_dict['single_split'][1].get())
        self.fold1_testing_years = (self.testing_years_dict['fold1'][0].get(), self.testing_years_dict['fold1'][1].get())
        self.fold2_testing_years = (self.testing_years_dict['fold2'][0].get(), self.testing_years_dict['fold2'][1].get())
        self.fold3_testing_years = (self.testing_years_dict['fold3'][0].get(), self.testing_years_dict['fold3'][1].get())
        self.fold4_testing_years = (self.testing_years_dict['fold4'][0].get(), self.testing_years_dict['fold4'][1].get())
        self.fold5_testing_years = (self.testing_years_dict['fold5'][0].get(), self.testing_years_dict['fold5'][1].get())

        # period filenames
        self.hresPeriodFilename_var = self.hresPeriodFilename_var.get()
        self.reanalysisName_var = self.reanalysisName_var.get()
        self.reanalysisPeriodFilename_var = self.reanalysisPeriodFilename_var.get()
        self.historicalPeriodFilename_var = self.historicalPeriodFilename_var.get()
        self.sspPeriodFilename_var = self.sspPeriodFilename_var.get()

        # grid_res
        self.grid_res_var = self.grid_res_var.get()
        self.saf_lat_up_var = self.saf_lat_up_var.get()
        self.saf_lon_left_var = self.saf_lon_left_var.get()
        self.saf_lon_right_var = self.saf_lon_right_var.get()
        self.saf_lat_down_var = self.saf_lat_down_var.get()

        # Models
        self.model_names_list = []
        for model in self.chk_dict_models:
            if self.chk_dict_models[model].get() == True:
                self.model_names_list.append(model)
        if 'Others:' in self.model_names_list:
            self.model_names_list.remove('Others:')
            otherModels = self.otherModels_var.get()
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

        # modelRealizationFilename
        self.modelRealizationFilename = self.modelRealizationFilename_var.get()

        return self.run, self.experiment, self.steps, self.methods, self.reaNames, self.modNames, self.preds_t_list, \
               self.preds_p_list, self.saf_list, self.climdex_names, self.calibration_years, \
               self.reference_years, self.historical_years, self.ssp_years, self.biasCorr_years, self.bc_method, \
               self.single_split_testing_years, self.fold1_testing_years, self.fold2_testing_years, \
               self.fold3_testing_years, self.fold4_testing_years, self.fold5_testing_years, \
               self.hresPeriodFilename_var, self.reanalysisName_var, self.reanalysisPeriodFilename_var, \
               self.historicalPeriodFilename_var, self.sspPeriodFilename_var, self.split_mode, \
               self.grid_res_var, \
               self.saf_lat_up_var, self.saf_lon_left_var, self.saf_lon_right_var, self.saf_lat_down_var, \
               self.model_names_list, self.scene_names_list, self.modelRealizationFilename

########################################################################################################################
def write_settings_file(showWelcomeMessage, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list,
                        saf_list, climdex_names, calibration_years, reference_years, historical_years, ssp_years, biasCorr_years,
                        bc_method, single_split_testing_years, fold1_testing_years, fold2_testing_years,
                        fold3_testing_years, fold4_testing_years, fold5_testing_years, hresPeriodFilename,
                        reanalysisName, reanalysisPeriodFilename, historicalPeriodFilename,
                        sspPeriodFilename, split_mode, grid_res, saf_lat_up, saf_lon_left, saf_lon_right,
                        saf_lat_down, model_names_list, scene_names_list, modelRealizationFilename):

    """This function prepares a new settings file with the user selected options"""

    # Open f for writing
    f = open('../config/settings.py', "w")

    # Write new settings
    f.write("showWelcomeMessage = " + str(showWelcomeMessage) + "\n")
    f.write("experiment = '" + str(experiment) + "'\n")
    f.write("steps = " + str(steps) + "\n")
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

    f.write("hresPeriodFilename = '" + str(hresPeriodFilename) + "'\n")
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
    f.write("modelRealizationFilename = '" + str(modelRealizationFilename) + "'\n")
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
        print('At least one step must be selected.')
        exit()

    f.write("\n")
    f.write("if __name__ == '__main__':\n")
    f.write("    start = datetime.datetime.now()\n")
    f.write("    main()\n")
    f.write("    end = datetime.datetime.now()\n")
    f.write("    print('Elapsed time: ' + str(end - start))")

    # Close f
    f.close()


########################################################################################################################
def main():
    """
    This function shows a graphical dialog to select settings and launch the main program.
    """

    # Welcome message
    run, showWelcomeMessage = welcomeMessage().get()
    if run == False:
        exit()

    # Seletcion window
    run, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list, saf_list, climdex_names, \
        calibration_years, reference_years, historical_years, ssp_years, biasCorr_years, bc_method, \
        single_split_testing_years, fold1_testing_years, fold2_testing_years, fold3_testing_years, fold4_testing_years, \
        fold5_testing_years, hresPeriodFilename, reanalysisName, reanalysisPeriodFilename, \
        historicalPeriodFilename, sspPeriodFilename, split_mode, grid_res, \
        saf_lat_up, saf_lon_left, saf_lon_right, saf_lat_down, model_names_list, scene_names_list, \
        modelRealizationFilename = selectionWindow().get()
    if run == False:
        exit()


    # Write settings file
    write_settings_file(showWelcomeMessage, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list,
                        saf_list, climdex_names, calibration_years, reference_years, historical_years, ssp_years, biasCorr_years,
                        bc_method, single_split_testing_years, fold1_testing_years, fold2_testing_years,
                        fold3_testing_years, fold4_testing_years, fold5_testing_years, hresPeriodFilename,
                        reanalysisName, reanalysisPeriodFilename, historicalPeriodFilename,
                        sspPeriodFilename, split_mode, grid_res, saf_lat_up, saf_lon_left, saf_lon_right,
                        saf_lat_down, model_names_list, scene_names_list, modelRealizationFilename)

    write_tmpMain_file(steps)


if __name__=="__main__":
    main()
    os.system('python3 .tmp_main.py')
    os.remove('.tmp_main.py')
