import sys
import os
import shutil
sys.path.append('../config/')
from manual_settings import *
if not os.path.isfile('../config/settings.py') or os.stat('../config/settings.py').st_size == 0:
    shutil.copyfile('../config/default_gui_settings.py', '../config/settings.py')
# shutil.copyfile('../config/default_gui_settings.py', '../config/settings.py')
from imports import *
from settings import *
from advanced_settings import *
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from tkinter import *
from tkinter import messagebox as messagebox
from tkinter import DISABLED
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
def switch_bc_method(bc_opt, bc_mehods_bt):
    for object in bc_mehods_bt:
        if bc_opt == 'No':
            object["state"] = "disabled"
        else:
            object["state"] = "normal"

    

########################################################################################################################
def enable(targetVar_active_var, frames):
    """Enable/diable targeVar options"""

    for frame in frames:
        for child in frame.winfo_children():
            if child.winfo_class() == 'TFrame':
                frames.append(child)

    for frame in frames:
        for child in frame.winfo_children():
            if child.winfo_class() != 'TFrame':
                try:
                    if targetVar_active_var == True:
                        child.configure(state='normal')
                    else:
                        child.configure(state='disable')
                except:
                    print(child.winfo_class())


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
            try:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            except:
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
        ttk.Label(tabSteps, text="").grid(column=1, row=irow, pady=10); irow+=1

        # frameExplanation
        frameExplanation = Frame(tabSteps)
        frameExplanation.grid(sticky="W", column=1, row=irow, padx=10, pady=10, columnspan=10)
        icol, irow = 0, 0
        explanation = 'Before running the different steps, select the methods and predictors to be used. For the GCMs evaluation, the selected predictors will be analyzed.\n' \
                      'Do not run a step before the previous steps have finished succesfully. When working at an HPC, wait for previous jobs to finish. Steps "Train methods", \n' \
                      '"Downscale", "Bias correcion" and "Calculate climdex" are prepared to be done as jobs, in parallel.\n'
        ttk.Label(frameExplanation, text=explanation).grid(sticky="W", column=icol, row=irow, padx=30, pady=20, columnspan=4)

        ttk.Label(tabSteps, text="").grid(column=1, row=irow, pady=10); irow+=5

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
                 'check_var_units': {'text': 'Check units', 'info':  'Check units for all variables from reanalysis and GCMs.',},
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
                'bias_correction': {'text': 'Bias correct (optional)', 'info': 'Bias correct after downscaling.'},
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
                'bias_correction': {'text': 'Bias correct (optional)', 'info': 'Bias correct after downscaling.'},
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
class tabModelsAndScenes(ttk.Frame):

    def __init__(self, notebook):
        tabModelsAndScenes = ttk.Frame(notebook)
        notebook.add(tabModelsAndScenes, text='Models and Scenarios')

        self.chk_dict_models = {}

        def add_to_chk_list(frame, name, list, icol, irow, obj=None, affectedBySelectAll=False):
            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            if name in list:
                checked = tk.BooleanVar(value=True)
            if obj == None:
                c = Checkbutton(frame, text=name.split('_')[0], variable=checked, takefocus=False)
            else:
                c = Checkbutton(frame, text=name.split('_')[0], variable=checked, command=lambda: switch(obj), takefocus=False)
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

        # frameModels
        frameModels = ttk.Frame(tabModelsAndScenes)
        frameModels.grid(row=0, column=0, sticky='n', padx=(40, 0), rowspan=2)

        # frameReanalysisName
        frameReanalysisName = ttk.Frame(tabModelsAndScenes)
        frameReanalysisName.grid(row=0, column=1, sticky='n', padx=(40, 0))

        # frameScenes
        frameScenes = ttk.Frame(tabModelsAndScenes)
        frameScenes.grid(row=1, column=1, sticky='n', padx=(40, 0))

        Label(frameModels, text="").grid(sticky="W", column=icol, row=irow, padx=20, pady=0); icol+=1; irow+=1
        Label(frameModels, text="Select models from the list to include their r1i1p1f1 run:")\
            .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=100)
        Label(frameModels, text="").grid(sticky="W", column=icol, row=irow, pady=10); irow+=1


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
            self.chk_dict_models.update(add_to_chk_list(frameModels, model, model_names_list, icol, irow, affectedBySelectAll=True))
            irow += 1; nrows-=1
            if nrows == 0:
                ncols+=1; nrows = maxRows; icol+=1; irow-=maxRows

        # Select all models
        irow+=1; icol-=ncols-1
        Button(frameModels, text='Select all', command=select_all, takefocus=False).grid(column=icol, row=irow, pady=10); icol += 1
        Button(frameModels, text='Deselect all', command=deselect_all, takefocus=False).grid(column=icol, row=irow)


        # Other models
        irow+=maxRows
        icol -=2

        Label(frameModels, text="").grid(sticky="W", column=icol, row=irow, padx=30, pady=5, columnspan=4); irow+=1
        Label(frameModels, text="In order to include other models and/or runs introduce them here "
                                       "separated by ';'")\
                                    .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4); irow+=1
        Label(frameModels, text="Example: ACCESS-CM2_r1i1p1f3; EC-Earth3_r2i1p1f1")\
                                    .grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4)
        otherModels_list = []
        for model in model_names_list:
            if model.split('_')[1] != 'r1i1p1f1':
                otherModels_list.append(model)
        otherModels_list = '; '.join((otherModels_list))

        icol += 1
        self.otherModels_var = tk.StringVar()
        self.otherModels_Entry = tk.Entry(frameModels, textvariable=self.otherModels_var, width=45,
                                          justify='left', state='normal', takefocus=False)
        self.otherModels_Entry.insert(END, otherModels_list)
        self.otherModels_Entry.grid(sticky="E", column=icol, row=irow, columnspan=3)
        icol += 1


        # frameReanalysisName
        Label(frameReanalysisName, text="").grid(sticky="W", column=icol, row=irow, pady=(40, 0)); icol+=1; irow+=1
        self.reanalysisName_var = tk.StringVar()
        Label(frameReanalysisName, text='Reanalysis name:').grid(sticky="W", column=icol, row=irow, padx=10); icol += 1
        reanalysisName_Entry = tk.Entry(frameReanalysisName, textvariable=self.reanalysisName_var, width=15, justify='right', takefocus=False)
        reanalysisName_Entry.insert(END, reanalysisName)
        reanalysisName_Entry.grid(sticky="W", column=icol, row=irow)


        # Scenes
        irow, icol = 0, 0
        Label(frameScenes, text="Select scenarios:").grid(sticky="NW", column=icol, row=irow, padx=50); irow+=1
        all_scenes = ['HISTORICAL', 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
        self.chk_dict_scenes = {}
        for scene in all_scenes:
            self.chk_dict_scenes.update(add_to_chk_list(frameScenes, scene, scene_names_list, icol, irow)); irow += 1

        # Other scenes
        Label(frameScenes, text="").grid(sticky="W", column=icol, row=irow)
        self.otherScenes_var = tk.StringVar()
        self.otherScenes_Entry = tk.Entry(frameScenes, textvariable=self.otherScenes_var, width=15, justify='right', state='disabled')
        self.otherScenes_Entry.grid(sticky="E", column=icol, row=irow, padx=100)
        CreateToolTip(self.otherScenes_Entry, "Enter scenario names separated by ';'")
        self.chk_dict_scenes.update(add_to_chk_list(frameScenes, 'Others:', scene_names_list, icol, irow, obj=self.otherScenes_Entry)); irow += 1

    def get(self):
        return self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var, self.reanalysisName_var


########################################################################################################################
class tabDomain(ttk.Frame):

    def __init__(self, notebook):

        tabDomain = ttk.Frame(notebook)
        notebook.add(tabDomain, text='Domain')

        icol, irow = 0, 0
        # ttk.Label(frameDomain, text="Define the following spatial information:").grid(sticky="W", column=icol, row=irow, padx=10, pady=30, columnspan=100); irow+=1
        padx = 100
        # frameDomain
        frameDomain = Frame(tabDomain)
        frameDomain.grid(sticky="W", column=0, row=0, padx=padx, pady=10)

        # frameVarNames
        frameVarNames = Frame(tabDomain)
        frameVarNames.grid(sticky="W", column=0, row=1, padx=padx, pady=10, columnspan=2)

        # frameSAFs
        frameSAFs = Frame(tabDomain)
        frameSAFs.grid(sticky="W", column=1, row=0, padx=padx, pady=10)


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

        # reaNames and modNames
        irow, icol = 0, 0
        ttk.Label(frameVarNames, text='').grid(column=icol, row=irow, pady=0, ); irow+=1
        ttk.Label(frameVarNames, text='Define variable names in netCDF files:')\
            .grid(column=icol, row=irow, pady=10, columnspan=13); irow += 1

        # reaNames and modNames
        self.reaNames = {}
        self.modNames = {}
        ncols = 13
        all_vars = {
                'tasmax': 'Surface maximum temperature',
                'tasmin': 'Surface minimum temperature',
                'tas': 'Surface mean temperature',
                'pr': 'Precipitation',
                'uas': 'Surface zonal wind component',
                'vas': 'Surface meridional wind component',
                'sfcWind': 'Surface wind speed',
                'hurs': 'Surface relative humidity',
                'clt': 'Cloud cover',
                'tdps': 'Surface dewpoint',
                'ps': 'Surface pressure',
                'huss': 'Surface specific humidity',
                'ua': 'Zonal wind',
                'va': 'Meridional wind',
                'ta': 'Temperature',
                'zg': 'Geopotential',
                'hus': 'Specific humidity',
                'hur': 'Relative humidity',
                'td': 'Dew point',
                'psl': 'Mean sea level pressure',
        }
        for var in all_vars:

            if icol == 0:
                ttk.Label(frameVarNames, text='').grid(column=icol, row=irow, pady=3, padx=60); irow += 1
                ttk.Label(frameVarNames, text='Reanalysis:').grid(sticky="E", column=icol, row=irow, padx=10, ); irow += 1
                ttk.Label(frameVarNames, text='Models:').grid(sticky="E", column=icol, row=irow, padx=10, ); irow += 1
                icol += 1; irow -= 3

            lab = ttk.Label(frameVarNames, text=var)
            CreateToolTip(lab, all_vars[var])
            lab.grid(sticky="E", column=icol, row=irow, pady=(10, 0)); irow+=1

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
            modName_Entry.grid(sticky="W", column=icol, row=irow); irow-=2
            self.modNames.update({var: self.modName_var})
            icol +=1
            if icol == ncols:
                irow+=3; icol-=ncols


        # SAFs
        self.SAFs = []
        self.SAFs = framePredictorsClass(notebook, frameSAFs, 'SAFs').get()


    def get(self):
        return self.grid_res_var, self.saf_lat_up_var, self.saf_lon_left_var, \
        self.saf_lon_right_var, self.saf_lat_down_var, self.reaNames, self.modNames, self.SAFs



########################################################################################################################
class tabDates(ttk.Frame):

    def __init__(self, notebook):
        tabDates = ttk.Frame(notebook)
        notebook.add(tabDates, text='Dates')
        padx = 100

        # frameDates
        frameDates = Frame(tabDates)
        frameDates.grid(sticky="W", column=0, row=0, padx=padx, pady=10)

        # frameSplitMode
        frameSplitMode = Frame(tabDates)
        frameSplitMode.grid(sticky="W", column=0, row=1, padx=padx, pady=10)

        # frameBiasCorrection
        frameBiasCorrection = Frame(tabDates)
        frameBiasCorrection.grid(sticky="W", column=1, row=0, padx=padx, pady=10)

        # frameSeasons
        frameSeasons = Frame(tabDates)
        frameSeasons.grid(sticky="W", column=1, row=1, padx=padx, pady=10, rowspan=2)


        icol, irow = 0, 0
        ttk.Label(frameDates, text="").grid(sticky="W", column=icol, row=irow, padx=50, pady=10, columnspan=100); irow+=1
        ttk.Label(frameDates, text="").grid(sticky="W", column=icol, row=irow, padx=50); icol += 1

        # Years
        for (text, var, info) in [
            ('Calibration years:', calibration_years, 'Longest period available by both reanalysis and hres data, \n'
                                                'which then can be split for training and testing'),
            ('Reference years:', reference_years, 'For standardization and as reference climatology. \n'
                                            'The choice of the reference period is constrained by availability of \n'
                                            'reanalysis, historical GCMs and hres data.'),
            ('Historical years:', historical_years, 'For historical projections'),
            ('SSPs years:', ssp_years, 'For future projections'),
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
            irow+=1; icol-=1

            if text == 'Calibration years:':
                self.calibration_years = (firstYear_var, lastYear_var)
            elif text == 'Reference years:':
                self.reference_years = (firstYear_var, lastYear_var)
            elif text == 'Historical years:':
                self.historical_years = (firstYear_var, lastYear_var)
            elif text == 'SSPs years:':
                self.ssp_years = (firstYear_var, lastYear_var)
            icol -= 2



        entriesW = 17

        # reanalysisPeriodFilename
        self.reanalysisPeriodFilename_var = tk.StringVar()
        Label(frameDates, text='Reanalysis period filename:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
        reanalysisPeriodFilename_Entry = tk.Entry(frameDates, textvariable=self.reanalysisPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        reanalysisPeriodFilename_Entry.insert(END, reanalysisPeriodFilename)
        reanalysisPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow, columnspan=8); icol-=1; irow+=1

        # historicalPeriodFilename
        self.historicalPeriodFilename_var = tk.StringVar()
        Label(frameDates, text='Historical period filename:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
        historicalPeriodFilename_Entry = tk.Entry(frameDates, textvariable=self.historicalPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        historicalPeriodFilename_Entry.insert(END, historicalPeriodFilename)
        historicalPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow, columnspan=8); icol-=1; irow+=1

        # sspPeriodFilename
        self.sspPeriodFilename_var = tk.StringVar()
        Label(frameDates, text='SSP period filename:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
        sspPeriodFilename_Entry = tk.Entry(frameDates, textvariable=self.sspPeriodFilename_var, width=entriesW, justify='right', takefocus=False)
        sspPeriodFilename_Entry.insert(END, sspPeriodFilename)
        sspPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow, columnspan=8); icol-=1; irow+=1


        # Bias correction
        irow, icol = 0, 0
        self.bc_option = StringVar()
        bc_options = {
            'No': 'Do not apply bias correction after downscaling.',
            'Yes': 'Apply bias correction after downscaling.',
            'By season': 'Apply a customized bias correction after downscaling for each season.',
        }
        Label(frameBiasCorrection, text='').grid(sticky="E", column=icol, row=irow, padx=10, pady=0, columnspan=3); irow+=1
        Label(frameBiasCorrection, text='Bias correction:').grid(sticky="E", column=icol, row=irow, padx=3, pady=0, columnspan=1); icol+=1
        if apply_bc == False:
            last_bc_opt = 'No'
        elif apply_bc_bySeason == False:
            last_bc_opt = 'Yes'
        else:
            last_bc_opt = 'By season'
        for bc_opt in bc_options:
            c = Radiobutton(frameBiasCorrection, text=bc_opt, variable=self.bc_option, value=bc_opt, command=lambda: switch_bc_method(self.bc_option.get(), bc_mehods_bt), takefocus=False)
            c.grid(sticky="W", column=icol, row=irow, padx=5, columnspan=1); irow+=1
            self.bc_option.set(last_bc_opt)
        icol+=1; irow-=3
        Label(frameBiasCorrection, text='').grid(sticky="E", column=icol, row=irow, padx=10, pady=0); icol+=1
        Label(frameBiasCorrection, text='Method:').grid(sticky="E", column=icol, row=irow, padx=3, pady=0, columnspan=1); icol+=1
        bc_methods = {
            'QM': 'Quantile Mapping',
            'DQM': 'Detrended Quantile Mapping',
            'QDM': 'Quantile Delta Mapping',
            'PSDM': '(Parametric) Scaled Distribution Mapping',
        }
        self.bc_method = StringVar()
        bc_mehods_bt = []
        for bc_meth in bc_methods:
            c1 = Radiobutton(frameBiasCorrection, text=bc_meth, variable=self.bc_method, value=bc_meth, takefocus=False)
            c1.grid(sticky="W", column=icol, row=irow, padx=5, columnspan=1); irow+=1
            CreateToolTip(c1, bc_methods[bc_meth])
            self.bc_method.set(bc_method)
            bc_mehods_bt.append(c1)
        switch_bc_method(last_bc_opt, bc_mehods_bt)


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


        # season_dict
        self.seasons = []
        def add_season(xMonth, text, seasonName, irow, icol):
            monthsVar = tk.StringVar()
            Label(frameSeasons, text=text).grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            seasonName_Entry = tk.Entry(frameSeasons, textvariable=monthsVar, width=10, justify='right',
                                        takefocus=False)
            seasonName_Entry.insert(END, seasonName)
            seasonName_Entry.grid(sticky="E", column=icol, row=irow)
            self.seasons.append(monthsVar)
            icol -= 1

        icol, irow = 0, 0
        Label(frameSeasons, text='Defina season names:').grid(sticky="W", column=icol, row=irow, padx=10, pady=5, columnspan=5); irow+=1
        for (xMonth, text, seasonName) in [
            (0, 'All year:', inverse_seasonNames[0]),
            (1, 'Jan:', inverse_seasonNames[1]),
            (2, 'Feb:', inverse_seasonNames[2]),
            (3, 'Mar:', inverse_seasonNames[3]),
            (4, 'Apr:', inverse_seasonNames[4]),
            (5, 'May:', inverse_seasonNames[5]),
            (6, 'Jun:', inverse_seasonNames[6]),
            (7, 'Jul:', inverse_seasonNames[7]),
            (8, 'Aug:', inverse_seasonNames[8]),
            (9, 'Sep:', inverse_seasonNames[9]),
            (10, 'Oct:', inverse_seasonNames[10]),
            (11, 'Nov:', inverse_seasonNames[11]),
            (12, 'Dec:', inverse_seasonNames[12]),
            ]:
            add_season(xMonth, text, seasonName, irow, icol)
            irow += 1



    def get(self):
        return self.calibration_years, self.reference_years, self.historical_years, self.ssp_years, self.bc_option, \
               self.bc_method, self.testing_years_dict, self.reanalysisPeriodFilename_var, self.historicalPeriodFilename_var, \
               self.sspPeriodFilename_var, self.split_mode, self.seasons



########################################################################################################################
class framePredictorsClass(ttk.Frame):

    def __init__(self, notebook, root, targetVar):


        def add_chk_bt_upperAir(chk_list, pred, irow, icol):
            """Check buttons for upper air predictors"""
            checked = tk.BooleanVar(value=False)
            if pred in pred_dictIn:
                checked = tk.BooleanVar(value=True)
            c = Checkbutton(root, variable=checked, takefocus=False)
            c.grid(row=irow, padx=padx, column=icol)
            chk_list.update({pred: checked})
            irow += 1
            return irow, icol

        def add_chk_bt_singleLevels(chk_list, pred, irow, icol, nrows):
            """Check buttons for single levels predictors"""
            checked = tk.BooleanVar(value=False)
            if pred in pred_dictIn:
                checked = tk.BooleanVar(value=True)
            c = Checkbutton(root, text=pred, variable=checked, takefocus=False)
            c.grid(sticky="W", row=irow, padx=2, column=icol, columnspan=5)
            CreateToolTip(c, singleLevelVars[pred])
            chk_list.update({pred: checked})
            irow += 1
            nrows += 1
            if nrows == 2:
                nrows = 0
                icol += 2
                irow -= 2
            return irow, icol, nrows

        irow = 0
        icol = 0


        Label(root, text="").grid(sticky="W", padx=10, row=irow, column=icol); icol += 1
        ttk.Label(root, text="").grid(sticky="W", column=icol, columnspan=100, row=irow, padx=20, pady=0); irow+=2

        # Levels
        ttk.Label(root, text="").grid(sticky="E", column=icol, row=irow, padx=30); irow+=1
        ttk.Label(root, text="").grid(sticky="E", column=icol, row=irow, pady=0, padx=30); irow+=1
        self.levels = [1000, 850, 700, 500, 250]
        for level in self.levels:
            Label(root,  text=str(level) + " hPa").grid(sticky="E", padx=10,  row=irow, column=icol); irow+=1
        Label(root, text="").grid(sticky="E", column=icol, row=irow, padx=10); irow-=6; icol+=1

        self.preds = {}

        if targetVar == 'SAFs':
            pred_dictIn = saf_dict
        else:
            try:
                pred_dictIn = preds_dict[targetVar]
            except:
                pred_dictIn = []

        irow -= 1
        if targetVar == 'SAFs':
            Label(root, text='Synoptic Analogy Fields').grid(columnspan=10, row=irow, pady=(10, 10), column=icol); irow += 1
        else:
            Label(root, text='Predictors').grid(columnspan=10, row=irow, pady=(10, 10), column=icol); irow += 1
        upperAirVars = {'ua': 'Eastward wind component',
                        'va': 'Northward wind component',
                        'ta': 'Temperature',
                        'zg': 'Geopotential',
                        'hus': 'Specific humidity',
                        'hur': 'Relative humidity',
                        'td': 'Dew point',
                        'Dtd': 'Dew point depresion',
                        'vort': 'Vorticity (derived from u and v)',
                        'div': 'Divergence (derived from u and v)',
                        }

        for var in upperAirVars:
            c = ttk.Label(root, text=var)
            c.grid(column=icol, row=irow, pady=10); irow += 1
            CreateToolTip(c, upperAirVars[var])
            padx = 2
            for level in self.levels:
                irow, icol = add_chk_bt_upperAir(self.preds, str(var) + str(level), irow, icol)
            irow -= 6; icol += 1

        Label(root, text="").grid(sticky="W", padx=20, row=irow, column=icol); icol += 1

        irow += 5
        icol -= 11

        singleLevelVars = {
                        'psl': 'Mean sea level pressure',
                        'clt': 'Cloud cover',
                        'uas': 'Surface eastward wind component',
                        'vas': 'Surface northward wind component',
                        'tas': 'Surface mean temperature',
                        'hurs': 'Surface relative humidity',
                        'K': 'K instability index',
                        'TT': 'Total Totals instability index',
                        # 'SSI': 'Showalter instability index',
                        # 'LI': 'Lifted instability index',
                        }

        # Label(root, text="").grid(sticky="W", row=irow, column=icol);
        irow += 1
        nrows = 0
        for pred in singleLevelVars:
            irow, icol, nrows = add_chk_bt_singleLevels(self.preds, pred, irow, icol, nrows)
        irow += 2
        ttk.Label(root, text='').grid(column=icol, row=irow, pady=5)

    def get(self):
        return self.preds


########################################################################################################################
class frameTargetVarInfoClass(ttk.Frame):

    def __init__(self, notebook, root, targetVar):

        icol = 0
        irow = 0
        entriesW = 17

        self.chk = {'hresPeriodFilename': tk.StringVar()}

        # try:
        #     isMyTargetVar = (targetVar == myTargetVar)
        # except:
        isMyTargetVar = (targetVar == 'myTargetVar')
        if isMyTargetVar == True:
            self.chk.update({
                    'myTargetVarName': tk.StringVar(),
                    'reaName': tk.StringVar(),
                    'modName': tk.StringVar(),
                    'myTargetVarMinAllowed': tk.StringVar(),
                    'myTargetVarMaxAllowed': tk.StringVar(),
                    'myTargetVarUnits': tk.StringVar(),
                    # 'myTargetVarIsGaussian': tk.StringVar(),
                    # 'treatAsAdditiveBy_DQM_and_QDM': tk.StringVar(),
                    'myTargetVarIsAdditive': tk.StringVar(),
                    })

        if isMyTargetVar == True:
            # myTargetVarName
            Label(root, text='Name:').grid(sticky="NE", column=icol, row=irow, padx=10); icol+=1
            myTargetVarName_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarName'], width=entriesW, justify='right', takefocus=False)
            try:
                myTargetVarName_Entry.insert(END, str(myTargetVarName))
            except:
                myTargetVarName_Entry.insert(END, '')
            myTargetVarName_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # reaName
            try:
                reaName = reaNames[targetVar]
            except:
                reaName = ''
            Label(root, text='Name in reanalysis netCDF:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            reaName_Entry = tk.Entry(root, textvariable=self.chk['reaName'], width=entriesW, justify='right', takefocus=False)
            reaName_Entry.insert(END, reaName)
            reaName_Entry.grid(sticky="W", column=icol, row=irow); irow+=1; icol-=1

            # modName
            try:
                modName = modNames[targetVar]
            except:
                modName = ''
            Label(root, text='Name in models netCDFs:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            modName_Entry = tk.Entry(root, textvariable=self.chk['modName'], width=entriesW, justify='right', takefocus=False)
            modName_Entry.insert(END, modName)
            modName_Entry.grid(sticky="W", column=icol, row=irow); irow+=1; icol-=1


        # hresPeriodFilename
        Label(root, text='Hres period filename:').grid(sticky="NE", column=icol, row=irow, padx=10); icol+=1
        hresPeriodFilename_Entry = tk.Entry(root, textvariable=self.chk['hresPeriodFilename'], width=entriesW, justify='right', takefocus=False)
        try:
            hresPeriodFilename_Entry.insert(END, hresPeriodFilename[targetVar])
        except:
            hresPeriodFilename_Entry.insert(END, '')
        hresPeriodFilename_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1


        if isMyTargetVar == True:

            # myTargetVarMinAllowed
            Label(root, text='Minimum value:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            myTargetVarMinAllowed_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarMinAllowed'], width=entriesW, justify='right', takefocus=False)
            try:
                myTargetVarMinAllowed_Entry.insert(END, str(myTargetVarMinAllowed))
            except:
                myTargetVarMinAllowed_Entry.insert(END, '')
            myTargetVarMinAllowed_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # myTargetVarMaxAllowed
            Label(root, text='Maximum value:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            myTargetVarMaxAllowed_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarMaxAllowed'], width=entriesW, justify='right', takefocus=False)
            try:
                myTargetVarMaxAllowed_Entry.insert(END, str(myTargetVarMaxAllowed))
            except:
                myTargetVarMaxAllowed_Entry.insert(END, '')
            myTargetVarMaxAllowed_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # myTargetVarUnits
            Label(root, text='Units:').grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            myTargetVarUnits_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarUnits'], width=entriesW, justify='right', takefocus=False)
            try:
                myTargetVarUnits_Entry.insert(END, str(myTargetVarUnits))
            except:
                myTargetVarUnits_Entry.insert(END, '')
            myTargetVarUnits_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # myTargetVarIsAdditive
            l = Label(root, text='Additive (A) / Multiplicative (M):')
            l.grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            CreateToolTip(l, 'Indicate whether biases and future change should be additive (A) or muliplicative (M)')
            myTargetVarIsAdditive_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarIsAdditive'], width=entriesW, justify='right', takefocus=False)
            try:
                if str(myTargetVarIsAdditive) == 'True':
                    aux = 'A'
                else:
                    aux = 'M'
                myTargetVarIsAdditive_Entry.insert(END, aux)
            except:
                myTargetVarIsAdditive_Entry.insert(END, '')
            myTargetVarIsAdditive_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # # myTargetVarIsGaussian
            # l = Label(root, text='Is gaussian:')
            # l.grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            # CreateToolTip(l, 'Set to True if your variable is gaussian and to False otherwise')
            # myTargetVarIsGaussian_Entry = tk.Entry(root, textvariable=self.chk['myTargetVarIsGaussian'], width=entriesW, justify='right', takefocus=False)
            # try:
            #     myTargetVarIsGaussian_Entry.insert(END, str(myTargetVarIsGaussian))
            # except:
            #     myTargetVarIsGaussian_Entry.insert(END, '')
            # myTargetVarIsGaussian_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1

            # # treatAsAdditiveBy_DQM_and_QDM
            # l = Label(root, text='Additive DQM/QDM:')
            # l.grid(sticky="E", column=icol, row=irow, padx=10); icol+=1
            # CreateToolTip(l, 'Set to True if your variable should be bias corrected additive when using DQM/QDM and to False otherwise\n'
            #                  'True is recommended in general, unless your variable is similar to precipitation, with a nongaussian\n'
            #                  'distribution and many zeros.')
            # treatAsAdditiveBy_DQM_and_QDM_Entry = tk.Entry(root, textvariable=self.chk['treatAsAdditiveBy_DQM_and_QDM'], width=entriesW, justify='right', takefocus=False)
            # try:
            #     treatAsAdditiveBy_DQM_and_QDM_Entry.insert(END, str(treatAsAdditiveBy_DQM_and_QDM))
            # except:
            #     treatAsAdditiveBy_DQM_and_QDM_Entry.insert(END, '')
            # treatAsAdditiveBy_DQM_and_QDM_Entry.grid(sticky="W", column=icol, row=irow); icol-=1; irow+=1


    def get(self):
        return self.chk


########################################################################################################################
class frameMethodsClass(ttk.Frame):

    def __init__(self, notebook, root, targetVar, isGaussian=True):

        self.cbuts = []
        self.chk_list = []
        if targetVar == 'pr':
            disabled_methods = ['MLR', 'MLR-ANA', 'MLR-WT']
        else:
            disabled_methods = ['GLM-LIN', 'GLM-EXP', 'GLM-CUB', 'WG-NMM']
            if targetVar == 'myTargetVar' and isGaussian != True:
                disabled_methods.append('PSDM')
                disabled_methods.append('WG-PDF')

        def add_method_to_chk_list(disabled_methods, methods_chk_list, targetVar, methodName, family, mode, fields, info, icol, irow):
            """This function adds all methods to a list. The checked variable will contain information about their status
            once the mainloop is finished"""

            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            for method_dict in methods:
                if (method_dict['var'], method_dict['methodName']) == (targetVar, methodName):
                    checked = tk.BooleanVar(value=True)

            # Enable/disable methods
            if methodName in disabled_methods:
                l = Label(root, text='     '+methodName, fg='darkgray')
                l.grid(sticky="W", column=icol, row=irow, padx=10)
            else:
                c = Checkbutton(root, text=methodName, variable=checked, takefocus=False)
                self.cbuts.append(c)
                CreateToolTip(c, info)
                c.grid(sticky="W", column=icol, row=irow, padx=10)
            # print('-------------------------')
            # print(targetVar, methodName, c["state"])
            # if methodName in disabled_methods:
            #     # c.config(state='disabled')
            #     self.cbuts[-1].config(state='disabled')
            # print(targetVar, methodName, c["state"])
            # print(len(self.cbuts))
            self.chk_list.append(
                {'var': targetVar, 'methodName': methodName, 'family': family, 'mode': mode, 'fields': fields,
                 'checked': checked})
            # return self.chk_list

        def select_all():
            for i in self.cbuts:
                i.select()

        def deselect_all():
            for i in self.cbuts:
                i.deselect()

        # Functions for selecting/deselecting all
        buttonWidth = 8
        icol, irow = 0, 0
        ttk.Label(root, text="").grid(column=icol, row=irow, padx=20, pady=0); icol += 1; irow += 1
        ttk.Label(root, text='Methods') .grid(sticky="W", column=icol, row=irow, padx=20, pady=(10, 10), columnspan=3); irow += 1

        # Raw
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'RAW', 'RAW', 'RAW', 'var', 'No downscaling, nearest gridpoint', icol, irow); icol += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'RAW-BIL', 'RAW', 'RAW', 'var', 'No downscaling, bilinear interpolation', icol, irow); irow += 1; icol -= 1

        # Model Output Statistics
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(root, text="Model Output Statistics:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'QM', 'MOS', 'MOS', 'var', 'Quantile Mapping', icol, irow); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'DQM', 'MOS', 'MOS', 'var', 'Detrended Quantile Mapping', icol, irow); icol += 1; irow -= 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'QDM', 'MOS', 'MOS', 'var', 'Quantile Delta Mapping', icol, irow); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'PSDM', 'MOS', 'MOS', 'var', '(Parametric) Scaled Distribution Mapping', icol, irow); icol -= 1; irow += 1

        # Analogs / Weather Typing
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(root, text="Analogs / Weather Typing:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-SYN-1NN', 'ANA', 'PP', 'saf', 'Nearest neighbour based on synoptic fields', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-SYN-kNN', 'ANA', 'PP', 'saf', 'k nearest neighbours based on synoptic fields', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-SYN-rand', 'ANA', 'PP', 'saf', 'Random neighbour based on synoptic fields', icol, irow);  irow-=2; icol+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-LOC-1NN', 'ANA', 'PP', 'pred+saf', 'Nearest neighbour based on combined synoptic and local analogies', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-LOC-kNN', 'ANA', 'PP', 'pred+saf', 'k nearest neighbours based on combined synoptic and local analogies', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-LOC-rand', 'ANA', 'PP', 'pred+saf', 'Random neighbour based on combined synoptic and local analogies', icol, irow); irow-=2; icol+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-VAR-1NN', 'ANA', 'PP', 'pcp', 'Nearest neighbour based on precipitation pattern', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-VAR-kNN', 'ANA', 'PP', 'pcp', 'k nearest neighbours based on precipitation pattern', icol, irow); irow+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANA-VAR-rand', 'ANA', 'PP', 'pcp', 'Random neighbour based on precipitation pattern', icol, irow); irow+=1; icol-=2

        # Linear methods
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(root, text="Linear methods:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'MLR', 'TF', 'PP', 'pred', 'Multiple Linear Regression', icol, irow); icol += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'MLR-ANA', 'ANA', 'PP', 'pred+saf', 'Multiple Linear Regression based on Analogs', icol, irow); icol += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'MLR-WT', 'ANA', 'PP', 'pred+saf', 'Multiple Linear Regression based on Weather Typing', icol, irow); irow += 1; icol-=2
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'GLM-LIN', 'TF', 'PP', 'pred', 'Generalized Linear Model (linear)', icol, irow); icol+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'GLM-EXP', 'TF', 'PP', 'pred', 'Generalized Linear Model (exponential)', icol, irow); icol+=1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'GLM-CUB', 'TF', 'PP', 'pred', 'Generalized Linear Model (cubic)', icol, irow); irow+=1; icol-=2

        # Machine Learning
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        ttk.Label(root, text="Machine Learning:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'SVM', 'TF', 'PP', 'pred', 'Support Vector Machine', icol, irow); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'LS-SVM', 'TF', 'PP', 'pred', 'Least Square Support Vector Machine', icol, irow); icol += 1; irow -= 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'RF', 'TF', 'PP', 'pred', 'Random Forest', icol, irow); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'XGB', 'TF', 'PP', 'pred', 'eXtreme Gradient Boost', icol, irow); icol += 1; irow -= 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'ANN', 'TF', 'PP', 'pred', 'Artificial Neural Network', icol, irow); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'CNN', 'TF', 'PP', 'pred', 'Convolutional Neural Network', icol, irow); icol -= 2; irow += 1

        # Weather Generators
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 2
        ttk.Label(root, text="Weather Generators:").grid(sticky="W", column=icol, row=irow, padx=30, columnspan=4); irow += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'WG-PDF', 'WG', 'WG', 'var', 'Weather generator from downscaled PDF', icol, irow); icol += 1
        add_method_to_chk_list(disabled_methods, self.chk_list, targetVar, 'WG-NMM', 'WG', 'WG', 'var', 'Weather generator Non-homogeneous Markov Model', icol, irow); icol -= 1; irow+=1

        # Select/deselect all
        # ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, padx=30); irow += 1
        frameButons = ttk.Frame(root)
        frameButons.grid(column=icol, row=irow, padx=10, pady=20, columnspan=3)
        Button(frameButons, text='Select all', command=select_all, width=buttonWidth, takefocus=False).grid(column=0, row=0)
        Button(frameButons, text='Deselect all', command=deselect_all, width=buttonWidth, takefocus=False).grid(column=1, row=0)


    def get(self):
        return self.chk_list


########################################################################################################################
class frameClimdexClass(ttk.Frame):

    def __init__(self, notebook, root, targetVar):

        self.chk_list = []

        def add_to_chk_list(chk_list, targetVar, climdex, info, icol, irow):

            # Initialize with default settings or last settings
            checked = tk.BooleanVar(value=False)
            try:
                if climdex in climdex_names[targetVar]:
                    checked = tk.BooleanVar(value=True)
            except:
                pass

            c = Checkbutton(root, text=climdex, variable=checked, takefocus=False)
            cbuts.append(c)
            CreateToolTip(c, info)
            c.grid(sticky="W", column=icol, row=irow, padx=10)
            self.chk_list.append({'var': targetVar, 'climdex': climdex, 'checked': checked})
            return self.chk_list

        # Functions for selecting/deselecting all
        cbuts = []
        buttonWidth = 8

        def select_all():
            for i in cbuts:
                i.select()

        def deselect_all():
            for i in cbuts:
                i.deselect()

        climdex_dict = {
            'tasmax': {
                'TXm': 'Mean value of daily maximum temperature',
                'TXx': 'Maximum value of daily maximum temperature',
                'TXn': 'Minimum value of daily maximum temperature',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p90': '90th percentile',
                'p10': '10th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                'TX99p': 'Percentage of days when TX > 99th percentile',
                'TX95p': 'Percentage of days when TX > 95th percentile',
                'TX90p': 'Percentage of days when TX > 90th percentile',
                'TX10p': 'Percentage of days when TX < 10th percentile',
                'TX5p': 'Percentage of days when TX < 5th percentile',
                'TX1p': 'Percentage of days when TX < 1st percentile',
                'SU': 'Number of summer days',
                'ID': 'Number of icing days',
                'WSDI': 'Warm spell duration index',
            },
            'tasmin': {
                'TNm': 'Mean value of daily minimum temperature',
                'TNx': 'Maximum value of daily minimum temperature',
                'TNn': 'Minimum value of daily minimum temperature',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p90': '90th percentile',
                'p10': '10th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                'TN99p': 'Percentage of days when TN > 99th percentile',
                'TN95p': 'Percentage of days when TN > 95th percentile',
                'TN90p': 'Percentage of days when TN > 90th percentile',
                'TN10p': 'Percentage of days when TN < 10th percentile',
                'TN5p': 'Percentage of days when TN < 5th percentile',
                'TN1p': 'Percentage of days when TN < 1st percentile',
                'FD': 'Number of frost days',
                'TR': 'Number of tropical nights',
                'CSDI': 'Cold spell duration index',
            },
            'tas': {
                'Tm': 'Mean value of daily mean temperature',
                'Tx': 'Maximum value of daily mean temperature',
                'Tn': 'Minimum value of daily mean temperature',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p90': '90th percentile',
                'p10': '10th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                'T99p': 'Percentage of days when T > 99th percentile',
                'T95p': 'Percentage of days when T > 95th percentile',
                'T90p': 'Percentage of days when T > 90th percentile',
                'T10p': 'Percentage of days when T < 10th percentile',
                'T5p': 'Percentage of days when T < 5th percentile',
                'T1p': 'Percentage of days when T < 1st percentile',
            },
            'pr': {
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
            },
            'uas': {
                'Um': 'Mean value of the zonal wind component',
                'Ux': 'Maximum value of the zonal wind component',
            },
            'vas': {
                'Vm': 'Mean value of the meridional wind component',
                'Vx': 'Maximum value of the meridional wind component',
            },
            'sfcWind': {
                'SFCWINDm': 'Mean value of the surface wind speed',
                'SFCWINDx': 'Maximum value of the surface wind speed',
            },
            'hurs': {
                'HRm': 'Mean relative humidity',
            },
            'clt': {
                'CLTm': 'Mean cloud cover',
            },
            'myTargetVar': {
                'm': 'Mean value',
                'x': 'Maximum value',
                'n': 'Minimum value',
                'p99': '99th percentile',
                'p95': '95th percentile',
                'p90': '90th percentile',
                'p10': '10th percentile',
                'p5': '5th percentile',
                'p1': '1st percentile',
                '99p_days': 'Percentage of days over the 99th percentile',
                '95p_days': 'Percentage of days over the 95th percentile',
                '90p_days': 'Percentage of days over the 90th percentile',
                '10p_days': 'Percentage of days under the 10th percentile',
                '5p_days': 'Percentage of days under the 5th percentile',
                '1p_days': 'Percentage of days under the 1st percentile',
            },
        }

        irow, icol = 0, 0

        ttk.Label(root, text="").grid(column=icol, row=irow, padx=30, pady=0); icol += 1; irow += 1
        ttk.Label(root, text='Climdex') .grid(sticky="W", column=icol, row=irow, padx=20, pady=(10, 10), columnspan=3); irow += 1

        nrows = 1
        colJumps = 0
        for climdex in climdex_dict[targetVar]:
            add_to_chk_list(self.chk_list, targetVar, climdex, climdex_dict[targetVar][climdex], icol, irow); irow += 1; nrows += 1
            if nrows == 17:
                icol += 1; irow -= nrows; nrows = 1; irow += 1; colJumps += 1
        irow = 18; icol -= colJumps

        # Select/deselect all
        ttk.Label(root, text="").grid(sticky="W", column=icol, row=irow, pady=0); irow += 1

        Button(root, text='Select all', command=select_all, width=buttonWidth, takefocus=False).grid(
            sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1
        Button(root, text='Deselect all', command=deselect_all, width=buttonWidth,
               takefocus=False).grid(sticky="W", column=icol, row=irow, padx=30, columnspan=3); irow += 1

    def get(self):
        return self.chk_list


########################################################################################################################
class tabTasmax(ttk.Frame):

    def __init__(self, notebook):
        targetVar = 'tasmax'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Temperature: max')

        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Maximum Temperature', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabTasmin(ttk.Frame):

    def __init__(self, notebook):
        targetVar = 'tasmin'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Temperature: min')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Minimum Temperature', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabTas(ttk.Frame):

    def __init__(self, notebook):
        targetVar = 'tas'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Temperature: mean')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Mean Temperature', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list



########################################################################################################################
class tabPr(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'pr'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Precipitation')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Precipitation', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabUas(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'uas'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Wind: u')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Zonal Wind Component', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabVas(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'vas'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Wind: v')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Meridional Wind Component', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list

########################################################################################################################
class tabSfcWind(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'sfcWind'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Wind: speed')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Wind Speed', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabHurs(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'hurs'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Humidity')


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Relative Humidity', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabClt(ttk.Frame):

    def __init__(self, notebook):

        targetVar = 'clt'
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=targetVar)
        # notebook.add(tab, text='Clouds')

        irow, icol = 0, 0
        Label(tab, text="Cloud Cover").grid(column=icol, row=irow, padx=100, pady=(20, 0), columnspan=100); irow+=1


        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='Cloud Cover', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, targetVar).get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, targetVar).get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, targetVar).get()

        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, targetVar).get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabMyTargetVar(ttk.Frame):

    def __init__(self, notebook):

        tab = ttk.Frame(notebook)
        notebook.add(tab, text='myTargetVar')

        try:
            targetVar = myTargetVar
        except:
            targetVar = 'myTargetVar'

        # Change myTargetVar to 'myTargetVar'
        for i in range(len(methods)):
            if methods[i]['var'] == myTargetVar:
                methods[i]['var'] = 'myTargetVar'
        try:
            hresPeriodFilename['myTargetVar'] = hresPeriodFilename.pop(myTargetVar)
            preds_dict['myTargetVar'] = preds_dict.pop(myTargetVar)
            climdex_names['myTargetVar'] = climdex_names.pop(myTargetVar)
        except:
            pass
        frames = []

        # Enable/disable targetVar
        irow, icol = 0, 0
        self.targetVar_active_var = tk.BooleanVar(value=False)
        if targetVar in targetVars:
            self.targetVar_active_var = tk.BooleanVar(value=True)
        c = Checkbutton(tab, text='User definded target variable', variable=self.targetVar_active_var,
                        command=lambda: enable(self.targetVar_active_var.get(), frames), takefocus=False)
        c.grid(column=icol, row=irow, padx=(100), pady=(20, 0), columnspan=100); irow+=1

        # framePredictors
        framePredictors = ttk.Frame(tab)
        frames.append(framePredictors)
        framePredictors.grid(row=irow, column=icol, sticky='n', padx=(40, 0))
        self.predictors_chk_list = []
        self.predictors_chk_list = framePredictorsClass(notebook, framePredictors, 'myTargetVar').get()

        # frameTargetVarInfo
        frameTargetVarInfo = ttk.Frame(tab)
        frames.append(frameTargetVarInfo)
        frameTargetVarInfo.grid(row=irow+1, column=icol, sticky='n')
        self.TargetVarInfo_chk_list = []
        self.TargetVarInfo_chk_list = frameTargetVarInfoClass(notebook, frameTargetVarInfo, 'myTargetVar').get()

        # frameMethods
        frameMethods = ttk.Frame(tab)
        frames.append(frameMethods)
        frameMethods.grid(row=irow, column=icol+1, sticky='n', rowspan=2)
        self.methods_chk_list = []
        # self.methods_chk_list = frameMethodsClass(notebook, frameMethods, 'myTargetVar',
        #                                           isGaussian=self.TargetVarInfo_chk_list['myTargetVarIsGaussian'].get()).get()
        self.methods_chk_list = frameMethodsClass(notebook, frameMethods, 'myTargetVar',
                                                  isGaussian=False).get()


        # frameClimdex
        frameClimdex = ttk.Frame(tab)
        frames.append(frameClimdex)
        frameClimdex.grid(row=irow, column=icol+2, sticky='n', rowspan=2)
        self.Climdex_chk_list = []
        self.Climdex_chk_list = frameClimdexClass(notebook, frameClimdex, 'myTargetVar').get()

        # Enabled/disbled by default
        enable(targetVar in targetVars, frames)

    def get(self):
        return self.targetVar_active_var, self.methods_chk_list, self.predictors_chk_list, self.TargetVarInfo_chk_list, self.Climdex_chk_list


########################################################################################################################
class tabFigures(ttk.Frame):

    def __init__(self, notebook):
        tabFigures = ttk.Frame(notebook)
        notebook.add(tabFigures, text='Figures')

        def open_figure(imgs):

            w = 600
            filename = '_'.join((self.fields[0], self.fields[1], self.fields[2], self.fields[3], self.fields[4],
                                 self.fields[5])) + '.png'
            try:
                text = '\n' + self.descriptions['_'.join((self.fields[0].split('-')[0], self.fields[1]))] + '\n'
            except:
                text = ''

            if os.path.isfile("../results/Figures/" + filename):

                imgs.append(Image.open("../results/Figures/" + filename))
                h = int(w * imgs[-1].height / imgs[-1].width)
                try:
                    imgs[-1] = imgs[-1].resize((w, h), Image.Resampling.LANCZOS)
                except:
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
                messagebox.showerror("pyClim-SDM",  "No figure has been generated matching the selection:\n" + filename)


        # frameFigSelection
        frameFigSelection = Frame(tabFigures, height=505, width=1140)
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
            'PRECONTROL_biasMap': 'Bias of one model compared to the reanalysis (in the mean value) in a historical '
                                      'period. For tmax/tmin absolute bias, for pcp relative bias and the rest '
                                      'standardized and absolute bias.',
            'PRECONTROL_evolSpaghetti': 'Evolution of one predictor by all models in the form of anomaly with respect to the '
                                        'reference period (absolute anomaly for tmax/tmin, relative anomaly for pcp and '
                                        'absolute anomaly of the standardized variables for the rest).',
            'PRECONTROL_qqPlot': 'QQ-plot for one variable by one model in historical vs. reanalysis.',
            'PRECONTROL_annualCycle': 'Annual cycle for one variable by all models in historical and reanalysis '
                                      '(monthly means for tmax/tmin, monthly accumulations for pcp and monthly means of '
                                      'the standardized variable for the rest). ',
            'PRECONTROL_changeMap': 'Change (abs/relative) in the mean value over a 30-year period by the middle and end '
                                   'of the century compared to the reference period.',
            'EVALUATION_annualCycle': 'Annual cycle for one variable, downscaled by all methods vs. observation '
                                      '(monthly means for tmax/tmin and monthly accumulations for pcp).',
            'EVALUATION_rmseBoxplot': 'RMSE of the daily series (downscaled and observed) by all methods. Boxes '
                                      'contain one value per grid point.',
            'EVALUATION_correlationBoxplot': 'Correlation (Pearson for temperature and Spearman for precipitation) of '
                                             'the daily series (downscaled and observed) by all methods. Boxes contain '
                                             'one value per grid point.',
            'EVALUATION_varianceBoxplot': 'Bias (relative, %) in the variance of the daily series (downscaled and '
                                          'observed) by all methods. Boxes contain one value per grid point.',
            'EVALUATION_qqPlot': 'QQ-plot for one variable by one method vs. observations.',
            'EVALUATION_r2Map': 'R2 score of the daily series (coefficient of determination) by one method.',
            'EVALUATION_rmseMap': 'RMSE of the daily series by one method.',
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
            'PROJECTIONS_spreadChangeMap': 'Standard deviation in the anomaly given by the multimodel ensemble (spread).',
            'PROJECTIONS_evolTrendRaw': 'Evolution graph, by one method vs. raw models, of one variable by the '
                                        'multimodel ensemble (the central line represents 50th percentile and the '
                                        'shaded area represents IQR), in the form of anomaly with respect to the '
                                        'reference period (absolute anomaly for tmax/tmin and relative anomaly for pcp).'
                                        'The number between brackets correspond to the number of models. Beware that, '
                                        'when using bias corrected SDMs, raw still correspong to not bias corrected RAW.'
                                        'Thus, if zero models are for RAW, that means that climdex without bias '
                                        'correction need to be calculated',
        }

        def callback_experiment(event):
            clear_comboboxes_from(1)
            self.fields[0] = self.experimentVar.get()

            def callback_figType(event):
                clear_comboboxes_from(2)
                self.fields[1] = self.figTypeVar.get()

                # Create label with description
                Label(frameFigSelection, text='').grid(sticky="W", column=1, row=5, padx=10, pady=20)
                text = 'Your current selection corresponds to: \n\n' + self.descriptions['_'.join((self.fields[0].split('-')[0], self.fields[1]))]
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
        if not os.path.exists('../results/Figures/'):
            os.makedirs('../results/Figures/')
        for file in os.listdir('../results/Figures/'):
            if file.endswith(".png") and file.split('_')[0] not in experiments:
                experiments.append(file.split('_')[0])

        # sort experiments
        ordered_experiments = ['PRECONTROL',
                               'EVALUATION',
                               'EVALUATION-BC-QM', 'EVALUATION-BC-DQM', 'EVALUATION-BC-QDM', 'EVALUATION-BC-PSDM',
                               'EVALUATION-BC-QM-s', 'EVALUATION-BC-DQM-s', 'EVALUATION-BC-QDM-s', 'EVALUATION-BC-PSDM-s',
                               'PROJECTIONS',
                               'PROJECTIONS-BC-QM', 'PROJECTIONS-BC-DQM', 'PROJECTIONS-BC-QDM', 'PROJECTIONS-BC-PSDM',
                               'PROJECTIONS-BC-QM-s', 'PROJECTIONS-BC-DQM-s', 'PROJECTIONS-BC-QDM-s',
                               'PROJECTIONS-BC-PSDM-s',
                               ]

        experiments = [x for x in ordered_experiments if x in experiments]

        self.experimentVar = tk.StringVar()
        combobox = ttk.Combobox(frameFigSelection, textvariable=self.experimentVar)
        combobox['values'] = experiments
        combobox['state'] = 'readonly'
        combobox.grid(sticky="W", column=1, row=4, padx=2, pady=10)
        combobox.bind('<<ComboboxSelected>>', callback_experiment)
        self.fields[0] = self.experimentVar.get()



        global imgs
        imgs = []
        Button(tabFigures, text="Open figure", width=10,
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

        # Tab: models
        self.chk_dict_models, self.otherModels_var, self.chk_dict_scenes, self.otherScenes_var, \
            self.reanalysisName_var_chk = tabModelsAndScenes(notebook).get()

        # Tab: dates
        self.calibration_years_chk, self.reference_years_chk, self.historical_years_chk, self.ssp_years_chk, self.bc_option_chk, \
            self.bc_method_chk, self.testing_years_dict_chk, self.reanalysisPeriodFilename_var_chk, self.historicalPeriodFilename_var_chk, \
            self.sspPeriodFilename_var_chk, self.split_mode_chk, self.seasons_chk = tabDates(notebook).get()

        # Tab: domain
        self.grid_res_var_chk, self.saf_lat_up_var_chk, self.saf_lon_left_var_chk, self.saf_lon_right_var_chk, \
            self.saf_lat_down_var_chk, self.reaNames, self.modNames, self.SAFs = tabDomain(notebook).get()

        # Tab: targetVars
        self.targetVars_dict = {}
        aux = tabTasmax(notebook).get()
        self.targetVars_dict.update({'tasmax': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabTasmin(notebook).get()
        self.targetVars_dict.update({'tasmin': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabTas(notebook).get()
        self.targetVars_dict.update({'tas': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabPr(notebook).get()
        self.targetVars_dict.update({'pr': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabUas(notebook).get()
        self.targetVars_dict.update({'uas': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabVas(notebook).get()
        self.targetVars_dict.update({'vas': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabSfcWind(notebook).get()
        self.targetVars_dict.update({'sfcWind': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabHurs(notebook).get()
        self.targetVars_dict.update({'hurs': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabClt(notebook).get()
        self.targetVars_dict.update({'clt': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})
        aux = tabMyTargetVar(notebook).get()
        self.targetVars_dict.update({'myTargetVar': {'active': aux[0], 'methods': aux[1], 'preds': aux[2], 'info': aux[3], 'climdex': aux[4], }})

        # Tab: visualization
        tabFigures(notebook)

        # Logo
        w = 120
        img = Image.open("../doc/pyClim-SDM_logo.png")
        h = int(w * img.height / img.width)
        try:
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        except:
            img = img.resize((w, h), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        canvas = Canvas(notebook, width=w, height=h)
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.grid(sticky="W", column=0, row=1, padx=10)

        self.run = False
        def run():

            self.all_checks_ok = False

            # Read experiment and steps
            self.experiment = self.experiment_chk.get()
            self.steps = []
            for step in self.all_steps[self.experiment]:
                if self.steps_dict[self.experiment][step].get() == True:
                    self.steps.append(step)

            # TargetVars
            self.targetVars = []
            for x in self.targetVars_dict:
                if self.targetVars_dict[x]['active'].get() == True:
                    self.targetVars.append(x)

            # Force to select at least one saf
            self.saf_list = [x for x in self.SAFs if self.SAFs[x].get() == True]
            if len(self.saf_list) == 0:
                self.all_checks_ok = False
                messagebox.showerror("pyClim-SDM",  "At least one Synoptic Analogy Field must be selected")
            else:
                self.all_checks_ok = True

            # Force to select at least one pred
            self.selected_all_preds = []
            self.preds_targetVars_dict = {}
            for x in self.targetVars_dict:
                aux = []
                for pred in self.targetVars_dict[x]['preds']:
                    if self.targetVars_dict[x]['preds'][pred].get() == True:
                        self.selected_all_preds.append(pred)
                        aux.append(pred)
                if len(aux) != 0:
                    self.preds_targetVars_dict.update({x: aux})
            if self.all_checks_ok == True:
                if len(self.selected_all_preds) == 0:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "At least one predictor must be selected")
                else:
                    self.all_checks_ok = True

            # Force at least one predictor for each targetVar
            for targetVar in self.targetVars:
                if self.all_checks_ok == True:
                    try:
                        npreds = len(self.preds_targetVars_dict[targetVar])
                    except:
                        npreds = 0
                    if npreds == 0:
                        self.all_checks_ok = False
                        messagebox.showerror("pyClim-SDM",  'No predictor has been selected for ' + targetVar + '')
                    else:
                        self.all_checks_ok = True

            # Check data availability for all targetVars
            for targetVar in self.targetVars:
                targetVarName = targetVar
                if targetVar == 'myTargetVar':
                    targetVarName = self.targetVars_dict[targetVar]['info']['myTargetVarName'].get()
                if self.all_checks_ok == True:
                    aux = self.targetVars_dict[targetVar]['info']['hresPeriodFilename'].get()
                    if not os.path.isfile(pathHres + targetVarName + '_hres_metadata.txt') or \
                            not os.path.isfile(pathHres + targetVarName + '_'+aux+'.txt'):
                        messagebox.showerror("pyClim-SDM",  'Missing hres data for variable ' + targetVarName + '.\n'
                           'Remove ' + targetVarName + ' from your selection or prepare the input_data/hres/ directory properly.')
                        self.all_checks_ok = False
                    else:
                        self.all_checks_ok = True

            # Force consistency between methods and experiment
            if self.all_checks_ok == True:
                if len(self.targetVars) == 0 and self.exp != 'PRECONTROL':
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  'For ' + self.exp + ' experiment, at least one method must be selected')
                else:
                    self.all_checks_ok = True

            # Methods
            self.methods = {}
            for targetVar in self.targetVars:
                aux = []
                for method in self.targetVars_dict[targetVar]['methods']:
                    if method['checked'].get() == True:
                        aux.append(method['methodName'])
                if len(aux) != 0:
                    self.methods.update({targetVar: aux})

            # Years
            ini, end = [], []
            for targetVar in self.targetVars:
                aux = self.targetVars_dict[targetVar]['info']['hresPeriodFilename'].get()
                try:
                    ini.append(int(aux.split('-')[0][:4]))
                    end.append(int(aux.split('-')[1][:4]))
                except:
                    pass
            ini = np.max(np.array(ini))
            end = np.min(np.array(end))
            self.aux_calibration_years = (int(self.calibration_years_chk[0].get()), int(self.calibration_years_chk[1].get()))
            self.aux_reference_years = (int(self.reference_years_chk[0].get()), int(self.reference_years_chk[1].get()))
            self.aux_historical_years = (int(self.historical_years_chk[0].get()), int(self.historical_years_chk[1].get()))
            self.aux_ssp_years = (int(self.ssp_years_chk[0].get()), int(self.ssp_years_chk[1].get()))
            self.all_years_hres = [x for x in range(ini, end+1)]
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
                    messagebox.showerror("pyClim-SDM",  "Calibration years not available by reanalysis or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
            if self.all_checks_ok == True:
                year = self.aux_calibration_years[1]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "Calibration years not available by reanalysis or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force reference years
            if self.all_checks_ok == True:
                year = self.aux_reference_years[0]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "Reference years not available by reanalysis, historical GCMs or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
            if self.all_checks_ok == True:
                year = self.aux_reference_years[1]
                if year not in self.all_years_hres or year not in self.all_years_reanalysis \
                        or year not in self.all_years_historical:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "Reference years not available by reanalysis, historical GCMs or hres data.\n"
                                                           "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force historical years
            if self.all_checks_ok == True:
                year = self.aux_historical_years[0]
                if year not in self.all_years_historical:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "Historical years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
                year = self.aux_historical_years[1]
                if year not in self.all_years_historical:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM",  "Historical years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True

            # Force ssp years
            if self.all_checks_ok == True:
                year = self.aux_ssp_years[0]
                if year not in self.all_years_ssp:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM", "SSP years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True
                year = self.aux_ssp_years[1]
                if year not in self.all_years_ssp:
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM", "SSP years selection out of file content.\n"
                                                       "Please, modify your selection.")
                else:
                    self.all_checks_ok = True


            # seasons
            if self.all_checks_ok == True:
                self.inverse_seasonNames = []
                for i in range(13):
                    seasonName = self.seasons_chk[i].get()
                    self.inverse_seasonNames.append(seasonName)
                    if seasonName == '':
                        self.all_checks_ok = False
                        messagebox.showerror("pyClim-SDM", "Month "+str(i)+" has no season defined.\n"
                                                           "Please, modify your selection.")

            # myTargetVar
            if self.all_checks_ok == True:
                if 'myTargetVar' in self.targetVars:
                    aux = self.targetVars_dict['myTargetVar']['info']['myTargetVarIsAdditive'].get()
                    if aux not in ['A', 'M']:
                        self.all_checks_ok = False
                        messagebox.showerror("pyClim-SDM", "Please, indicate whether your personal target variable "
                                                              "should be treated as additive (A) or multiplicative (M)")
                    else:
                        self.all_checks_ok = True


            # myTargetVar bc method
            if self.all_checks_ok == True:
                if 'myTargetVar' in self.targetVars and self.bc_option_chk.get() != 'No' and self.bc_method_chk.get() == 'PSDM':
                    self.all_checks_ok = False
                    messagebox.showerror("pyClim-SDM", "PSDM not allowed as bias correction method when using a user defined target variable")
                else:
                    self.all_checks_ok = True


            # # Force at least one step
            # if self.all_checks_ok == True:
            #     if len(self.steps) == 0:
            #         self.all_checks_ok = False
            #         messagebox.showerror("pyClim-SDM",  "At least one step must be selected")
            #     else:
            #         self.all_checks_ok = True

            # Force all checks ok
            if self.all_checks_ok == True:
                self.run = True

                # # Run .tmp_main
                # if platform.system() == 'Linux':
                #     subprocess.call(['xterm', '-e', 'python .tmp_main.py'])
                # else:
                #     root.destroy()
                #     os.system('python3 .tmp_main.py')

                root.destroy()

                # Years
                self.calibration_years = (self.calibration_years_chk[0].get(), self.calibration_years_chk[1].get())
                self.reference_years = (self.reference_years_chk[0].get(), self.reference_years_chk[1].get())
                self.historical_years = (self.historical_years_chk[0].get(), self.historical_years_chk[1].get())
                self.ssp_years = (self.ssp_years_chk[0].get(), self.ssp_years_chk[1].get())

                # split_mode and testing years
                self.split_mode = self.split_mode_chk.get()
                self.single_split_testing_years = (
                    self.testing_years_dict_chk['single_split'][0].get(),
                    self.testing_years_dict_chk['single_split'][1].get())
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

                # period filenames
                self.hresPeriodFilename = {}
                for targetVar in self.targetVars:
                    self.hresPeriodFilename.update({targetVar: self.targetVars_dict[targetVar]['info']['hresPeriodFilename'].get()})
                self.reanalysisName = self.reanalysisName_var_chk.get()
                self.reanalysisPeriodFilename = self.reanalysisPeriodFilename_var_chk.get()
                self.historicalPeriodFilename = self.historicalPeriodFilename_var_chk.get()
                self.sspPeriodFilename = self.sspPeriodFilename_var_chk.get()


                # reaNames and modNames
                for var in self.reaNames:
                    self.reaNames.update({var: self.reaNames[var].get()})
                for var in self.modNames:
                    self.modNames.update({var: self.modNames[var].get()})

                # climdex
                self.climdex_names = {}
                for targetVar in self.targetVars:
                    aux = []
                    for x in self.targetVars_dict[targetVar]['climdex']:
                        if x['checked'].get() == True:
                            aux.append(x['climdex'])
                    self.climdex_names.update({targetVar: aux})

                # Bias correction
                self.bc_option_str = self.bc_option_chk.get()
                if self.bc_option_str == 'No':
                    self.apply_bc, self.apply_bc_bySeason = False, False
                elif self.bc_option_str == 'Yes':
                    self.apply_bc, self.apply_bc_bySeason = True, False
                elif self.bc_option_str == 'By season':
                    self.apply_bc, self.apply_bc_bySeason = True, True
                self.bc_method = self.bc_method_chk.get()


                if 'myTargetVar' in self.targetVars:
                    info = self.targetVars_dict['myTargetVar']['info']
                    self.myTargetVarName = info['myTargetVarName'].get()
                    self.myTargetReaName = info['reaName'].get()
                    self.myTargetModName = info['modName'].get()
                    self.myTargetVarMinAllowed = info['myTargetVarMinAllowed'].get()
                    self.myTargetVarMaxAllowed = info['myTargetVarMaxAllowed'].get()
                    self.myTargetVarUnits = info['myTargetVarUnits'].get()
                    # self.myTargetVarIsGaussian = info['myTargetVarIsGaussian'].get()
                    aux = info['myTargetVarIsAdditive'].get()
                    if aux == 'A':
                        self.myTargetVarIsAdditive = True
                    else:
                        self.myTargetVarIsAdditive = False
                    # self.treatAsAdditiveBy_DQM_and_QDM = info['treatAsAdditiveBy_DQM_and_QDM'].get()
                    self.reaNames.update({'myTargetVar': self.myTargetReaName})
                    self.modNames.update({'myTargetVar': self.myTargetModName})
                else:
                    self.myTargetVarName = ''
                    self.myTargetReaName = ''
                    self.myTargetModName = ''
                    self.myTargetVarMinAllowed = None
                    self.myTargetVarMaxAllowed = None
                    self.myTargetVarUnits = ''
                    # self.myTargetVarIsGaussian = False
                    self.myTargetVarIsAdditive = False
                    # self.treatAsAdditiveBy_DQM_and_QDM = False
                if self.myTargetVarMinAllowed == '':
                    self.myTargetVarMinAllowed = None
                if self.myTargetVarMaxAllowed == '':
                    self.myTargetVarMaxAllowed = None

                # print(self.experiment)
                # print(self.steps)
                # print(self.targetVars)
                # print(self.methods)
                # print(self.calibration_years)
                # print(self.single_split_testing_years)
                # print(self.fold1_testing_years)
                # print(self.fold2_testing_years)
                # print(self.fold3_testing_years)
                # print(self.fold4_testing_years)
                # print(self.fold5_testing_years)
                # print(self.split_mode)
                # print(self.reference_years)
                # print(self.historical_years)
                # print(self.ssp_years)
                # print(self.hresPeriodFilename)
                # print(self.reanalysisName)
                # print(self.reanalysisPeriodFilename)
                # print(self.historicalPeriodFilename)
                # print(self.sspPeriodFilename)
                # print(self.grid_res)
                # print(self.saf_lat_up)
                # print(self.saf_lat_down)
                # print(self.saf_lon_left)
                # print(self.saf_lon_right)
                # print(self.reaNames)
                # print(self.modNames)
                # print(self.preds_targetVars_dict)
                # print(self.saf_list)
                # print(self.scene_names_list)
                # print(self.model_names_list)
                # print(self.climdex_names)
                # print(self.apply_bc, self.apply_bc_bySeason, self.bc_method)
                #
                # print(self.myTargetVarName)
                # print(self.myTargetVarMinAllowed)
                # print(self.myTargetVarMaxAllowed)
                # print(self.myTargetVarUnits)
                # print(self.myTargetVarIsGaussian)
                # print(self.myTargetVarIsAdditive)
                # print(self.treatAsAdditiveBy_DQM_and_QDM)
                # exit()

                # Write settings file
                write_settings_file(self.showWelcomeMessage, self.experiment, self.targetVars, self.steps, self.methods,
                                    self.calibration_years, self.single_split_testing_years,
                                    self.fold1_testing_years, self.fold2_testing_years, self.fold3_testing_years,
                                    self.fold4_testing_years, self.fold5_testing_years, self.split_mode,
                                    self.reference_years, self.historical_years, self.ssp_years,
                                    self.hresPeriodFilename, self.reanalysisName, self.reanalysisPeriodFilename,
                                    self.historicalPeriodFilename, self.sspPeriodFilename,
                                    self.grid_res, self.saf_lat_up, self.saf_lat_down, self.saf_lon_left, self.saf_lon_right,
                                    self.reaNames, self.modNames, self.preds_targetVars_dict, self.saf_list,
                                    self.scene_names_list, self.model_names_list, self.climdex_names,
                                    self.apply_bc, self.apply_bc_bySeason, self.bc_method,
                                    self.myTargetVarName, self.myTargetVarMinAllowed, self.myTargetVarMaxAllowed,
                                    self.myTargetVarUnits, self.myTargetVarIsAdditive,
                                    # self.myTargetVarIsGaussian,self.treatAsAdditiveBy_DQM_and_QDM
                                    )

                # Write tmp_main file
                write_tmpMain_file(self.steps)

                os.system('python3 .tmp_main.py')

                # Delete tmp_main
                try:
                    os.remove('.tmp_main.py')
                except:
                    pass

        # Run butnon
        frame = Frame(notebook)
        frame.grid(sticky="SE", column=0, row=0, padx=560, pady=272)
        Button(notebook, text="Run", width=10, command=run).grid(sticky="W", column=2, row=1, padx=20, pady=0)

        # Mainloop
        root.mainloop()



########################################################################################################################
def write_settings_file(showWelcomeMessage, experiment, targetVars, steps, methods,
                                calibration_years, single_split_testing_years,
                                fold1_testing_years, fold2_testing_years, fold3_testing_years,
                                fold4_testing_years, fold5_testing_years, split_mode,
                                reference_years, historical_years, ssp_years,
                                hresPeriodFilename, reanalysisName, reanalysisPeriodFilename,
                                historicalPeriodFilename, sspPeriodFilename,
                                grid_res, saf_lat_up, saf_lat_down, saf_lon_left, saf_lon_right,
                                reaNames, modNames, preds_targetVars_dict, saf_list,
                                scene_names_list, model_names_list, climdex_names,
                                apply_bc, apply_bc_bySeason, bc_method,
                                myTargetVarName, myTargetVarMinAllowed, myTargetVarMaxAllowed,
                                myTargetVarUnits, myTargetVarIsAdditive,
                                # myTargetVarIsGaussian, treatAsAdditiveBy_DQM_and_QDM
                        ):

    """This function prepares a new settings file with the user selected options"""

    # Open f for writing
    f = open('../config/settings.py', "w")

    # Write new settings
    f.write("showWelcomeMessage = " + str(showWelcomeMessage) + "\n")
    f.write("experiment = '" + str(experiment) + "'\n")
    f.write("targetVars = " + str(targetVars) + "\n")
    f.write("methods = " + str(methods) + "\n")
    f.write("calibration_years = (" + str(calibration_years[0]) + ", " + str(calibration_years[1]) + ")\n")
    f.write("single_split_testing_years = (" + str(single_split_testing_years[0]) + ", " + str(single_split_testing_years[1]) + ")\n")
    f.write("fold1_testing_years = (" + str(fold1_testing_years[0]) + ", " + str(fold1_testing_years[1]) + ")\n")
    f.write("fold2_testing_years = (" + str(fold2_testing_years[0]) + ", " + str(fold2_testing_years[1]) + ")\n")
    f.write("fold3_testing_years = (" + str(fold3_testing_years[0]) + ", " + str(fold3_testing_years[1]) + ")\n")
    f.write("fold4_testing_years = (" + str(fold4_testing_years[0]) + ", " + str(fold4_testing_years[1]) + ")\n")
    f.write("fold5_testing_years = (" + str(fold5_testing_years[0]) + ", " + str(fold5_testing_years[1]) + ")\n")
    f.write("split_mode = '" + str(split_mode) + "'\n")
    f.write("reference_years = (" + str(reference_years[0]) + ", " + str(reference_years[1]) + ")\n")
    f.write("historical_years = (" + str(historical_years[0]) + ", " + str(historical_years[1]) + ")\n")
    f.write("ssp_years = (" + str(ssp_years[0]) + ", " + str(ssp_years[1]) + ")\n")
    f.write("hresPeriodFilename = " + str(hresPeriodFilename) + "\n")
    f.write("reanalysisName = '" + str(reanalysisName) + "'\n")
    f.write("reanalysisPeriodFilename = '" + str(reanalysisPeriodFilename) + "'\n")
    f.write("historicalPeriodFilename = '" + str(historicalPeriodFilename) + "'\n")
    f.write("sspPeriodFilename = '" + str(sspPeriodFilename) + "'\n")
    f.write("grid_res = " + str(grid_res) + "\n")
    f.write("saf_lat_up = " + str(saf_lat_up) + "\n")
    f.write("saf_lon_left = " + str(saf_lon_left) + "\n")
    f.write("saf_lon_right = " + str(saf_lon_right) + "\n")
    f.write("saf_lat_down = " + str(saf_lat_down) + "\n")
    f.write("reaNames = " + str(reaNames) + "\n")
    f.write("modNames = " + str(modNames) + "\n")
    f.write("preds_targetVars_dict = " + str(preds_targetVars_dict) + "\n")
    f.write("saf_list = " + str(saf_list) + "\n")
    f.write("scene_names_list = " + str(scene_names_list) + "\n")
    f.write("model_names_list = " + str(model_names_list) + "\n")
    f.write("climdex_names = " + str(climdex_names) + "\n")
    f.write("inverse_seasonNames = " + str(inverse_seasonNames) + "\n")
    f.write("apply_bc = " + str(apply_bc) + "\n")
    f.write("apply_bc_bySeason = " + str(apply_bc_bySeason) + "\n")
    f.write("bc_method = '" + str(bc_method) + "'\n")

    f.write("myTargetVarName = '" + str(myTargetVarName) + "'\n")
    f.write("myTargetVarMinAllowed = " + str(myTargetVarMinAllowed) + "\n")
    f.write("myTargetVarMaxAllowed = " + str(myTargetVarMaxAllowed) + "\n")
    f.write("myTargetVarUnits = '" + str(myTargetVarUnits) + "'\n")
    # f.write("myTargetVarIsGaussian = " + str(myTargetVarIsGaussian) + "\n")
    f.write("myTargetVarIsAdditive = " + str(myTargetVarIsAdditive) + "\n")
    # f.write("treatAsAdditiveBy_DQM_and_QDM = " + str(treatAsAdditiveBy_DQM_and_QDM) + "\n")


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
    f.write("    aux_lib.initial_checks()\n")

    if 'check_var_units' in steps:
        f.write("    aux_lib.check_var_units()\n")
    if 'preprocess' in steps:
        f.write("    preprocess.preprocess()\n")
    if 'missing_data_check' in steps:
        f.write("    precontrol.missing_data_check()\n")
    if 'predictors_correlation' in steps:
        f.write("    precontrol.predictors_correlation()\n")
    if 'GCMs_evaluation' in steps:
        f.write("    precontrol.GCMs_evaluation()\n")
    if 'train_methods' in steps:
        f.write("    preprocess.train_methods()\n")
    if 'downscale' in steps:
        f.write("    process.downscale()\n")
    if 'bias_correction' in steps:
        f.write("    postprocess.bias_correction()\n")
    if 'calculate_climdex' in steps:
        f.write("    postprocess.get_climdex()\n")
    if 'plot_results' in steps:
        f.write("    postprocess.plot_results()\n")
    if 'nc2ascii' in steps:
        f.write("    postprocess.nc2ascii()\n")

    f.write("\n")
    f.write("if __name__ == '__main__':\n")
    f.write("    start = datetime.datetime.now()\n")
    f.write("    main()\n")
    f.write("    end = datetime.datetime.now()\n")
    if running_at_HPC == True:
        f.write("    pyClim-SDM has finished, but submited jobs can be still running\n")
        f.write("    Do not launch more jobs until they have succesfully finished\n")
    f.write("    print('Elapsed time: ' + str(end - start))\n")

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