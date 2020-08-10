========================================================================
    Modgen 12 Case-based model : ModgenExample Project Overview
========================================================================

The Modgen 12 Case-based model wizard has created this ModgenExample application for you.  
This file contains a summary of what you will find in each of the files that
make up your ModgenExample application, as well as instructions on what to do next.


ModgenExample.vcxproj
    This is the main project file for VC++ projects.  The settings for the project were 
    modified specifically for Modgen 12 models.  Although they can be modified, it is recommended
    that only experienced developer do so.


Files included in the project have been group under 3 filters:

- C++ Files
- Modules (mpp)
- Scenarios

Modules (mpp)

    Mpp Files are the modules of the model and contain the actual Modgen code.  Each mpp file 
    constitues a module of the model.  Run Modgen 12 through the "Run Modgen 12" toolbar button
    or item of the menu "Tools" to generate corresponding C++ files before compiling.

C++ FILES
 
    This filter contains the extra C++ files that are needed for your model to compile and run. 
	You can add	any other custom cpp files to this filter to have them compiled along with the model.

SCENARIOS

    Scenario files are included so that the model developer has a basic scenario to start 
    using the model once it is compiled.  Although they have been included in the C++ project,
    they are not used to compile the model.  Therefore, model developers can choose to remove
    those files from the C++ project.

/////////////////////////////////////////////////////////////////////////////
The Modgen 12 Case-based model wizard has created the following modules:

ModgenExample.mpp

    This module contains core simulation functions and definitions.

PersonCore.mpp

    This module contains the basic information which defines the Person case.

/////////////////////////////////////////////////////////////////////////////
The Modgen 12 Case-based model wizard has created the following scenario files:

Base.scex

   This file contains the settings of the basic scenario included with the model.  To view and
   edit this file, it is recommended that model developers first compile their model and 
   then use the model's interface.  To open the scenario in the model, use the "Scenario / Open" 
   menu in the model executable.  Once the scenario is opened, settings can be viewed and
   modified using the "Scenario / Settings" menu item.  Refer to the Modgen 12 User's Guide 
   for more information.

Base(PersonCore).dat

   This file contains values for the parameters declared in the PersonCore module.
   These values can be viewed and modified through the Visual Studio environment, or through
   the model interface once the model is compiled.  To see and edit parameter values in the
   model interface, open the scenario and double-click on the parameter in the left pane.


/////////////////////////////////////////////////////////////////////////////
What to do next

- Build the model: use the "Build" command or "Batch Build" to create both debug and release
  versions of the model

- run the simulation: open the model executable.  Open the scenario using the "Scenario / Open"
  menu item.  Run the simulation using the "Scenario / Run/resume" menu item or corresponding toolbar
  button.

Or

- edit the model: developers can modify existing modules or add new ones.  To add a new module,
  developers are encouraged to use the new Modgen 12 module wizard which is available through
  the "Add new item" dialog in Visual Studio 2013.

/////////////////////////////////////////////////////////////////////////////
