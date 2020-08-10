# [Veusz 2.1](https://veusz.github.io)

Veusz is a scientific plotting package.  It is designed to produce
publication-ready PDF or SVG output. Graphs are built-up by combining
plotting widgets. The user interface aims to be simple, consistent and
powerful.

Veusz provides GUI, Python module, command line, scripting, DBUS and
SAMP interfaces to its plotting facilities. It also allows for
manipulation and editing of datasets. Data can be captured from
external sources such as Internet sockets or other programs.

## Changes in 2.1:
  * If file cannot be imported on document load, ask user for a new filename
  * Only open HDF5 files in readonly mode
  * Do not simply log errors when loading data with ImportPlugins
  * Cleanup of stylesheet, setting and widget code
  * Fix filename chooser and embedding in ImageFile widget
  * Force C++11 compilation on Unix
  * Fix document reload, export warning and unsafe loading dialogs
  * Parameterize number of line steps in covariance widget
  * Report error line number for csv reading exceptions
  * Convert values from expressions to 1D arrays when required
  * Take account of endsize setting for bar plot error bars
  * Remove dependence on sipconfig in build and add sip build parameters

## Features of package:

### Plotting features:
  * X-Y plots (with errorbars)
  * Line and function plots
  * Contour plots
  * Images (with colour mappings and colorbars)
  * Stepped plots (for histograms)
  * Bar graphs
  * Vector field plots
  * Box plots
  * Polar plots
  * Ternary plots
  * Plotting dates
  * Fitting functions to data
  * Stacked plots and arrays of plots
  * Nested plots
  * Plot keys
  * Plot labels
  * Shapes and arrows on plots
  * LaTeX-like formatting for text
  * Multiple axes
  * Axes with steps in axis scale (broken axes)
  * Axis scales using functional forms
  * Plotting functions of datasets

### Input and output:
  * PDF/EPS/PNG/SVG/EMF export
  * Dataset creation/manipulation
  * Embed Veusz within other programs
  * Text, HDF5, CSV, FITS, NPY/NPZ, QDP, binary and user-plugin importing
  * Data can be captured from external sources

### Extending:
  * Use as a Python module
  * User defined functions, constants and can import external Python functions
  * Plugin interface to allow user to write or load code to
    - import data using new formats
    - make new datasets, optionally linked to existing datasets
    - arbitrarily manipulate the document
  * Scripting interface
  * Control with DBUS and SAMP

### Other features:
  * Data filtering and manipulation
  * Data picker
  * Interactive tutorial
  * Multithreaded rendering

## Requirements for source install:
  * [Python](https://www.python.org/) 2.x (2.6 or greater required) or 3.x (3.3 or greater required)   
  * [Qt](https://www.qt.io/) >= 5.2 (free edition)
  * [PyQt](http://www.riverbankcomputing.co.uk/software/pyqt/) >= 5.2  (Qt and SIP is required to be installed first)
  * [SIP](http://www.riverbankcomputing.co.uk/software/sip/) >= 4.15
  * [Numpy](http://numpy.scipy.org/) >= 1.0

## Optional requirements:
* [h5py](http://www.h5py.org/) (optional for HDF5 support)
* [pyemf](http://pyemf.sourceforge.net/) >= 2.0.0 (optional for EMF export)
  - [Python 3 port in development](https://github.com/jeremysanders/pyemf)
* [iminuit](https://github.com/iminuit/iminuit) or PyMinuit >= 1.12 (optional improved fitting)
* [dbus-python](http://dbus.freedesktop.org/doc/dbus-python/), for dbus interface
* [astropy](http://www.astropy.org/) (optional for VO table import or FITS import)
* [SAMPy](http://pypi.python.org/pypi/sampy/) or astropy >= 0.4 (optional for SAMP support)
* [Ghostscript](https://www.ghostscript.com/) (for EPS/PS output)   

## License
Veusz is Copyright (C) 2003-2017 Jeremy Sanders <jeremy@jeremysanders.net>
 and contributors.
It is licensed under the GPL (version 2 or greater).

The latest source code can be found in [this GitHub repository](https://github.com/veusz/veusz).

