# fit.py
# fitting plotter

#    Copyright (C) 2005 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

import re
import sys

import numpy as N

from .. import document
from .. import setting
from .. import utils
from .. import qtall as qt

from .function import FunctionPlotter
from . import widget

# try importing iminuit first, then minuit, then None
try:
    import iminuit as minuit
except ImportError:
    try:
        import minuit
    except ImportError:
        minuit = None

# Check whether iminuit version is old (1.x)
if minuit is not None:
    if minuit.__version__[0:1] == '1':
        isiminuit1 = True
    else:
        isiminuit1 = False

def _(text, disambiguation=None, context='Fit'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

#SUGGESTION - simple function to parse a dict of PARAMETER[low_limit,high_limit]:VALUE to PARAMETER:(low_limit,high_limit) and PARAMETER:VALUE dicts
def valsExtractNamesLimits(vals):
    retvals = {}
    retlimits = {}
    paramvals = dict(vals)
    for key in paramvals:
        #remove extraneous spaces from key name
        key=key.replace(" ","")
        #match key with PARAMETER[low_limit,high_limit], limits can include 'inf', 'nan', etc
        km=re.match('(\w+)\[([\d\.\-infeEna\+]+),([\d\.\-infeEna\+]+)\]',key) 
        #record low and high limits
        if km != None:
            retvals[km[1]]=paramvals[key]
            try:
              retlimits[km[1]]=(float(km[2]),float(km[3]))
            except ValueError:
              retlimits[km[1]]=(-N.inf,N.inf)
        else:
            #no limits defined, therefore implicitly -inf,inf
            retvals[key]=paramvals[key]
            retlimits[key]=(-N.inf,N.inf)
    return retvals,retlimits

#SUGGESTION - accepting dict of initial parameter values where parameter names also contain [low_limit,high_limit] definitions
def minuitFit(evalfunc, params, names, inivalues, xvals, yvals, yserr):
    """Do fitting with minuit (if installed)."""

    def chi2(params):
        """generate a lambda function to impedance-match between PyMinuit's
        use of multiple parameters versus our use of a single numpy vector."""
        ef = evalfunc(params, xvals) #SUGGESTION
        #c = ((evalfunc(params, xvals) - yvals)**2/yserr**2 ).sum()   ##JAI ##SUGGESTIOn removed /yserr**2 -- introduced signifciant artefact, prob /1e-8**2=/1e-16**2  must be addressed!!
        if chi2.iters < 1 or yserr is None:
            c = ((ef - yvals)**2 ).sum() #SUGGESTION - first iteration no weighting, as per GraphPad Prism 5+
        else:
            c = ((ef - yvals)**2/ef**2 ).sum() #SUGGESTION - subsequent iterations weighted as Ycurve**2 as per GraphPad Prism 5+
        if chi2.runningFit:
            chi2.iters += 1
            p = [chi2.iters, c] + params.tolist()
            str = ("%5i " + "%8g " * (len(params)+1)) % tuple(p)
            print(str)

        return c

    #SUGGESTION - hack so this comparison does not need to be made during iteration
    if yserr.sum()==0.:
        yserr = None

    namestr = ', '.join(names)
    fnstr = 'lambda %s: chi2(N.array([%s]))' % (namestr, namestr)

    # this is safe because the only user-controlled variable is len(names)
    fn = eval(fnstr, {'chi2' : chi2, 'N' : N})

    #SUGGESTION - extract limits from parameter definitions, if present
    values,limits=valsExtractNamesLimits(inivalues)

    print(_('Fitting via Minuit:'))
    m = minuit.Minuit(fn, **values)

    # set errordef explicitly (least-squares: 1.0 or log-likelihood: 0.5) 
    m.errordef = 1.0
    
    #SUGGESTION - set lower convergence tolerance, default 0.1
    m.tol = 1e-3
    
    #SUGGESTION - careful strategy
    m.strategy = 2

    #SUGGESTION - if limits defined, pass them to the minuit object
    if limits != None:
        for key in limits:
            m.limits[key]=limits[key]
            print(_("Parameter %s defined with limits [%g:%g]\n" % (key,*limits[key])))

    # run the fit
    chi2.runningFit = True
    chi2.iters = 0
    m.migrad()

    # do some error analysis
    have_symerr, have_err = False, False
    try:
        chi2.runningFit = False
        m.hesse()
        have_symerr = True
        m.minos()
        have_err = True
    except Exception as e:
        print(e)
        if str(e).startswith('Discovered a new minimum'):
            # the initial fit really failed
            raise

    # print the results
    retchi2 = m.fval
    dof = len(yvals) - len(params)
    redchi2 = retchi2 / dof
    
    if have_err:
        if isiminuit1:
            results = ["    %s = %g \u00b1 %g (+%g / %g)" % (
                n, m.values[n], m.errors[n], m.merrors[(n, 1.0)],
                m.merrors[(n, -1.0)]) for n in names]
        else:
            results = ["    %s = %g \u00b1 %g (+%g / %g)" % (
                n, m.values[n], m.errors[n], m.merrors[n].upper,
                m.merrors[n].lower) for n in names]
        print(_('Fit results:\n') + "\n".join(results))
    elif have_symerr:
        print(_('Fit results:\n') + "\n".join([
            "    %s = %g \u00b1 %g" % (n, m.values[n], m.errors[n])
            for n in names]))
        print(_('MINOS error estimate not available.'))
    else:
        print(_('Fit results:\n') + "\n".join([
            '    %s = %g' % (n, m.values[n]) for n in names]))
        print(_('No error analysis available: fit quality uncertain'))

    print("chi^2 = %g, dof = %i, reduced-chi^2 = %g" % (retchi2, dof, redchi2))

    #SUGGESTION - reconstruct output parameter names with [low_limit,high_limit] limit definitions
    if limits != None:
      vals = {("%s[%g,%g]" % (name,*limits[name]) if name in limits else name):m.values[name] for name in names}
    else:
      vals = {name:m.values[name] for name in names}

    #SUGGESTION - calculate standard errors
    if have_err or have_symerr:
      chi2.runningFit = False
      cov=N.array(m.covariance)
      try:
        diag=N.diag(cov)
        diagSS=diag*chi2(params)/dof
        SE=N.sqrt(diagSS)
        print('Standard errors')
        print(SE)
        errs={names[idx]:float(SE[idx]) for idx in range(names.__len__())}
      except ValueError:
        errs={name:-1. for name in names}
    else:
      errs={name:-1. for name in names}
    ##
    
    return vals, retchi2, dof, errs

class Fit(FunctionPlotter):
    """A plotter to fit a function to data."""

    typename='fit'
    allowusercreation=True
    description=_('Fit a function to data')

    def __init__(self, parent, name=None):
        FunctionPlotter.__init__(self, parent, name=name)

        self.addAction( widget.Action(
            'fit', self.actionFit,
            descr=_('Fit function'),
            usertext=_('Fit function')) )

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        FunctionPlotter.addSettings(s)

        s.add( setting.FloatDict(
            'values',
            {'a': 0.0, 'b': 1.0},
            descr=_('Variables and fit values'),
            usertext=_('Parameters')), 1 )
        s.add( setting.DatasetExtended(
            'xData', 'x',
            descr=_('X data to fit (dataset name, list of values or expression)'),
            usertext=_('X data')), 2 )
        s.add( setting.DatasetExtended(
            'yData', 'y',
            descr=_('Y data to fit (dataset name, list of values or expression)'),
            usertext=_('Y data')), 3 )
        s.add( setting.Bool(
            'fitRange', False,
            descr=_(
                'Fit only the data between the minimum and maximum '
                'of the axis for the function variable'),
            usertext=_('Fit only range')), 4 )
        s.add( setting.WidgetChoice(
            'outLabel', '',
            descr=_('Write best fit parameters to this text label after fitting'),
            widgettypes=('label',),
            usertext=_('Output label')), 5 )
        s.add( setting.Str(
            'outExpr', '',
            descr=_('Output best fitting expression'),
            usertext=_('Output expression')),
            6, readonly=True )
        s.add( setting.Float(
            'chi2', -1,
            descr='Output chi^2 from fitting',
            usertext=_('Fit &chi;<sup>2</sup>')),
            7, readonly=True )
        s.add( setting.Int(
            'dof', -1,
            descr=_('Output degrees of freedom from fitting'),
            usertext=_('Fit d.o.f.')),
            8, readonly=True )
        s.add( setting.Float(
            'redchi2', -1,
            descr=_('Output reduced-chi-squared from fitting'),
            usertext=_('Fit reduced &chi;<sup>2</sup>')),
            9, readonly=True )
        #SUGGESTION - saveable representation of errors of fitting parameters
        s.add( setting.FloatDict(
            'errors',
            {'a': 0.0, 'b': 1.0},
            descr=_('Errors of fits'),
            usertext=_('Errors')), 10 )
        #SUGGESTION - choice of weighting
        s.add( setting.Bool(
            'weightYsquared', False,
            descr=_(
                'Weight data points by 1/Y<sup>2</sup>'),
            usertext=_('Weight 1/Y<sup>2</sup>')), 11 )

        f = s.get('function')
        f.newDefault('a + b*x')
        f.descr = _('Function to fit')

        # modify description
        s.get('min').usertext=_('Min. fit range')
        s.get('max').usertext=_('Max. fit range')

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Update range with range of data."""
        dataname = {'sx': 'xData', 'sy': 'yData'}[depname]
        data = self.settings.get(dataname).getData(self.document)
        if data:
            drange = data.getRange()
            if drange:
                axrange[0] = min(axrange[0], drange[0])
                axrange[1] = max(axrange[1], drange[1])

    def initEnviron(self):
        """Copy data into environment."""
        env = self.document.evaluate.context.copy()
        #SUGGESTION - if the PARAMETER[low_limit,high_limit] syntax is used these must be stripped for the parameters to be properly matched to the function
        values,limits=valsExtractNamesLimits(self.settings.values)
        env.update( values )
        return env

    def updateOutputLabel(self, ops, vals, chi2, dof):
        """Use best fit parameters to update text label."""
        s = self.settings
        labelwidget = s.get('outLabel').findWidget()
        #SUGGESTION - if the PARAMETER[low_limit,high_limit] syntax is used these must be stripped
        vals,limits=valsExtractNamesLimits(vals)
        if labelwidget is not None:
            # build up a set of X=Y values
            loc = self.document.locale
            txt = []
            for l, v in sorted(vals.items()):
                val = utils.formatNumber(v, '%.4Vg', locale=loc)
                txt.append( '%s = %s' % (l, val) )
            # add chi2 output
            txt.append( r'\chi^{2}_{\nu} = %s/%i = %s' % (
                utils.formatNumber(chi2, '%.4Vg', locale=loc),
                dof,
                utils.formatNumber(chi2/dof, '%.4Vg', locale=loc) ))

            # update label with text
            text = r'\\'.join(txt)
            ops.append( document.OperationSettingSet(
                labelwidget.settings.get('label') , text ) )

    def actionFit(self):
        """Fit the data."""

        s = self.settings

        # check and get compiled for of function
        compiled = self.document.evaluate.compileCheckedExpression(s.function)
        if compiled is None:
            return

        #SUGGESTION - extract the parameter names from PARAMETER[low_limit,high_limit] syntax
        paramvalues,limits=valsExtractNamesLimits(s.values)
        paramnames = sorted(paramvalues)
        params = N.array( [paramvalues[p] for p in paramnames] )

        # FIXME: loads of error handling!!
        d = self.document

        # choose dataset depending on fit variable
        if s.variable == 'x':
            xdata = s.get('xData').getData(d)
            ydata = s.get('yData').getData(d)
        else:
            xdata = s.get('yData').getData(d)
            ydata = s.get('xData').getData(d)
        xvals = xdata.data
        yvals = ydata.data
        yserr = ydata.serr

        # if there are no errors on data
        #if yserr is None:
        #    if ydata.perr is not None and ydata.nerr is not None:
        #        print("Warning: Symmeterising positive and negative errors")
        #        yserr = N.sqrt( 0.5*(ydata.perr**2 + ydata.nerr**2) )
        #    else:
        #        print("Warning: No errors on y values. Assuming 5% errors.")
        #        yserr = N.abs(yvals*0.05)
        #        yserr[yserr < 1e-8] = 1e-8
        ##SUGGESTION - currently adding a value to an error column populates it with zeroes, resulting in a failure to fit 
        #elif N.sum(yserr==0.) > 0:
        #    averr = N.sum(N.abs(yserr[yserr!=0.]))/N.sum(N.abs(yvals[yserr!=0.]))
        #    print("Warning: Missing errors for some y values, replacing with average relative %g % errors." % averr*100)
        #    yserr[yserr==0.] = N.abs(yvals[yserr==0.]*averr)
        ##
        
        #SUGGESTION - avoid truncation & floating point overflow, relative errors are what are important for the fitting
        #minyserr=N.abs(N.min(yserr))
        #if minyserr<=1e-6:
        #   print("Warning: Excessively small errors of %g, scaling by %g." % (minyserr,1e-6/minyserr))
        #   yserr*=1e-6/minyserr

        #SUGGESTION - option to not weight by Y value
        if not s.weightYsquared:
            print("No 1/Y2 errors used.")
            yserr = yvals * 0.
        else:
            print("1/Y2 weights of fitted curve used")
            yserr = yvals * 0. + 1.

        # allow exclusion of data from fitting where 'do not process' flag is set 
        usepoint = ydata.flagDontProcessUnset() & xdata.flagDontProcessUnset()

        # if the fitRange parameter is on, we chop out data outside the
        # range of the axis
        if s.fitRange:
            # get ranges for axes
            if s.variable == 'x':
                drange = self.parent.getAxes((s.xAxis,))[0].getPlottedRange()
                mask = N.logical_and(xvals >= drange[0], xvals <= drange[1])
            else:
                drange = self.parent.getAxes((s.yAxis,))[0].getPlottedRange()
                mask = N.logical_and(yvals >= drange[0], yvals <= drange[1])
            xvals, yvals, yserr, usepoint = xvals[mask], yvals[mask], yserr[mask], usepoint[mask]
            print("Fitting %s from %g to %g" % (
                s.variable, drange[0], drange[1]))

        evalenv = self.initEnviron()
        def evalfunc(params, xvals):
            # update environment with variable and parameters
            evalenv[self.settings.variable] = xvals
            evalenv.update( zip(paramnames, params) )

            try:
                return eval(compiled, evalenv) + xvals*0.
            except Exception as e:
                self.document.log(str(e))
                return N.nan

        # minimum set for fitting
        if s.min != 'Auto':
            if s.variable == 'x':
                mask = xvals >= s.min
            else:
                mask = yvals >= s.min
            xvals, yvals, yserr, usepoint = xvals[mask], yvals[mask], yserr[mask], usepoint[mask]

        # maximum set for fitting
        if s.max != 'Auto':
            if s.variable == 'x':
                mask = xvals <= s.max
            else:
                mask = yvals <= s.max
            xvals, yvals, yserr, usepoint = xvals[mask], yvals[mask], yserr[mask], usepoint[mask]

        if s.min != 'Auto' or s.max != 'Auto':
            print("Fitting %s between %s and %s" % (s.variable, s.min, s.max))

        # various error checks
        if len(xvals) != len(yvals) or len(xvals) != len(yserr) or len(xvals) != len(usepoint):
            sys.stderr.write(_('Fit data not equal in length. Not fitting.\n'))
            return
        if len(params) > len(xvals):
            sys.stderr.write(_('No degrees of freedom for fit. Not fitting\n'))
            return

        # actually do the fit, either via Minuit or our own LM fitter
        chi2 = 1
        dof = 1

        # only consider finite values
        finite = N.isfinite(xvals) & N.isfinite(yvals) & N.isfinite(yserr) & usepoint
        xvals = xvals[finite]
        yvals = yvals[finite]
        yserr = yserr[finite]

        # check length after excluding non-finite values
        if len(xvals) == 0:
            sys.stderr.write(_('No data values. Not fitting.\n'))
            return

        if minuit is not None:
            vals, chi2, dof, errs = minuitFit(
                evalfunc, params, paramnames, s.values,
                xvals, yvals, yserr)
        else:
            print(_('Minuit not available, falling back to simple L-M fitting:'))
            retn, chi2, dof = utils.fitLM(
                evalfunc, params, xvals, yvals, yserr)
            vals = {}
            errs = {}
            #SUGGESTION - keep the limits format in case iminuit is used in the future            
            for i, v in zip(paramnames, retn):
                name="%s[%g,%g]" % (i,*limits)
                vals[name] = float(v)
                errs[name] = -1.

        # list of operations so we can undo the changes
        operations = []

        # populate the return parameters
        operations.append( document.OperationSettingSet(s.get('values'), vals) )
        #SUGGESTION - record the errors
        operations.append( document.OperationSettingSet(s.get('errors'), errs) )

        # populate the read-only fit quality params
        operations.append( document.OperationSettingSet(s.get('chi2'), float(chi2)) )
        operations.append( document.OperationSettingSet(s.get('dof'), int(dof)) )
        if dof <= 0:
            print(_('No degrees of freedom in fit.\n'))
            redchi2 = -1.
        else:
            redchi2 = float(chi2/dof)
        operations.append( document.OperationSettingSet(s.get('redchi2'), redchi2) )

        # expression for fit
        expr = self.generateOutputExpr(vals)
        operations.append( document.OperationSettingSet(s.get('outExpr'), expr) )

        self.updateOutputLabel(operations, vals, chi2, dof)

        # actually change all the settings
        d.applyOperation(
            document.OperationMultiple(operations, descr=_('fit')) )

    def generateOutputExpr(self, vals):
        """Try to generate text form of output expression.

        vals is a dict of variable: value pairs
        returns the expression
        """

        #SUGGESTION - separate parameters and limits
        paramvals,limits = valsExtractNamesLimits(vals)
        
        s = self.settings

        # also substitute in data name for variable
        if s.variable == 'x':
            paramvals['x'] = s.xData
        else:
            paramvals['y'] = s.yData

        # split expression up into parts of text and nums, separated
        # by non-text/nums
        parts = re.split('([^A-Za-z0-9.])', s.function)

        # replace part by things in paramvals, if they exist
        for i, p in enumerate(parts):
            if p in paramvals:
                parts[i] = str(paramvals[p])

        return ''.join(parts)

# allow the factory to instantiate an x,y plotter
document.thefactory.register(Fit)
