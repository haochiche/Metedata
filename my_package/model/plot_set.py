#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:20:41 2018

@author: che

import modul for plot settings

"""
import os
import iris
import matplotlib.pylab as plt
import iris.quickplot as qplt
import iris.plot as iplt
import numpy as np
import cf_units
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
import matplotlib.ticker
np.random.seed(0)


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_all_time(ocube,outpath,title='name'):
    if len(ocube.name())>0:
        title = ocube.name()
    for time in range(len(ocube.coord('time').points)):
        fig = plt.figure(figsize=(9, 5))
        brewer_cmap = plt.get_cmap('brewer_Blues_09')
        contour_result = iplt.contourf(ocube[time,:,:], 25,linewidth=0, cmap=brewer_cmap,extend='both')
#        contour_result = iplt.pcolormesh(aerosol_sum[psl,time,:,:])
        time_name = ocube.coord('time').units.num2date(ocube.coord('time').points[time])
        plt.title('{0} in {1} [{3}]'.format(title,time_name.strftime("%b %Y")),ocube.units)
        ax = plt.gca()
        ax.coastlines()
        left, bottom, width, height = ax.get_position().bounds
        colorbar_axes = fig.add_axes([left +0.01, bottom -0.06,width-0.02, 0.03])
        cbar = plt.colorbar(contour_result, colorbar_axes,orientation='horizontal')
#        cbar.set_label('')
        cbar.ax.tick_params(length=0)
        plt.tight_layout()
#        plt.show()
        plt.savefig(os.path.join(outpath,time_name.strftime("%Y_%m")+'.png'))
        plt.clf()
        plt.close(fig)
        


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)