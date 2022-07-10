#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:15:05 2018

@author: che

use this script to plot aerosol and cloud figures from model
"""
import matplotlib.pylab as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import setup_ukca
import os

datdir = r'/Users/che/work/clarify/data/UKCA/u-au742'
#s_humidity =  iris.load_cube(os.path.join(datdir,'au742a.pg2015sep.pp'),iris.AttributeConstraint(STASH='m01s00i010'))
#airmass = iris.load_cube(os.path.join(datdir,'au742a.pg2015sep.pp'),iris.AttributeConstraint(STASH='m01s50i063'))
#dry_air = airmass(1-s_humidity)
cubes = iris.load(os.path.join(datdir,'au742a.pc2015sep.pp'))
#%%
cross_section = next(cubes[4].slices(['longitude','model_level_number']))
qplt.contourf(cross_section, coords=['longitude', 'altitude'],cmap='RdBu_r')

iplt.show()