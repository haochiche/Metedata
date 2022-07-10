#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:25:39 2018

@author: che

use this script to caculate AOD from UM pp files
"""
import iris
import iris.time

def write_aod_single(fn,pls,fout):
# =============================================================================
#     input variable are:
#     fn: filename, can be a list
#     pls: specific the pseudo level (0.38, 0.44, 0.55, 0.67, 0.87, and 1.02 um)
#     fout: out put filename
# =============================================================================
    fname= fn
#    # constraint on time to get 2nd radiation timestep
#    tconstr=iris.Constraint(time=lambda cell: cell.point.hour == 2)

# load all AOD components at 0.55 micron
# must use this way of loading to account for constraint on time
    with iris.FUTURE.context(cell_datetime_objects=True):
        aod=iris.load(fname,[
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i285'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i300'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i301'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i302'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i303'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i304'),
                iris.Constraint(pseudo_level=pls) & iris.AttributeConstraint(STASH='m01s02i305')])

    # make cube to store total AOD
    aodsum=aod[0].copy()

    # add-up components
    aodsum.data=aod[0].data+aod[1].data+aod[2].data+aod[3].data+aod[4].data+aod[5].data+aod[6].data

    # rename
    aodsum.rename('atmosphere_optical_thickness_due_to_aerosol')

    # remove unlimited dimension when writing to netCDF
    iris.FUTURE.netcdf_no_unlimited=True

    # output to netCDF
    outdir = '{}/AOD_l{}.nc'.format(fout,pls) 
    iris.save(aodsum,outdir,netcdf_format='NETCDF3_CLASSIC')


def write_aod_all(fn,fout):
# =============================================================================
#     input variable are:
#     fn: filename, can be a list
#     pls: specific the pseudo level (0.38, 0.44, 0.55, 0.67, 0.87, and 1.02 um)
#     fout: out put filename
# =============================================================================
    fname= fn
#    # constraint on time to get 2nd radiation timestep
#    tconstr=iris.Constraint(time=lambda cell: cell.point.hour == 2)

# load all AOD components at 0.55 micron
# must use this way of loading to account for constraint on time
    with iris.FUTURE.context(cell_datetime_objects=True):
        aod=iris.load(fname,[
            iris.AttributeConstraint(STASH='m01s02i285'),
            iris.AttributeConstraint(STASH='m01s02i300'),
            iris.AttributeConstraint(STASH='m01s02i301'),
            iris.AttributeConstraint(STASH='m01s02i302'),
            iris.AttributeConstraint(STASH='m01s02i303'),
            iris.AttributeConstraint(STASH='m01s02i304'),
            iris.AttributeConstraint(STASH='m01s02i305')])

    # make cube to store total AOD
    aodsum=aod[0].copy()

    # add-up components
    aodsum.data=aod[0].data+aod[1].data+aod[2].data+aod[3].data+aod[4].data+aod[5].data+aod[6].data

    # rename
    aodsum.rename('atmosphere_optical_thickness_due_to_aerosol')

    # remove unlimited dimension when writing to netCDF
    iris.FUTURE.netcdf_no_unlimited=True

    # output to netCDF
    iris.save(aodsum,fout,netcdf_format='NETCDF3_CLASSIC')
