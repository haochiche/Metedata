#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:16:54 2018

@author: che

main fuctions of model data read and plot
"""

# write to ukca emission file
import iris
import os
import pandas as pd
import numpy as np
import matplotlib
import scipy as sp
import cf_units
import xarray as xr

def write_emission(cube,outpath,name ):
    ats = cube.attributes
    saver = iris.fileformats.netcdf.Saver(filename=os.path.join(outpath,name), netcdf_format='NETCDF3_CLASSIC')
    if 'highest_level' in ats.keys():
        saver.write(cube, local_keys=['vertical_scaling', 'missing_value','um_stash_source','tracer_name','highest_level','lowest_level'])
    else:
        saver.write(cube, local_keys=['vertical_scaling', 'missing_value','um_stash_source','tracer_name'])
    print('finish')

    
def get_altitude(cubelist):
    tp = type(cubelist)
    #use tp to check if input is a cubelist or a cube
    orog=iris.load_cube(r'/a/home/cc/students/space/haochiche/data/orography.nc')
    auxcoord=iris.coords.AuxCoord(orog.data,standard_name=str(orog.standard_name),var_name="surface_altitude",
                                  units=orog.units)
    if tp ==iris.cube.CubeList:
        for cube in cubelist:
            if not cube.coords('surface_altitude'):
    #get coord dim
                a, = cube.coord_dims('latitude')
                b, = cube.coord_dims('longitude')
    #add to cube
                try:
                    cube.add_aux_coord(auxcoord,(a,b,))
    #calc
                    factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
                    cube.add_aux_factory(factory)
                except Exception as e:
                    print(cube.attributes['STASH'],e)
            else:
                if not cube.coords('altitude'):
                    try:
                        factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
                        cube.add_aux_factory(factory)
                    except Exception as e:
                        print(cube.attributes['STASH'],e)

            if cube.coords('atmosphere_hybrid_height_coordinate'):
                cube.coord('atmosphere_hybrid_height_coordinate').standard_name = None
            try:

                cube.coord('surface_altitude').attributes = {}
                cube.coord('sigma').var_name = 'sigma'
                cube.coord('model_level_number').var_name='model_level_number'
                cube.coord('level_height').var_name = 'level_height'
            except:
                pass
        cubelist = cubelist.merge()
    
    elif tp ==iris.cube.Cube:
        if not cubelist.coords('surface_altitude'):
    #get coord dim
            a, = cubelist.coord_dims('latitude')
            b, = cubelist.coord_dims('longitude')
    #add to cube
            try:
                cubelist.add_aux_coord(auxcoord,(a,b,))
    #calc
                factory=iris.aux_factory.HybridHeightFactory(delta=cubelist.coord("level_height"),sigma=cubelist.coord("sigma"),
                                             orography=cubelist.coord("surface_altitude"))
                cubelist.add_aux_factory(factory)
            except Exception as e:
                print(cubelist.attributes['STASH'],e)
        else:
            if not cubelist.coords('altitude'):
                try:
                    factory=iris.aux_factory.HybridHeightFactory(delta=cubelist.coord("level_height"),sigma=cubelist.coord("sigma"),
                                             orography=cubelist.coord("surface_altitude"))
                    cubelist.add_aux_factory(factory)
                except Exception as e:
                    print(cubelist.attributes['STASH'],e)

        if cubelist.coords('atmosphere_hybrid_height_coordinate'):
            cubelist.coord('atmosphere_hybrid_height_coordinate').standard_name = None
        try:

            cubelist.coord('surface_altitude').attributes = {}
            cubelist.coord('sigma').var_name = 'sigma'
            cubelist.coord('model_level_number').var_name='model_level_number'
            cubelist.coord('level_height').var_name = 'level_height'
        except:
            pass

def get_altitude_clarify(cube):
    orog=iris.load_cube(r'/a/home/cc/students/space/haochiche/data/orography_clarify.nc')
    auxcoord=iris.coords.AuxCoord(orog.data,standard_name=str(orog.standard_name),var_name="surface_altitude",
                                  units=orog.units)
    if not cube.coords('surface_altitude'):
    #get coord dim
        a, = cube.coord_dims('latitude')
        b, = cube.coord_dims('longitude')
    #add to cube
        try:
            cube.add_aux_coord(auxcoord,(a,b,))
    #calc
            factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
            cube.add_aux_factory(factory)
        except Exception as e:
            print(cube.attributes['STASH'],e)
    else:
        if cube.coords('atmosphere_hybrid_height_coordinate'):
            cube.coord('atmosphere_hybrid_height_coordinate').standard_name = None
        
        try:
            cube.coord('surface_altitude').attributes = {}
            cube.coord('sigma').var_name = 'sigma'
            cube.coord('model_level_number').var_name='model_level_number'
            cube.coord('level_height').var_name = 'level_height'
        except:
            pass
        if not cube.coords('altitude'):
            try:
                factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
                cube.add_aux_factory(factory)
            except Exception as e:
                print(cube.attributes['STASH'],e)


def get_altitude_2016(cube):
    orog=iris.load_cube(r'/a/home/cc/students/space/haochiche/data/orography_2016.nc')
    auxcoord=iris.coords.AuxCoord(orog.data,standard_name=str(orog.standard_name),var_name="surface_altitude",
                                  units=orog.units)
    if not cube.coords('surface_altitude'):
    #get coord dim
        a, = cube.coord_dims('latitude')
        b, = cube.coord_dims('longitude')
    #add to cube
        try:
            cube.add_aux_coord(auxcoord,(a,b,))
    #calc
            factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
            cube.add_aux_factory(factory)
        except Exception as e:
            print(cube.attributes['STASH'],e)
    else:
        if cube.coords('atmosphere_hybrid_height_coordinate'):
            cube.coord('atmosphere_hybrid_height_coordinate').standard_name = None
        
        try:
            cube.coord('surface_altitude').attributes = {}
            cube.coord('sigma').var_name = 'sigma'
            cube.coord('model_level_number').var_name='model_level_number'
            cube.coord('level_height').var_name = 'level_height'
        except:
            pass
        if not cube.coords('altitude'):
            try:
                factory=iris.aux_factory.HybridHeightFactory(delta=cube.coord("level_height"),sigma=cube.coord("sigma"),
                                                 orography=cube.coord("surface_altitude"))
                cube.add_aux_factory(factory)
            except Exception as e:
                print(cube.attributes['STASH'],e)



def get_all_mean(cubelist):
    tp = type(cubelist)
    #use tp to check if input is a cubelist or a cube
    if tp ==iris.cube.CubeList:
        #cubelist
        #check if there are 12 time steps
        for items in cubelist:
            if items.coord('time').points.size in [24,2]:
                continue
            else:
                print('time length error, {} have time length of {}'.format(items.attributes['STASH'],
                                                             items.coord('time').points.size))
    #input a cubelist and output the sum cube of the lists
        new = sum(cubelist).collapsed('time',iris.analysis.MEAN)
    elif tp==iris.cube.Cube:
        #input is a cube
        if cubelist.coord('time').points.size in [24,2]:
            pass
        else:
            print('time length error, have time length of {}'.format(items.coord('time').points.size))
        new = cubelist.collapsed('time',iris.analysis.MEAN)
#     new.var_name = sp
    return(new)


def get_area_mean(cube):
    #get mean
    try:
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    except:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN,weights=grid_areas).data
    return(mean)

def get_area_mean_cube(cube):
    #get mean
    try:
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    except:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN,weights=grid_areas)
    return(mean)

def get_integral(cube):
    if 'altitude' not in [coord.name() for coord in cube[-2].coords()]:
        cube = get_altitude(cube)
    bounds = cube.coord('altitude').bounds[:,:,:,1] - cube.coord('altitude').bounds[:,:,:,0]
    # mutliply by the height of each cell
    cube.data = cube.data * bounds
    cube_int=cube.collapsed('model_level_number',iris.analysis.SUM)
    return(cube_int)

def read_aeronet(filename):
    """Read a given AERONET AOT data file, and return it as a dataframe.
    
    This returns a DataFrame containing the AERONET data, with the index
    set to the timestamp of the AERONET observations. Rows or columns
    consisting entirely of missing data are removed. All other columns
    are left as-is.
    """
    dateparse = lambda x: pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    aeronet = pd.read_csv(filename, skiprows=6, na_values=['N/A'],
                          parse_dates={'times':[0,1]},
                          date_parser=dateparse)

    aeronet = aeronet.set_index('times')
    
    # Drop any rows that are all NaN and any cols that are all NaN
    # & then sort by the index
    aeronet = aeronet.replace(-999.0,np.nan)
    an = (aeronet.dropna(axis=1, how='all')
                .dropna(axis=0, how='all')
                .rename(columns={'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'})
                .sort_index())

    return an

def get_exner_pressure(p):
    #input air pressure and calc exner pressure
    p0 = iris.coords.AuxCoord(1000, long_name='P0', units='hPa')
    p0.convert_units(p.units)
    exner_pressure = (p / p0) ** (287.05 / 1005.0)
    exner_pressure.rename('exner_pressure')
    return(exner_pressure)

def get_temperature(pressure,theta):
    #calc temperature from the theta
    #input pressure and theta
    exner_p = get_exner_pressure(pressure)
    t  = exner_p * theta
    return(t)

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

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

def get_names(cubes):
    #use tp to determind if the input is a cube or a cubelist
    tp = type(cubes)
    #use tp to check if input is a cubelist or a cube
    #input a cube list 
    from iris.fileformats.um_cf_map import STASH_TO_CF
    from model.setup_ukca import add_ukca_to_iris
    add_ukca_to_iris()
    if tp ==iris.cube.CubeList:
        for c in cubes:
            a = str(c.attributes['STASH'])
            if a in STASH_TO_CF:
                if STASH_TO_CF[a].standard_name:
                    c.rename(STASH_TO_CF[a].standard_name)
                else:
                    c.rename(STASH_TO_CF[a].long_name)
                c.units = cf_units.Unit(STASH_TO_CF[a].units)
    elif tp==iris.cube.Cube:
        a = str(cubes.attributes['STASH'])
        if a in STASH_TO_CF:
            if STASH_TO_CF[a].standard_name:
                cubes.rename(STASH_TO_CF[a].standard_name)
            else:
                cubes.rename(STASH_TO_CF[a].long_name)
            cubes.units = cf_units.Unit(STASH_TO_CF[a].units)


def panoply():
    my_color = ['#0000CC','#0103CE','#0307D1','#040BD3','#060FD6','#0713D9','#0917DB',
           '#0A1BDE','#0C1EE0','#0D22E3','#0F26E6','#102AE8','#122EEB','#1332ED','#1536F0',
           '#173AF3','#183EF5','#1A41F7','#1C46F9','#1E4BFC','#204FFC','#2152FD','#2356FE',
           '#265CFF','#2760FF','#2964FF','#2B68FF','#2C6BFF','#2D6EFF','#2E71FF','#3175FF',
           '#3279FF','#357EFF','#3681FF','#3985FF','#3B89FF','#3D8DFF','#3F90FF','#4093FF',
           '#4396FF','#469AFF','#499EFF','#4DA2FF','#4FA5FF','#51A7FF','#53A9FF','#55ABFF',
           '#56ACFF','#58AEFF','#5AB0FF','#5DB2FF','#5FB4FF','#62B7FF','#65BAFF','#67BCFF',
           '#6ABEFF','#6DC1FF','#6EC2FF','#71C5FF','#73C7FF','#75C9FF','#77CAFF','#78CBFF',
           '#79CCFF','#7ACDFF','#7BCFFF','#7DD1FF','#7FD2FF','#80D3FF','#81D4FF','#82D5FF',
           '#83D6FF','#84D7FF','#85D8FF','#86D9FF','#87DAFF','#88DBFF','#89DCFF','#8ADDFF',
           '#8BDEFF','#8CDFFF','#8DE0FF','#8FE2FF','#8FE1FF','#90E3FF','#91E4FF','#92E5FF',
           '#93E6FF','#95E8FF','#95E7FF','#96E9FF','#97EAFF','#99EBFF','#9AECFF','#9CEDFF',
           '#9DEEFF','#9EEFFF','#A1F0FF','#A3F1FF','#A5F2FF','#A6F2FF','#AAF3FF','#A9F3FF',
           '#ABF4FF','#ADF4FF','#AFF5FF','#B0F5FF','#B3F6FF','#B1F6FF','#B5F7FF','#B6F7FF',
           '#B7F8FF','#B9F9FF','#BAF9FF','#BBFAFF','#BDFAFF','#BFFBFF','#C0FBFF','#C3FCFF',
           '#C4FCFF','#C7FDFF','#CAFDF5','#CEFDEB','#D2FDE1','#D5FDD7','#D9FDCC','#DDFDC2',
           '#E1FDB8','#E4FEAE','#E8FEA4','#ECFE99','#F0FE8F','#F3FE85','#F7FE7B','#FBFE71',
           '#FFFF66','#FFFF63','#FFFE5D','#FFFD57','#FFFC51','#FFFB4B','#FFFA45','#FFF83E',
           '#FFF738','#FFF632','#FFF52C','#FFF426','#FFF21F','#FFF119','#FFF013','#FFEF0D',
            '#FFEE07','#FFEC00','#FFEA00','#FFE700','#FFE500','#FFE200','#FFE000','#FFDD00',
            '#FFDB00','#FFD800','#FFD600','#FFD300','#FFD100','#FFCE00','#FFCC00','#FFC900',
            '#FFC700','#FFC400','#FFC100','#FFBE00','#FFBB00','#FFB700','#FFB400','#FFB100',
            '#FFAE00','#FFAA00','#FFA700','#FFA400','#FFA100','#FF9D00','#FF9A00','#FF9700',
            '#FF9400','#FF9000','#FF8C00','#FF8800','#FF8300','#FF7F00','#FF7A00','#FF7600',
            '#FF7100','#FF6D00','#FF6900','#FF6400','#FF6000','#FF5B00','#FF5700','#FF5200',
            '#FF4E00','#FF4900','#FF4500','#FF4000','#FF3C00','#FF3700','#FF3300','#FF2E00',
            '#FF2A00','#FF2500','#FF2000','#FF1C00','#FF1700','#FF1300','#FF0E00','#FF0A00',
            '#FF0500','#FF0000','#FD0000','#FA0000','#F80000','#F50000','#F20000','#F00000',
            '#ED0000','#EA0000','#E80000','#E50000','#E30000','#E00000','#DD0000','#DB0000',
            '#D80000','#D50000','#D20000','#CF0000','#CC0000','#C90000','#C60000','#C30000',
            '#C00000','#BD0000','#BA0000','#B70000','#B40000','#B10000','#AE0000','#AB0000',
            '#A80000','#A40000','#9F0000','#9A0000','#940000','#8F0000','#8A0000','#850000',
            '#800000']
    my_cmap = matplotlib.colors.ListedColormap(my_color, name='panoply')
    return(my_cmap)

def calc_tropopause(temperature):
    import tropo
    #Input the temperature (time, lon,lat,lev)
    #The vertical level should be pressure
    plimu=45000
    pliml=7500
    plimlex=7500
    dofill=True
    #transpose the dimension of temperature to match the fortran code
    temperature = temperature.transpose('time', 'lon', 'lat', 'plev')
    ntime, nlon,nlat,nlev = temperature.shape
    pres = temperature['plev']/100 #pressure in hPa

    #xarray for result
    dims = ('time', 'lon', 'lat')
    coords = {dim: temperature.coords[dim] for dim in dims}
    # Create an empty DataArray with the specified dimensions and coordinates
    tp = xr.DataArray(np.nan, dims=dims, coords=coords,name='tp')

    for time in range(ntime):
        t = temperature.isel(time=time)
        temptp,tperr= tropo.tropo(t,pres,plimu,pliml,plimlex,dofill)
        tp[dict(time=time)] = temptp    
    return tp

def xr_area_mean(ds):
    "calc the area weighted mean of an xarray array"
    ds_weighted = ds.weighted(np.cos(np.deg2rad(ds.lat)))
    mean = ds_weighted.mean(dim=['lat','lon'])
    return (mean)

def running_mean(ds,years=21):
    #calc 21 years running mean for each month
    monthly_means = []
    for month in range(1, 13):
        # Subset the data for the specific month
        subset = ds.sel(time=ds['time.month'] == month)
        # Compute the 21-year rolling mean centered
        window_size = years  # 21 years
        rolling_mean = subset.rolling(time=window_size, center=True).mean()
        # Replace NaNs for the first 10 years with the value from the 11th year
        center_year = years//2  #10
        rolling_mean[:center_year] = rolling_mean[center_year]
        # #do nothing for the last 10 years
        # # Replace NaNs for the last 10 years with the value from the 11th last year
        # rolling_mean[-10:] = rolling_mean[-11]
        monthly_means.append(rolling_mean)
        
    # Concatenate monthly rolling means
    monthly_means = xr.concat(monthly_means, dim="time")
    monthly_means = monthly_means.sortby('time')
    return (monthly_means)