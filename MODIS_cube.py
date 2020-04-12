from collections import OrderedDict
from iris.coords import DimCoord
import iris
import numpy as np
import xarray as xr

def parse_hdfeos_metadata(string):
    out = OrderedDict()
    lines = [i.replace('\t','') for i in string.split('\n')]
    i = -1
    while i<(len(lines))-1:
        i+=1
        line = lines[i]
        if "=" in line:
            key,value = line.split('=')
            if key in ['GROUP','OBJECT']:
                endIdx = lines.index('END_{}={}'.format(key,value))
                out[value] = parse_hdfeos_metadata("\n".join(lines[i+1:endIdx]))
                i = endIdx
            else:
                if ('END_GROUP' not in key) and ('END_OBJECT' not in key):
                    try:
                        out[key] = eval(value)
                    except NameError:
                        out[key] = str(value)
    return(out)


def make_cube(ds,keywd,scale=1e6):
    #read data and convert to a iris cube
    tps = ds[keywd]
    tps.attrs['units'] = None
    cube = tps.to_iris()
    metadata = parse_hdfeos_metadata(ds.attrs['StructMetadata.0'])
    gridInfo = metadata['GridStructure']['GRID_1']
    x1,y1 = gridInfo['UpperLeftPointMtrs']
    x2,y2 = gridInfo['LowerRightMtrs']
    yRes = (y1-y2)/gridInfo['YDim']
    xRes = (x1-x2)/gridInfo['XDim']
    x = np.arange(x2,x1,xRes)[::-1]/scale
    y = np.arange(y2,y1,yRes)[::-1]/scale
    latitude = DimCoord(y,standard_name='latitude',units='degrees')
    longitude = DimCoord(x,standard_name='longitude',units='degrees')
    cube.add_dim_coord(longitude,1)
    cube.add_dim_coord(latitude,0)
    return(cube)

#Example to get land types data
ds = xr.open_dataset(r'/Users/Dropbox/ASR/Data/MCD12C1.A2017001.006.2019192025407.hdf')
cube = make_cube(ds,'Majority_Land_Cover_Type_1')

