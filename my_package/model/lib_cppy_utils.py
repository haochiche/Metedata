"""
Assorted utilities

(c) Duncan watson-parris 2017
"""
import matplotlib.colors as colors
import numpy as np

lat_two_degree_bins = np.linspace(-90, 90, 91)
lon_two_degree_bins = np.linspace(-180, 180, 181)

lat_one_degree_bins = np.linspace(-90, 90, 181)
lon_one_degree_bins = np.linspace(-180, 180, 361)

lat_half_degree_bins = np.linspace(-90, 90, 361)
lon_half_degree_bins = np.linspace(-180, 180, 721)


def get_git_rev():
    import subprocess
    import os
    if os.name == 'nt':
        git_path = "C:\\Program Files\\Git\\bin\\git.exe"
    else:
        git_path = "/usr/bin/git"

    return subprocess.check_output([git_path, "describe", "--always"])


def data_list_to_df(data_list, units=None):
    """
    Convert a list (stack) of UnigriddedData objects to one big DataFrame

    This assumes the coordinates have the same *sets* of coordinates, but that the coordinates themselves are different.
    """
    import pandas as pd
    import logging
    from cf_units import Unit

    df_list = []
    for c in data_list:
        df = c.as_data_frame()
        df['Campaign'] = c.metadata.campaign

        if units:
            curr_units = c.metadata.units
            if not isinstance(curr_units, Unit):
                curr_units = Unit(curr_units)
            try:
                new_values = curr_units.convert(df[c.name()].values, units)
                df[c.name()] = new_values
            except ValueError as e:
                logging.warning("Unable to convert {} to {}, for {}".format(c.metadata.units,
                                                                           units,
                                                                           c.name()))
                continue

        df_list.append(df)
    return pd.concat(df_list)


def UngriddedData_from_data_frame(df, cols, names=None, air_pressure_units=None):
    """
    Create an UngriddedData object from the cols of a datafame (df)

    :param df: The input dataframe
    :param cols: The columns to extract (note that coordinates are dealt with automatically)
    :param names: The names to give the date objects (if different to the column names)
    :param air_pressure_units: Optional air pressure units which the data was ORIGINALLY in. The output is always hPa
    :return UngriddedDataList: List of UngriddedData objects, one for each column specified
    """
    from cis.data_io.ungridded_data import UngriddedData, UngriddedDataList, Metadata
    from cis.data_io.Coord import Coord, CoordList
    from cis.data_io.write_netcdf import types as valid_types
    from cis.time_util import cis_standard_time_unit
    from cf_units import Unit
    from numpy import ma

    fill_value = np.nan
    # define our function to perform the case-insensitive search
    def find_col_name(name):
        col_list = list(df)
        try:
            # this uses a generator to find the index if it matches, will raise an exception if not found
            return col_list[next(i for i, v in enumerate(col_list) if v.lower() == name)]
        except:
            return ''

    lat_col_name = find_col_name('latitude')
    lon_col_name = find_col_name('longitude')
    alt_col_name = find_col_name('altitude')
    pres_col_name = find_col_name('air_pressure')

    coords = CoordList()
    out_data = UngriddedDataList()
    numpy_time_unit = Unit('ns since 1970-01-01T00:00Z')
    time_vals = numpy_time_unit.convert(df.index.values.astype('float64'), cis_standard_time_unit)
    coords.append(Coord(time_vals, Metadata(standard_name='time', units=str(cis_standard_time_unit))))
    coords.append(Coord(df[lat_col_name].values, Metadata(standard_name='latitude', units='degrees_north')))
    coords.append(Coord(df[lon_col_name].values, Metadata(standard_name='longitude', units='degrees_east')))
    df = df.drop([lat_col_name, lon_col_name], axis=1)
    if alt_col_name in df:
        coords.append(Coord(df[alt_col_name].values, Metadata(standard_name='altitude', units='meters')))
        df = df.drop([alt_col_name], axis=1)
    if pres_col_name in df:
        air_pressure_units = air_pressure_units if air_pressure_units is not None else 'hPa'
        pres_data = Unit(air_pressure_units).convert(df[pres_col_name].values, 'hPa')
        coords.append(Coord(pres_data, Metadata(standard_name='air_pressure', units='hPa')))
        df = df.drop([pres_col_name], axis=1)

    # Check the names and cols match up if present
    # if (cols and names) and (len(cols) != len(names)):
    #     raise ValueError()
    # elif not names:
    #     names = cols

    for col, _name in zip(cols, names):
        if str(df[col].values.dtype) in valid_types.keys():
            if df[col].isnull().any():
                # Note we specify the mask explitly for each column because we could have coiord values which are valid
                # for some data variables and not others
                data = ma.array(df[col].values, mask=df[col].isnull(), fill_value=fill_value)
                data[data.mask] = fill_value
            else:
                data = df[col].values
            meta = Metadata(long_name=col, name=_name)
            if str(df[col].values.dtype) != 'object':
                meta.missing_value = fill_value
            out_data.append(UngriddedData(data, meta, coords.copy()))
    return out_data


def split_dataset_based_on_id(data_var, id_var):
    """
    Split one ungridded dataset based on ids from another ungridded dataset
    :param data_var:
    :param id_var:
    :return:
    """
    from cis.data_io.ungridded_data import UngriddedDataList
    res = UngriddedDataList([])



def stack_data_list(data_list, var_name=None, units=None):
    """
    Stacks a list of Ungridded data objects with the same data variable, but different coordinates into a single
    UngriddedData object, with accompanying lat, lon and time data.


    It assumes the coordinates have the same *sets* of coordinates, but that the coordinates themselves are different.

    :param data_list: list of UngriddedData objects to be merged (stacked vertically)
    :param string var_name: Name of final data variable
    :param string units: Units of final data variable
    :return: A merged UngriddedData object
    """
    from cis.data_io.ungridded_data import UngriddedData
    from cis.data_io.Coord import Coord

    metadata = data_list[0].metadata
    if var_name is not None:
        metadata._name = var_name

    if units is not None:
        metadata.units = units

    coords = []
    all_data = np.hstack((d.data for d in data_list))
    for c in data_list[0].coords():
        coord_data = np.hstack((d.coord(c).data for d in data_list))
        coords.append(Coord(coord_data, c.metadata))

    return UngriddedData(all_data, metadata, coords)


def plot_spatial_heatmap(data, x=None, y=None, title=''):
    """
    Plots a heatmap over blue marble with invalid values masked out.
    :param data: A 2-d numpy array of shape (360,180)
    :param string title:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    import cartopy.crs as ccrs

    x = x if x is not None else lon_one_degree_bins
    y = y if y is not None else lat_one_degree_bins

    proj = ccrs.PlateCarree()

    ax = plt.axes(projection=proj)
    ax.coastlines()

    Y, X = np.meshgrid(y, x)

    Zm = ma.masked_invalid(data)

    mapable = ax.pcolormesh(X, Y, Zm, vmin=-1, vmax=1, cmap='RdBu_r')

    plt.colorbar(mapable, orientation='horizontal', shrink=0.6)

    plt.gcf().set_size_inches(18.5, 10.5)


def plot_spatial_scatter(data, label='', **kwargs):
    """
    Plots a heatmap over blue marble from a dataframe.
    :param data: A df with Lat, Lon and label coordinates
    :param string label: The name of the data variable in the dataframe to plot
    :return:
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    proj = ccrs.PlateCarree()

    ax = plt.axes(projection=proj)
    ax.coastlines()

    mapable = ax.scatter(data['Lon'], data['Lat'], c=data[label], **kwargs)

    plt.colorbar(mapable, orientation='horizontal', shrink=0.6)

    plt.title(label)
    # plt.xlabel("Lon")
    # plt.ylabel("Lat")
    plt.gcf().set_size_inches(18.5, 10.5)


def power_spectum(time_series):
    return np.abs(np.fft.fft(time_series))**2


def plot_monthly_histograms(time_series):
    from pywork.constants import months
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-100, 1000)

    legend = []

    for i in range(0,12):
        mon = time_series[time_series.index.month==i]
        if not mon.empty:
            legend.append(months[i])
            mon.hist(bins=100, range=(0,1000),histtype='step', ax=ax, normed=True)

    ax.legend(legend)
    plt.gcf().set_size_inches(18.5, 10.5)


def plot_monthly_means(time_series, **kwargs):
    from pywork.constants import months
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 13)

    legend = []

    for i in range(0,12):
        mon_mean = time_series[time_series.index.month==i].mean()
        mon_std = time_series[time_series.index.month==i].std()
        if mon_mean is not np.NaN:
            legend.append(months[i])
            ax.errorbar(i, mon_mean, yerr=mon_std, **kwargs)

    ax.legend(legend)
    plt.gcf().set_size_inches(18.5, 10.5)


def get_geographic_mask(ungridded_data, resolution='110m', category='physical', name='land', attribute_filter=None,
                        by_bounding_box=False):
    """
    Return a mask which masks any points over a geographic region. By default will select land regions. Based on
     the features available from NaturalEarth.com

    :param UngriddedData ungridded_data: The dataset containing the points to be masked - No actual masking is applied though.
    :param str resolution: The resolution of the shape file to use, default is 110m
    :param str category: The Natural Earth category to search
    :param str name: The name of the Natural Earth feature set to use.
    :param dict attribute_filter: A dictionary of attribute filters to apply to the NE records.
    E.g. 'country': 'Albania' would return only those records with the 'country' attribute 'Albalia'
    :param bool by_bounding_box: Use a simple bounding box to perform the masking? This can be much faster
    :return: A numpy array with 'False' representing a point over land, and a 'True' for a point not over land
    """
    from cartopy.io.shapereader import Reader, natural_earth
    from shapely.geometry import MultiPoint, box
    from shapely.ops import unary_union
    from cis.utils import fix_longitude_range

    attribute_filter = attribute_filter if attribute_filter is not None else {}

    # The shape files are -180 to 180, so make sure our lon coords are too
    fixed_lon_points = fix_longitude_range(ungridded_data.lon.points, -180)

    shpfile = Reader(natural_earth(resolution=resolution, category=category, name=name))

    filtered_records = shpfile.records()
    for key, val in attribute_filter.items():
        filtered_records = filter(lambda x: x.attributes[key] == val, filtered_records)

    region_poly = unary_union([r.geometry for r in filtered_records])

    # It would be nice to make use of the asMultiPoint interface but it's not working for z-points at the moment
    # cis_data = np.vstack([sim_points.lon.data, sim_points.lat.data, np.arange(len(sim_points.lat.data))])
    # points = MultiPoint(cis_data.T)
    # OR:
    # geos_data = asMultiPoint(cis_data.T)

    # Create shapely Points, we use the z coord as a cheat for holding an index to our original points
    points = MultiPoint([(lon, lat, i)
                         for i, (lat, lon) in enumerate(zip(ungridded_data.lat.points, fixed_lon_points))])

    # If we're only doing this by bounding box then turn the (potentially complex polygon) into a box
    if by_bounding_box:
        region_poly = box(*region_poly.bounds)

    # Perform the actual calculation
    selection = region_poly.intersection(points)

    # Pull out the indices
    mask_indices = np.asarray(selection).T[2].astype(np.int)

    # Build our mask
    mask = np.ones(ungridded_data.shape)
    mask[mask_indices] = 0

    return mask.astype(np.bool)


def get_geographic_mask_df(df, resolution='110m', category='physical', name='land', attribute_filter=None,
                        by_bounding_box=False):
    #TODO: This is basically a duplicate of the above method - it needs refactoring
    """
    Return a mask which masks any points over a geographic region. By default will select land regions. Based on
     the features available from NaturalEarth.com

    :param UngriddedData ungridded_data: The dataset containing the points to be masked - No actual masking is applied though.
    :param str resolution: The resolution of the shape file to use, default is 110m
    :param str category: The Natural Earth category to search
    :param str name: The name of the Natural Earth feature set to use.
    :param dict attribute_filter: A dictionary of attribute filters to apply to the NE records.
    E.g. 'country': 'Albania' would return only those records with the 'country' attribute 'Albalia'
    :param bool by_bounding_box: Use a simple bounding box to perform the masking? This can be much faster
    :return: A numpy array with 'False' representing a point over land, and a 'True' for a point not over land
    """
    from cartopy.io.shapereader import Reader, natural_earth
    from shapely.geometry import MultiPoint, box
    from shapely.ops import unary_union
    from cis.utils import fix_longitude_range

    attribute_filter = attribute_filter if attribute_filter is not None else {}

    # The shape files are -180 to 180, so make sure our lon coords are too
    fixed_lon_points = fix_longitude_range(df.longitude, -180)

    shpfile = Reader(natural_earth(resolution=resolution, category=category, name=name))

    filtered_records = shpfile.records()
    for key, val in attribute_filter.items():
        filtered_records = filter(lambda x: x.attributes[key] == val, filtered_records)

    region_poly = unary_union([r.geometry for r in filtered_records])

    # It would be nice to make use of the asMultiPoint interface but it's not working for z-points at the moment
    # cis_data = np.vstack([sim_points.lon.data, sim_points.lat.data, np.arange(len(sim_points.lat.data))])
    # points = MultiPoint(cis_data.T)
    # OR:
    # geos_data = asMultiPoint(cis_data.T)

    # Create shapely Points, we use the z coord as a cheat for holding an index to our original points
    points = MultiPoint([(lon, lat, i)
                         for i, (lat, lon) in enumerate(zip(df.latitude, fixed_lon_points))])

    # If we're only doing this by bounding box then turn the (potentially complex polygon) into a box
    if by_bounding_box:
        region_poly = box(*region_poly.bounds)

    # Perform the actual calculation
    selection = region_poly.intersection(points)

    # Pull out the indices
    mask_indices = np.asarray(selection).T[2].astype(np.int)

    # Build our mask
    mask = np.ones(len(df))
    mask[mask_indices] = 0

    return mask.astype(np.bool)


def apply_new_mask(ungridded_data, new_mask):
    """
    Apply a new mask to an ungridded data object in-place - this mask will be OR'd with any existing mask
    :param ungridded_data: The data object to apply the mask to
    :param new_mask: The new mask to be applied
    """
    from numpy.ma import MaskedArray

    if isinstance(ungridded_data.data, MaskedArray):
        ungridded_data.data.mask = ungridded_data.data.mask | new_mask.astype(np.bool)
    else:
        ungridded_data.data = MaskedArray(ungridded_data.data, mask=new_mask)

    # Be sure to apply the new mask across coordinates as appropriate
    ungridded_data._post_process()


def calculate_cumulative_of_log_normal_dist(mode_radius, mode_width, lower_radius):
    """
        Calculate the fractional area under a log-normal distribution described by a median (mode_radius) and a sigma
         (mode_width). Use a lower cut-off of lower_radius.

        See Eq. 8.39 in Seinfeld & Pandis (2016)

    :param mode_radius: Median radius of mode [m]
    :param mode_width: Width of mode [m]
    :param lower_radius: Lower bound of integral [m]
    :return: Cumulative fraction of the described distribution
    """
    # This scipy erf gives values to within 4dp of Nick's implementation
    from scipy.special import erf

    # Calculate the normal distribution from log-normal shape
    x = np.log(lower_radius/mode_radius) / (np.sqrt(2.0) * np.log(mode_width))
    # Now just return the cumulative of the normalised distribution
    return 1.0 - 0.5*(1.0 + erf(x))


def calculate_cumulative_of_box_dist(mid_radius, width, lower_bound):
    """
        Calculate the fractional area under a box distribution described by a middle radius and a width.
        Use a lower cut-off of lower_radius.

    :param numpy.ndarray mid_radius: Median radius of mode [m]
    :param float width: Width of mode [m]
    :param float lower_radius: Lower bound of integral [m]
    :return: Cumulative fraction of the described distribution
    """
    # This is where the upper end of the box must be
    upper_radius = mid_radius + width / 2.0
    # Calculate the fraction
    frac = (upper_radius - lower_bound) / width
    # Clip between 0.0 and 1.0 for the regions above / below the box.
    return np.clip(frac, 0.0, 1.0)


def set_year(datetime, new_year):
    """
    Change the year of a datetime to some new year. If datetime is a leapday then return None
    :param datetime.datetime datetime: Datetime object to change
    :param int new_year: The new year
    :return: A datetime with the same date as the original except the changed year
    """
    # Once we have the data as datetimes we can just use replace to change the year...
    try:
        new_dt = datetime.replace(year=new_year)
    except ValueError:
        # ...Unless the date is 29th of Feb!
        new_dt = None
    return new_dt


def change_year_of_ungridded_data(data, new_year):
    """
     This slightly roundabout method works fine, but isn't particularly quick.
      I could just add the number of years times 365, but that does't take leap years into account. If I want to take
      leap years into account I can't use fractional days which would break the time. In principle I could take calculate
      the exact difference in integer days between the first date and the first date in the new year then apply that
      scaling - but THAT won't work if the data set spans a leap day...
    :param data: An ungridded data object to update in-place
    :param int new_year: The year to change the data to
    """
    from cis.time_util import convert_std_time_to_datetime, convert_datetime_to_std_time
    import numpy as np

    dates = data.coord('time').data

    dt = convert_std_time_to_datetime(dates)

    np_set_year = np.vectorize(set_year)

    updated_dt = np_set_year(dt, new_year)
    new_dates = convert_datetime_to_std_time(updated_dt)

    data.coord('time').data = new_dates


def calc_diff(x, y):
    """
    Calculate the difference between two ungridded data objects which must have the same coordinates and compatible
    units

    :param x: Original
    :param y: Reference
    :return: (x - y)
    """
    assert x.units == y.units

    diff = x.make_new_with_same_coordinates(data=x.data - y.data,
                                            var_name="difference",
                                            standard_name="",
                                            long_name="({x} - {y})".format(x=x.name(), y=y.name()),
                                            units=y.units)
    return diff


def calc_rel_diff(x, y):
    """
    Calculate the relative difference between two ungridded data objects which must have the same coordinates and
     compatible units

    :param x: Original
    :param y: Reference
    :return: (x - y) / (x + y)
    """
    assert x.units == y.units

    diff = x.make_new_with_same_coordinates(data=(x.data - y.data) / (x.data + y.data),
                                            var_name="relative_difference",
                                            standard_name="",
                                            long_name="({x} - {y}) / ({x} + {y})".format(x=x.name(), y=y.name()),
                                            units='1')
    return diff


def convert_units(ug_data, new_units):
    """
    Convert units of ug_data object to new_units in place
    :param LazyData ug_data:
    :param cf_units.Unit or str new_units:
    :raises ValueError if units can't be converted to standard units, or units are incompatible
    """
    from cf_units import Unit
    if not isinstance(new_units, Unit):
        new_units = Unit(new_units)
    old_units = Unit(ug_data.units)
    old_units.convert(ug_data.data, new_units, inplace=True)
    ug_data.units = new_units


def filename_suffix(f, suffix):
    """
    Append a suffix to a filename, before the extension
    :param str f: Filename (and optionally path)
    :param str suffix: The suffix
    :return str: The full filename with new suffix
    """
    from os.path import splitext
    f, ext = splitext(f)
    return f + suffix + ext


def filename_prefix(prefix, f):
    """
    Prefix a filename
    :param str prefix: The prefix to apply
    :param str f: The filename (this can include a full path)
    :return str: The prefixed file (including path if given)
    """
    from os.path import dirname, basename, join
    path, f = dirname(f), basename(f)
    return join(path, prefix + f)


def dataframe_to_UngriddedData(df, original_ug_data):
    from cis.data_io.write_netcdf import types as valid_types
    from cis.time_util import cis_standard_time_unit
    from cf_units import Unit
    from numpy.ma import MaskedArray
    from cis.data_io.ungridded_data import UngriddedDataList, UngriddedData
    from cis.data_io.Coord import Coord, CoordList
    from copy import copy

    output_data = UngriddedDataList()

    is_single_instance = False

    if not isinstance(original_ug_data, list):
        is_single_instance = True
        original_ug_data = [original_ug_data]

    for d in original_ug_data:
        # Setup the coords
        coords = CoordList()
        for c in d.coords():
            if c.name().lower() == 'time':
                numpy_time_unit = Unit('ns since 1970-01-01T00:00Z')
                new_data = numpy_time_unit.convert(df.index.values.astype('float64'), cis_standard_time_unit)
            else:
                new_data = df[c.name()].values.copy()
            coords.append(Coord(np.asarray(new_data), metadata=copy(c.metadata)))

        if str(df[d.name()].values.dtype) not in valid_types:
            raise ValueError("Unable to recreate UngriddedData object for {} with type {}".format(d.name(),
                                                                                                  df[d.name()].values.dtype))

        if np.asarray(df[d.name()].isnull()).any():
            # Note we specify the mask explicitly for each column because we could have coord values which are valid
            # for some data variables and not others
            new_data = MaskedArray(np.asarray(df[d.name()].values.copy()), mask=np.asarray(df[d.name()].isnull()))
        else:
            new_data = np.asarray(df[d.name()].values.copy())

        output_data.append(UngriddedData(new_data, copy(d.metadata), coords))

    if is_single_instance:
        output_data = output_data[0]

    return output_data


def ungridded_dataframe_wrapper(func):
    """
    Wrap a function which works on dataframes with an UngriddedData->DataFrame converter to allow calling with an
     UngriddedData object.
    :param func: A function which takes a dataframe as its first argument and returns a dataframe
    :return: A function which takes an UngriddedData objects as its first argument and returns an UngriddedData object
    """
    def df_func(ug_data, *args, **kwargs):
        df = ug_data.as_data_frame()
        df = func(df, *args, **kwargs)
        return dataframe_to_UngriddedData(df, ug_data)
    return df_func


@ungridded_dataframe_wrapper
def resample(df, *args, **kwargs):
    """
    Resample a dataframe (UngriddedData) object onto the given period using mean
    :param df:
    :return: Resampled df
    """
    return df.resample(*args, **kwargs).mean()


@ungridded_dataframe_wrapper
def reindex(df, *args, **kwargs):
    """
    Resample a dataframe (UngriddedData) object onto the given period using mean
    :param df:
    :return: Resampled df
    """
    return df.reindex(*args, **kwargs)


def groupby(ug_data, *args, **kwargs):
    """
    Pandas groupby on an UngriddedDataList, returns a grouped UngriddedDataList
    :param ug_data:
    :return: regrouped ungridded data list
    """
    from cis.data_io.ungridded_data import UngriddedDataList
    df = ug_data.as_data_frame()
    return UngriddedDataList([dataframe_to_UngriddedData(d, ug_data) for d in df.groupby(df, *args, **kwargs)])


@ungridded_dataframe_wrapper
def dropna(df, *args, inplace=None, **kwargs):
    """
    Drop invalid elements from a dataframe (UngriddedData) object along a given axis.

    *Note* This can't be done inplace because of the design of the DataFrame wrapper.

    :param pandas.DataFrame df:
    :return: Reduced df
    """
    return df.dropna(*args, inplace=False, **kwargs)


def convert_string_val(s, unit):
    from cf_units import Unit
    import re
    match = re.match('([0-9\.]*)\s?([a-zA-Z]*)', s)
    val, old_unit = match.groups()
    old_unit = Unit(old_unit)
    return old_unit.convert(float(val), unit)


def convert_stp_number_conc_to_ambient(nc, temperature, pressure=None,
                                       standard_temperature=273.15, standard_pressure=1013.0):
    """
    Take a number concentration measured at STP and convert to an ambient concentration in-place using the ideal gas
    law:

    N_s = p_s V / R T_s  -> N_s T_s / p_s = V / R

    N_a = p_a V / R T_a -> N_a T_a / p_a = V / R

    -> N_a T_a / p_a = N_s T_s / p_s

    -> N_a = N_s T_s p_a / p_s T_a

    :param UngriddedData nc:
    :param UngriddedData temperature:
    :param UngriddedData pressure: If None then it is taken from the air_pressure coordinate of nc
    :param float standard_temperature: in Kelvin
    :param float standard_pressure: in hPa (mbar)
    :return:
    """
    from pywork.GASSP.helper_functions import clean_pressure_unit, clean_temperature_unit
    pressure = nc.coord(standard_name='air_pressure') if pressure is None else pressure
    pressure.units = clean_pressure_unit(pressure.units)
    convert_units(pressure, "hPa")
    pressure_ratio = pressure.data / standard_pressure
    # Ensure the temperature units are parseable by cf_units
    temperature.units = clean_temperature_unit(temperature.units)
    convert_units(temperature, "Kelvin")
    temperature_ratio = (temperature.data) / standard_temperature
    nc.data *= (pressure_ratio / temperature_ratio)


def readable_dir(prospective_dir):
    import os
    from argparse import ArgumentTypeError
    if not os.path.isdir(prospective_dir):
        raise ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
    if os.access(prospective_dir, os.R_OK):
        return prospective_dir
    else:
        raise ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def writeable_dir(prospective_dir):
    import os
    from argparse import ArgumentTypeError
    if not os.path.isdir(prospective_dir):
        raise ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
    if os.access(prospective_dir, os.W_OK):
        return prospective_dir
    else:
        raise ArgumentTypeError("readable_dir:{0} is not a writeable dir".format(prospective_dir))


def print_array_differences(x, y, rtol=1e-7, atol=0):
    """
    Pretty print the difference between two arrays x and y. With defined tolerances
    """
    diff = ma_isclose(x, y, rtol=rtol, atol=atol)
    print("\n".join(["{}  !=  {}".format(a, b) for a, b in zip(x[~diff], y[~diff])]))


def ma_isclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
    """
    Copied from numpy.ma.allclose but removed the inf logic and returned the bool array rather than np.all(d)

    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    masked_equal : bool
        Whether to compare masked elements as equal.  If True, masked elements in `a` will be
        considered equal to masked elements in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    Examples
    --------
    >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([True, False])
    >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([True, True])
    >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False, True])
    >>> np.isclose([1.0, np.nan], [1.0, np.nan])
    array([True, True])
    >>> np.isclose([1.0, np.nan], [1.0, np.nan], masked_equal=False)
    array([True, False])
    """
    from numpy.ma import masked_array, filled
    import numpy.core.umath as umath

    x = masked_array(a, copy=False)
    y = masked_array(b, copy=False)

    # make sure y is an inexact type to avoid abs(MIN_INT); will cause
    # casting of x later.
    dtype = np.result_type(y, 1.)
    if y.dtype != dtype:
        y = masked_array(y, dtype=dtype, copy=False)

    d = filled(umath.less_equal(umath.absolute(x - y),
                                atol + rtol * umath.absolute(y)),
               masked_equal)

    return d


def calc_bin_bounds(bin_mid_points):
    """
    Given a set of bin mid-points return the bounds (by linearly interpolating)
    :param numpy.ndarray bin_mid_points: A 1-D array of locations of the center of a bin
    :return: A 1-d array of the bounds of the bins (of length one more than the input array)
    """
    import numpy as np
    from scipy.interpolate import InterpolatedUnivariateSpline

    n = len(bin_mid_points)
    spl = InterpolatedUnivariateSpline(np.arange(n), bin_mid_points, k=1)
    outer_bounds = spl(np.arange(n+1)-0.5)
    return outer_bounds


def ttest_rel_from_stats(diff_mean, diff_std, diff_num):
    """
    Calculates the T-test for the means of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    It is deliberately similar in interface to the other scipy.stats.ttest_... routines

    See e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
    and pg. 140 in Statistical methods in Atmos Sciences
    
    :param diff: The mean difference, x_d (|x1 - x1| == |x1| - |x2|)
    :param diff_std: The standard deviation in the difference, s_d (sqrt(Var[x_d]))
    :param diff_num: The number of points, n (n == n1 == n2)
    :return float, float: t-statistic, p-value
    """
    from scipy.stats import distributions
    
    try:
        from cis.data_io.common_data import CommonData
        # Unpack common data objects
        if isinstance(diff_mean, CommonData):
            diff_mean = diff_mean.data
        if isinstance(diff_std, CommonData):
            diff_std = diff_std.data
        if isinstance(diff_num, CommonData):
            diff_num = diff_num.data
    except:
        pass

    z = diff_mean / np.sqrt(diff_std ** 2 / diff_num)
    # use np.abs to get upper tail, then multiply by two as this is a two-tailed test
    p = distributions.t.sf(np.abs(z), diff_num - 1) * 2
    return z, p


def get_file_dates(files):
    import pandas as pd
    from pywork.parallel_collocate_utils import get_date_from_filename
    df = pd.DataFrame(data={'File': files},
                      index=[get_date_from_filename(fname) for fname in files])
    return df


def split_datagroup_into_daily(args, logger=None, suffix='_daily_'):
    """
    Take a set of cis collocate arguments (as a dict) and expand into many collocation commands for each corresponding
     pair of data and sample files (based on matching YYYYmmdd in the filenames).
    :param args: parsed arguments
    :param logger: logger to log to
    :param str suffix: suffix to add to split output
    :return: expanded list of arguments, one for each day of data
    """
    from cis.data_io.data_reader import expand_filelist
    from copy import deepcopy
    from os.path import dirname, join
    import os.path
    import logging

    logger = logger if logger is not None else logging

    # Get the expanded filenames
    data_files = expand_filelist(args.datagroups[0]['filenames'])

    logger.debug("Expanded files list:")
    logger.debug(data_files)

    # Get a dataframe of the files with the corresponding date as the index
    matched_df = get_file_dates(data_files)

    logger.debug("Date-matched dataframe:")
    logger.debug(matched_df)

    # Generate the new commands
    cmds = []
    for date, files in matched_df.groupby(level=0):
        new_args = deepcopy(args)
        new_args.datagroups[0]['filenames'] = [str(f[0]) for f in files.values]
        first_input_file = new_args.datagroups[0]['filenames'][0]
        new_args.output = join(dirname(first_input_file), date.strftime("%Y%m%d") + suffix + '.nc')

        # Don't process existing files
        if os.path.isfile(new_args.output):
            logger.debug("Skipping {} as it already exists".format(new_args.output))
        else:
            cmds.append(new_args)
            logger.debug("Adding new command: \n" + str(new_args))
    return cmds


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        resdat = np.asarray(result.data)
        result = np.ma.array(resdat, mask=result.mask, copy=False)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        res = np.interp(result, x, y)
        result = np.ma.array(res, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result


def create_spatial_categorical_coord(data, regions):
    from cis.data_io.ungridded_data import UngriddedData, Metadata
    from cis.data_io.Coord import Coord
    import iris.coords
    if isinstance(data, UngriddedData):
        # We have to use flatten rather than flat, GEOS creates a copy of the data if it's a view anyway.
        categories = _categorise_points(data.lon.data.flatten(), data.lat.data.flatten(), regions)
        data._coords.append(Coord(categories, Metadata('Region')))
    else:
        # Using X and Y is a bit more general than lat and lon - the shapefiles needn't actually represent lat/lon
        x, y = np.meshgrid(data.coord(axis='X').points, data.coord(axis='Y').points)
        categories = _categorise_points(x.flat, y.flat, regions)
        new_coord = iris.coords.AuxCoord(categories, units='1', long_name='Region')

        # Add into the cube, spanning both the lat and lon dimensions
        data.add_aux_coord(new_coord, (data.coord_dims(data.coord(axis='X'))[0],
                                       data.coord_dims(data.coord(axis='Y'))[0]))


def _categorise_points(lats, lons, regions):
    from shapely.geometry import MultiPoint

    lat_lon_points = np.vstack([lats, lons])
    points = MultiPoint(lat_lon_points.T)

    cats = np.zeros(lats)

    for i, p in enumerate(points):
        # Performance in this loop might be an issue. I could use a GeoPandas spatial join which should be quicker,
        #  as long as I have rtree installed (see http://geopandas.org/mergingdata.html?highlight=spatial%20join)
        cats[i] = next(region.name() for region in regions if p.intersects(region))


def stratify_echam(data, pressure_bins=None, pressure_range=(200,1000,25)):
    from iris.experimental import stratify
    from cis.data_io.gridded_data import make_from_cube
    import numpy.ma as ma
    # Put the hybrid levels onto constant pressure levels
    p_levels = pressure_bins if pressure_bins is not None else np.linspace(*pressure_range)
    cube = stratify.relevel(data, data.coord('air_pressure'), p_levels, 'hybrid level at layer midpoints')
    # Remove the surface pressure (which remains as an aux coord)
    cube.remove_coord("surface pressure")
    # Mask NaN values
    cube.data = ma.masked_invalid(cube.data)
    return make_from_cube(cube)


def namedtuple_with_defaults(typename, field_names, default_values=()):
    import collections
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def ordinal_suffix(n):
    """
    Take a number and add the ordinal suffix (e.g. 'st' in 1st)
    :param int n: Number to use
    :return str: Number plus ordinal
    """
    return str(n)+("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def compress_ungridded_data(data, use_first=False):
    """
    Compress (remove all masked elements from) an UngriddedData / UngriddedDataList

    Given an UngriddedDataList will compress based on the combined mask, unless 'use_first' is True in which case
    the mask of the first element in the list will be used.

    :param UngriddedData data:
    :param bool use_first: Use the mask from the first element in the list as the combined mask?
    :return:
    """
    from cis.data_io.ungridded_data import UngriddedDataList
    from operator import and_
    from functools import reduce

    if isinstance(data, list):
        if use_first:
            combined_mask = data[0].data.mask
        else:
            combined_mask = reduce(and_, [d.data.mask for d in data], True)
        return UngriddedDataList([d[~combined_mask] for d in data])
    else:
        return data[~data.data.mask]


def get_solar_zenith_angles(data):
    """
    Calculate the solar zenith angle across all the points in a dataset (assumes it has time, lat and lon coordinates)

    :param UngriddedData data:
    :return ndarray: The solar zenith angle of the points
    """
    import pandas as pd
    from cf_units import Unit
    from cis.time_util import cis_standard_time_unit
    from cis.data_io.ungridded_data import _to_flat_ndarray

    time_coord = data.coord('time')
    # Store the original shape then flatten the data for converting to a Pandas index
    original_shape = time_coord.shape

    if str(time_coord.units).lower() == 'datetime object':
        time_points = _to_flat_ndarray(time_coord.data)
    elif isinstance(time_coord.units, Unit):
        time_points = time_coord.units.num2date(_to_flat_ndarray(time_coord.data))
    else:
        time_points = cis_standard_time_unit.num2date(_to_flat_ndarray(time_coord.data))

    dates = pd.DatetimeIndex(time_points)

    # At somepoint between pandas 0.19 and 0.20 these attributes changed their return type - this seems to be safe
    calendar_day = np.array(dates.dayofyear)
    hour = np.array(dates.hour)

    # solar declination angle
    lmda = 360. * (calendar_day / 365.24)  # degrees
    DEC = np.sin(23.5 * np.pi / 180.)*np.sin(lmda *np.pi / 180)  # radians

    # hour angle
    GMT = (24. / 360.) * hour
    h = (((hour + (24. / 360. * data.lon.data.ravel())) - 12.) / 24.) * 360.

    cosT = np.sin(data.lat.data.ravel() * np.pi / 180.)*np.sin(DEC) + \
           np.cos(data.lat.data.ravel() * np.pi / 180.)*np.cos(DEC) * np.cos(h * np.pi / 180.)

    theta = np.arccos(cosT) * 180. / np.pi

    # Reshape the array and return
    return theta.reshape(original_shape)


def is_daytime(data):
    """
    Use spatio-temporal information to infer the solar zenith angle. An angle < 90 degrees is considered day-time

    :param UngriddedData data:
    :return mask: A mask which is true everywhere the computed solar zenith angle is < 90 degrees
    """
    return get_solar_zenith_angles(data) < 90.0


def get_time_delta(time_coord):
    """
    Return the unique timestep from a time coordinate, or the non-unique timesteps for monthly data

    :param cis.data_id.coords.Coord time_coord:
    :return ndarry: Array of timesteps
    :raises ValueError when the step can't be determined or a non-regular period is found
    """
    import datetime
    time_delta = np.unique(np.diff(time_coord.units.num2date(time_coord.points)))
    if len(time_delta) == 0:
        if time_coord.has_bounds():
            time_delta, = np.diff(time_coord.units.num2date(time_coord.bounds))[0]
        else:
            raise ValueError("Cannot guess time step from a single one without bounds")
    elif len(time_delta) == 1:
        time_delta = time_delta[0]
    elif (np.amin(time_delta) >= datetime.timedelta(days=28) and
                np.amax(time_delta) <= datetime.timedelta(days=31)):
        # Monthly timedelta
        time_delta = datetime.timedelta(days=30)
    else:
        raise ValueError("Non-uniform period (%g to %g) between timesteps" % (
            np.amin(time_delta), np.amax(time_delta)))
    return time_delta


def subset_dataframe(df, subset_args):
    for col, args in subset_args.items():
        start, stop = args.start, args.stop if isinstance(args, slice) else args
        df = df[(df[col] >= start) & (df[col] < stop)]
    return df


def smart_post_process(self):
    """
    Perform a post processing step on lazy loaded Ungridded Data preserving the shape of 2-D arrays

    :return:
    """
    from cis.data_io.ungridded_data import UngriddedData
    # Load the data if not already loaded
    if self._data is None:
        self._data = self.retrieve_raw_data(self._data_manager[0])
        if len(self._data_manager) > 1:
            for manager in self._data_manager[1:]:
                self._data = np.ma.concatenate((self._data, self.retrieve_raw_data(manager)), axis=0)

    # Remove any points with missing coordinate values:
    combined_mask = np.zeros(self._data.shape[0], dtype=bool)
    # Don't take into account the aux coord
    for coord in self._coords[:-1]:
        combined_mask |= np.ma.getmaskarray(coord.data[:, 0])
        if coord.data.dtype != 'object':
            combined_mask |= np.isnan(coord.data[:, 0])
    if combined_mask.any():
        n_points = np.count_nonzero(combined_mask)
        print("Identified {n_points} point(s) which were missing values for some or all coordinates - "
              "these points have been removed from the data.".format(n_points=n_points))
        new_coords = []
        for c in self._coords:
            new_coords.append(c[~combined_mask, :])

        self = UngriddedData(data=self._data[~combined_mask, :], metadata=self.metadata, coords=new_coords)

    return self


def get_time_series(data, kernel='mean'):
    """
    Get the time series of a multi-dimensional dataset by collapsing over all dimensions except time

    :param data:
    :return:
    """
    from cis.data_io.gridded_data import make_from_cube, GriddedDataList
    from cis.utils import listify
    from iris.cube import Cube
    from iris.analysis import MEAN
    coords_to_collapse = [c for c in data.coords(dim_coords=True) if c.standard_name != 'time']
    # Remove the aux coords and aux factories which can be a pain to collapse
    bare_data = GriddedDataList([d.copy() for d in listify(data)])
    for d in bare_data:
        if d._aux_factories:
            d._aux_factories.pop()
        for c in d.coords(dim_coords=False):
            d.remove_coord(c)

    # coords_to_collapse = ['longitude', 'latitude']
    # This will only add it if it's present
    #coords_to_collapse.extend(data.coords(standard_name='air_pressure'))
    # print(coords_to_collapse)
    #result = GriddedDataList()
    #for d in listify(data):
    #    c = Cube.collapsed(d, coords_to_collapse, MEAN)
    #    print(c)
    #    result.append(make_from_cube(c))
    result = bare_data.collapsed(coords_to_collapse, 'mean')
    return result


def global_mean(cube):
    """
    Return the (weighted) global mean of the cube
    :param cube:
    :return:
    """
    import iris.analysis.cartography
    # Guess bounds and calculate weights
    if cube.coord('latitude').bounds is None:
        cube.coord('latitude').guess_bounds()
    if cube.coord('longitude').bounds is None:
        cube.coord('longitude').guess_bounds()
    weights = iris.analysis.cartography.area_weights(cube)
    cube.coord('longitude').bounds = None
    cube.coord('latitude').bounds = None
    return cube.collapsed(['longitude', 'latitude', 'time'], iris.analysis.MEAN, weights=weights)
