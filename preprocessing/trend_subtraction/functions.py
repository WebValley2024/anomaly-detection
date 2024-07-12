import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import xarray as xr
from scipy.optimize import curve_fit
import pandas as pd
from zarr import Blosc

# START = 1577893961
# END = 1633042861


def block_coordinates(data):
    '''
    This function will take an xarray dataset and will return a dictionary 
    with the block coordinates as keys and the indices of the data points in that
    block as values.

    Parameters:
    data: xarray dataset

    Returns:
    dict
    '''
    arr = {}

    for i in range(len(data["Block_x"])):
        x_coord = int(data["Block_x"][i].data)
        y_coord = int(data["Block_y"][i].data)
        if not x_coord in arr:
            arr[x_coord] = {}

        inside_dict = arr[x_coord]
        if not y_coord in inside_dict:
            inside_dict[y_coord] = []
        inside_dict = inside_dict[y_coord]

        inside_dict.append(i)

    return arr

def files_to_xarray(path, night, instrument):
    '''
    This function will take a path to a directory containing zarr files and will 
    return an xarray dataset
    
    Parameters:
    path: str
        The path to the directory containing the zarr files
    night: bool
        If True, the function will only consider the night time data
    instrument: str
        The instrument name

    Returns:
    xarray dataset
    '''
    
    index = 0
    data = None
    for file in os.listdir(path):
        if night:
            a = instrument+"_1"
        else:
            a = instrument+"_0"
        if not file.startswith(a):
            continue
        thing = xr.open_zarr(path + file)
        if index == 0:
            data = thing
        else:
            data = xr.concat([data, thing], dim="ID", data_vars="all")
        index += 1

    return data

def convert_unix_time_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time)

def resample(time, count, resample_time):
    '''
    This function will take a time series and will resample it to the given time 
    resolution.
    
    Parameters:
    time: list
        The time series
    count: list
        The count series
    resample_time: str
        The time resolution to resample to
        
    Returns:
        list, list
    '''
    time = [convert_unix_time_to_datetime(i) for i in time]
    time = pd.Series(count, index=time)
    time = time.resample(resample_time).mean()
    time = time.dropna()
    return time.index, time.values

def count_bg(range_data, range_arr, attribute, angles=None, END=None):
    '''
    This function will take an xarray dataset and will calculate the background for
    the count data.
    
    Parameters:
    data: xarray dataset
        The dataset containing the count data
    arr: dict
        The block coordinates
    attribute: str
        The attribute name
    angles: int
        The number of angles

    Returns:
    mean: np.array
        The background for the count data
    '''

    if angles is None:
        mean = np.zeros((10,10))
    else:
        mean = np.zeros((10,10, angles))

    for blockx in range_arr:
        for blocky in range_arr[blockx]:
            block = range_arr[blockx][blocky]
            if angles is None:
                count = range_data[attribute][block].data
                mean[blockx, blocky] = np.mean(count)
            else:
                for i in range(0,angles):
                    count = range_data[attribute + str(i)][block].data
                    mean[blockx, blocky, i] = np.mean(count)

    return mean

def subtract_count_bg(data, arr, mean, attribute, angles=None):
    '''
    This function will take an xarray dataset and will subtract the background from
    the count data.
    
    Parameters:
    data: xarray dataset
        The dataset containing the count data
    arr: dict
        The block coordinates
    dest_path: str
        The path to save the new dataset
    '''

    for blockx in arr:
        for blocky in arr[blockx]:
            block = arr[blockx][blocky]    
            if angles is None:
                data[attribute][block] = data[attribute][block].data - mean[blockx, blocky]
                [blockx, blocky]
            else:
                for i in range(0,angles):
                    instrum = attribute + str(i)
                    data[instrum][block] = data[instrum][block].data - mean[blockx, blocky, i]
                
def meadium_spectra(range_data, range_arr, attribute, END=None):
    '''
    This function will calculate the medium spectra for each block in the given xarray dataset

    Parameters:
    data: xarray dataset
        The dataset containing the spectra data
    arr: dict
        The block coordinates
    attribute: str
        The attribute name
    
    Returns:
    medium_spectra: np.array
        the medium spectra for each block
    '''
    medium_spectra= np.zeros((10,10,3, 9))
    for blockx in range_arr:
        if (blockx > 10) or (blockx < 0):
            continue
        for blocky in range_arr[blockx]:
            if (blocky > 10) or (blocky < 0):
                continue
            block = range_arr[blockx][blocky] 

            time = range_data["TIME"][block].data

            spectra_block = np.zeros((len(time), 3, 9))
            for i_energy in range(0,3):
                for j_pitch in range(0,9):
                    spectra_block[:, i_energy, j_pitch] = range_data[attribute + str(i_energy) + "_" + str(j_pitch)][block].data
            medium_spectra[blockx, blocky] = np.mean(spectra_block, axis=0)

    return medium_spectra

def subtract_medium_spectra(data, arr, medium_spectra, attribute):
    '''
    This function will subtract the medium spectra from the given xarray dataset
    
    Parameters:
    data: xarray dataset
        The dataset containing the spectra data
    arr: dict
        The block coordinates
    medium_spectra: np.array
        The medium spectra
    attribute: str
        The attribute name
    '''

    for blockx in arr:
        if (blockx > 10) or (blockx < 0):
            continue
        for blocky in arr[blockx]:
            if (blocky > 10) or (blocky < 0):
                continue
            block = arr[blockx][blocky] 

            for i_energy in range(0,3):
                for j_pitch in range(0,9):
                    instrum = attribute + str(i_energy) + "_" + str(j_pitch)
                    data[instrum][block] = data[instrum][block].data - medium_spectra[blockx, blocky, i_energy, j_pitch]

def write_to_zarr(data, dest_path):
    '''
    This function will write the given xarray dataset to a zarr file

    Parameters:
    data: xarray dataset
        The dataset to write
    dest_path: str
        The path to save the dataset
    '''
    
    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.AUTOSHUFFLE)
    data.to_zarr(dest_path, mode='w', encoding={k: {"compressor": compressor} for k in data.data_vars})

def drop_unwanted_data(data, start=None, end=None):
    '''
    This function will drop all data points before 2020. This is done because the data before 2020 has
    a unusual behavior.
    
    Parameters:
    data: xarray dataset
        The dataset to drop the data points from

    Returns:
    data: xarray dataset
    '''
    if end is not None:
        data = data.where(data["TIME"] < end, drop=True)
    if start is not None:
        data = data.where(data["TIME"] > start, drop=True)
    return data

def plot_all_blocks(data, arr, attr, resample_time="1d"):
    '''
    This function will plot the given attribute for all blocks in the given xarray dataset

    Parameters:
    data: xarray dataset
        The dataset containing the attribute data
    arr: dict
        The block coordinates
    attr: str
        The attribute name
    '''

    fig, axs = plt.subplots(10, 10, figsize=(30, 30))

    for blocky in arr:
        if blocky > 10 or blocky < 0:
            continue
        for blockx in arr[blocky]:
            if blockx > 10 or blockx < 0:
                continue
            block = arr[blocky][blockx]
            time = data["TIME"][block].data
            count = data[attr][block].data
            resampled_time, resampled_counts = resample(time, count, resample_time)

            axs[blockx, blocky].scatter(resampled_time, resampled_counts, s=3)
            axs[blockx, blocky].set_xlabel("Time")
            axs[blockx, blocky].set_ylabel("Count")

            axs[blockx, blocky].set_title(attr + " - " + str(blockx) + "," + str(blocky))

    plt.show()
