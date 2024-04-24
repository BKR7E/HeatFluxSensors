"""This code calculates heat transfer coefficient[W/m^2-K] from angle[deg] and velocity[m/s] from data using a FluxTeq PHFS-01 heat flux sensor. 
Populated with data from AngleVelHtc.csv
Coded by Eric Alar, UW-Madison (4/24/24) with help from ChatGPT 3.5"""

import subprocess
import sys

"""Checking for and installing necessary libraries
****************************************************
"""
def check_and_install(package_name):
    try:
        # Try to import the package
        __import__(package_name)
        # print(f"{package_name} is already installed.")
    except ImportError:
        # If package is not found, install it
        print(f"{package_name} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# List of packages to check and potentially install
packages = [
    "numpy",  # numpy is imported as np but the package name is numpy
    "pandas",
    "scipy",  # griddata is part of scipy
]

for package in packages:
    check_and_install(package)
    
"""
****************************************************
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

#This takes 3 columns of data (30,000 + rows), organizes the data into bins, then interpolates the data set. It preserves the original points, but fills in the missing data
#so that the data can be plotted in a smooth 3D surface plot - this version is formatted for (x,y,z) corresponding to (angle[degrees], HTC[w/m^2-K], velocity[m/s])

"""User inputs
****************************************************
"""
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('AngleVelHtc.csv', parse_dates=True)

# define the number of x and y bins
x_bin_N = 50
y_bin_N = 50

# Define the points where you want to interpolate to find a heat transfer coefficient [W/m^2-K]
x_interpolate = 0 # Replace with your angle in degrees
y_interpolate = 4 # Replace with your velocity [m/s]

"""
****************************************************
"""

# Extract columns 1, 2, and 3
x = df.iloc[:, 0].values #Angle
y = df.iloc[:, 1].values #Nusselt #
z = df.iloc[:, 2].values #Velocity

# Reshape arrays to be 2-dimensional
x = x.reshape((-1, 1))
y = y.reshape((-1, 1))
z = z.reshape((-1, 1))

x_data = np.array(x)
y_data = np.array(y)
z_data = np.array(z)

#Incrementing from lowest to highest values in uniform increments for each bin - this is only to remake the bins for graphing
x_bins_test = np.arange(min(x_data)-.001, max(x_data), ((max(x_data)-min(x_data))/x_bin_N)+.001/(x_bin_N+1))

# Create x and y arrays for bin centers
x_centers = (x_bins_test[:-1] + x_bins_test[1:]) / 2

#Incrementing from lowest to highest values in uniform increments for each bin - this is only to remake the bins for graphing
y_bins_test = np.arange(min(y_data)-.001, max(y_data), ((max(y_data)-min(y_data))/y_bin_N)+.001/(y_bin_N+1))
# Create x and y arrays for bin centers
y_centers = (y_bins_test[:-1] + y_bins_test[1:]) / 2

# define the x and y bins using pd.cut()
x_bins = pd.cut(df['x'], x_bin_N)
y_bins = pd.cut(df['y'], y_bin_N)

# group the data by x_bins and y_bins
grouped = df.groupby([x_bins, y_bins])

# get the number of data points in each bin
counts = grouped.size()

# get the standard deviation of 'z' in each bin
stds = grouped['z'].std()

# convert counts to a DataFrame
counts_df = np.reshape(counts.values, (x_bin_N, y_bin_N))

# convert stds to a DataFrame
stds_df = np.reshape(stds.values, (x_bin_N, y_bin_N))

z_means = grouped['z'].mean()

# convert the mean values to a 2D array
z_array = np.reshape(z_means.values, (x_bin_N, y_bin_N))

# create the 2D table of values with NaNs
arr_interp = z_array.copy()

# find the indices of the non-NaN values in the array
not_nan_indices = np.array(np.where(~np.isnan(arr_interp))).T

# create a meshgrid of all indices in the non-NaN region
all_indices = np.indices(arr_interp.shape).transpose(1, 2, 0).reshape(-1, 2)

# interpolate the NaN values using griddata
interpolated_values = griddata(not_nan_indices, arr_interp[not_nan_indices[:, 0], not_nan_indices[:, 1]],
                               all_indices, method='cubic')

# reshape the interpolated values to the shape of the original array
interpolated_values = interpolated_values.reshape(arr_interp.shape)

# replace the NaN values with the interpolated values
arr_interp[np.isnan(arr_interp)] = interpolated_values[np.isnan(arr_interp)]

# # Reverse the order of the 1D array
x_centers_r = x_centers[::-1]

# Create a meshgrid from x and y values
x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)

# Flatten the 2D arrays to 1D arrays
x_flat = x_mesh.flatten()
y_flat = y_mesh.flatten()
z_flat = arr_interp.flatten(order='F')

def interpolate_value(x_interpolate, y_interpolate):
    x_interpolate = x_centers[x_bin_N-1] if x_interpolate > x_centers[x_bin_N-1] else x_interpolate #Constrain angle to less than 90
    x_interpolate = x_centers[0] if x_centers[0] > x_interpolate else x_interpolate #Constrain angle to greater than 0.5 degrees
    y_interpolate = y_centers[y_bin_N-1] if y_interpolate > y_centers[y_bin_N-1] else y_interpolate #Constrain Reynolds number to less than the max in the data set
    y_interpolate = y_centers[0] if y_centers[0] > y_interpolate else y_interpolate #Constrain Reynolds number to greater than the min in the data set
    # Perform 2D interpolation using griddata
    interpolated_value = griddata((x_flat, y_flat), z_flat, (x_interpolate, y_interpolate), method='linear')
    return interpolated_value

result = interpolate_value(x_interpolate,y_interpolate)
print('Heat transfer coefficient:',result,'W/m^2-K')


