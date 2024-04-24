"""This code plots Nusselt # as a function of Angle and Reynolds #; this is the same code used in dataBinner.py
Coded by Eric Alar, UW-Madison (4/24/24) with help from ChatGPT 3.5
"""

import subprocess
import sys

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
    "matplotlib",  # both pyplot and cm are part of matplotlib
    "pandas",
    "scipy",  # griddata is part of scipy
    "plotly"  # both graph_objs and io are part of plotly
]

for package in packages:
    check_and_install(package)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objs as go
import plotly.io as pio

#This takes 3 columns of data (30,000 + rows), organizes the data into bins, then interpolates the data set. It preserves the original points, but tries to fill in the missing data
#so that the data can be plotted in a smooth 3D surface plot - this version is formatted for (x,y,z) corresponding to (angle, Reynolds #, Nusselt #)

"""User Inputs
*********************************** 
"""
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('UserData.csv', parse_dates=True)

#define the number of x and y bins
x_bin_N = 20   
y_bin_N = 10

#OPTIONAL(True/False): Define a point where you want to interpolate to find the corresponding Nusselt#:
interpolationcall = True
x_interpolate = 5 # Replace with your desired x-coordinate
y_interpolate = 80000 # Replace with your desired y-coordinate
"""
*********************************** 
"""

# Extract columns 1, 2, and 3
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
z = df.iloc[:, 2].values

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

# Reverse the order of the 1D array
x_centers_r = x_centers[::-1]

"""First figure
**************************************************
"""
# create a 3D scatter plot of the z_array
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_centers, y_centers)

# print("mesh X", X)
# print("mesh Y", Y)
ax.scatter(X, Y, z_array.T)

# set the viewing position to an azimuth angle of 30 degrees and an elevation angle of 45 degrees
ax.view_init(azim=45, elev=25)

#Eliminating the perspective view
ax.set_box_aspect([1, 1, .7])  # set aspect ratio
ax.set_proj_type('ortho')  # set orthographic projection

# set the axis labels
ax.set_ylabel('RE')
ax.set_xlabel('Angle (degrees)')
ax.set_zlabel('Nu')

# Add a title to the plot
ax.set_title('User Provided Data Organized into Bins')
"""
**************************************************
"""

"""Second figure
*************************************************
"""
# create a 3D scatter plot of the z_array
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_centers, y_centers)

ax2.scatter(X, Y, arr_interp.T)
ax2.scatter(X, Y, z_array.T)

# set the viewing position to an azimuth angle of 30 degrees and an elevation angle of 45 degrees
ax2.view_init(azim=45, elev=25)

#Eliminating the perspective view
ax2.set_box_aspect([1, 1, .7])  # set aspect ratio
ax2.set_proj_type('ortho')  # set orthographic projection

# set the axis labels
ax2.set_ylabel('RE')
ax2.set_xlabel('Angle (degrees)')
ax2.set_zlabel('Nu')

# Add a title to the plot
ax2.set_title('Data Interpolation')
"""
*************************************************
"""

"""Third figure
*************************************************
"""
# create a figure and 3D axes
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

# set the viewing position to an azimuth angle of 30 degrees and an elevation angle of 45 degrees
ax3.view_init(azim=45, elev=25)

#Eliminating the perspective view
ax3.set_box_aspect([1, 1, .7])  # set aspect ratio
ax3.set_proj_type('ortho')  # set orthographic projection

# set the x, y, and z limits of the plot
ax3.set_xlim([min(x_centers)-0.5, max(x_centers)+0.5])
ax3.set_ylim([min(y_centers)-0.5, max(y_centers)+0.5])
ax3.set_zlim([0, max(counts_df.T.flatten())+1])

# create the 3D bar chart
dz = np.array(counts_df.T).flatten()
x, y = X.flatten(), Y.flatten()

# calculate the range of the x and y data
x_range = max(X.flatten()) - min(X.flatten())
y_range = max(Y.flatten()) - min(Y.flatten())

# calculate the size of the x and y bins
x_bin_size = x_range / x_bin_N
y_bin_size = y_range / y_bin_N

# set the width and depth of the bars based on the bin size
dx = x_bin_size * 0.8
dy = y_bin_size * 0.8

ax3.bar3d(x, y, 0, dx, dy, dz, color='blue', alpha=0.8)

# set the axis labels
ax3.set_ylabel('Re')
ax3.set_xlabel('Angle (degrees)')
ax3.set_zlabel('# of Data Points')

# Add a title to the plot
ax3.set_title('Number of Data Samples in Each Bin')
"""
*************************************************
"""

"""Fourth figure
*************************************************
"""
# create a figure and 3D axes
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')

# set the viewing position to an azimuth angle of 30 degrees and an elevation angle of 45 degrees
ax4.view_init(azim=45, elev=25)

#Eliminating the perspective view
ax4.set_box_aspect([1, 1, .7])  # set aspect ratio
ax4.set_proj_type('ortho')  # set orthographic projection

# set the x, y, and z limits of the plot
ax4.set_xlim([min(x_centers)-0.5, max(x_centers)+0.5])
ax4.set_ylim([min(y_centers)-0.5, max(y_centers)+0.5])
ax4.set_zlim([0, max(stds_df.T.flatten())+1])

# create the 3D bar chart
dz = np.array(stds_df.T).flatten()
x, y = X.flatten(), Y.flatten()

# calculate the range of the x and y data
x_range = max(X.flatten()) - min(X.flatten())
y_range = max(Y.flatten()) - min(Y.flatten())

# calculate the size of the x and y bins
x_bin_size = x_range / x_bin_N
y_bin_size = y_range / y_bin_N

# set the width and depth of the bars based on the bin size
dx = x_bin_size * 0.8
dy = y_bin_size * 0.8

ax4.bar3d(x, y, 0, dx, dy, dz, color='blue', alpha=0.8)

# set the axis labels
ax4.set_ylabel('RE')
ax4.set_xlabel('Angle (degrees)')
ax4.set_zlabel('Standard Deviation')

# Add a title to the plot
ax4.set_title('Data Standard Deviation')
"""
*************************************************
"""

"""Fifth figure
*************************************************
"""
# create a new figure for the surface plot
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')

# create a surface plot of the z_array
surf = ax5.plot_surface(X, Y, arr_interp.T, cmap=cm.turbo,
                        linewidth=0, antialiased=False)

# set the viewing position to an azimuth angle of 30 degrees and an elevation angle of 45 degrees
ax5.view_init(azim=45, elev=25)

#Eliminating the perspective view
ax5.set_box_aspect([1, 1, .7])  # set aspect ratio
ax5.set_proj_type('ortho')  # set orthographic projection

# set the axis labels
ax5.set_ylabel('Re')
ax5.set_xlabel('Angle (degrees)')
ax5.set_zlabel('Nu')

# Add a title to the plot
ax5.set_title('Surface Plot of all Data')

# add a color bar which maps values to colors
fig5.colorbar(surf, shrink=0.5, aspect=5)
"""
*************************************************
"""

"""This creates an HTML surface plot that can be opened using a browser
**********************************************************************
"""
surface_plot = go.Surface(x = y_centers, y = x_centers, z=arr_interp)  

#Create a figure
fig = go.Figure(data=[surface_plot])

#Set layout options (optional)
fig.update_layout(
    title='Nusselt # as a function of RE# and Angle',
    scene=dict(
        xaxis_title='Re',
        yaxis_title='Angle (degrees)',
        zaxis_title='Nu'
    )
)
#Save the plot as an interactive HTML file
pio.write_html(fig, 'SurfacePlot.html')
"""
**********************************************************************
"""

"""Interpolator function that you can send individual points to, or even arrays
**********************************************************************
"""
# Flatten the 2D arrays to 1D arrays
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = arr_interp.flatten(order='F')


def interpolate_value(x_interpolate, y_interpolate):
    x_interpolate = x_centers[x_bin_N-1] if x_interpolate > x_centers[x_bin_N-1] else x_interpolate #Constrain angle to less than 90
    x_interpolate = x_centers[0] if x_centers[0] > x_interpolate else x_interpolate #Constrain angle to greater than 0.5 degrees
    y_interpolate = y_centers[y_bin_N-1] if y_interpolate > y_centers[y_bin_N-1] else y_interpolate #Constrain Reynolds number to less than the max in the data set
    y_interpolate = y_centers[0] if y_centers[0] > y_interpolate else y_interpolate #Constrain Reynolds number to greater than the min in the data set
    # Perform 2D interpolation using griddata
    interpolated_value = griddata((x_flat, y_flat), z_flat, (x_interpolate, y_interpolate), method='linear')
    return interpolated_value


# Check the condition before executing the function
if interpolationcall:
#Calling function:
    result = interpolate_value(x_interpolate,y_interpolate)
    print('Interpolated result:', result)

"""
**********************************************************************
"""

#shows the other plots
plt.show()



