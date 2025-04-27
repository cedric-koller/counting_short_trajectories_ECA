import matplotlib.pyplot as plt

def plot_ECA(states, title=None):
    fig=plt.figure(figsize=(10,10))
    ax=plt.axes()
    ax.set_axis_off()
    fig.suptitle(title)
    ax.grid(color='k', linestyle='-', linewidth=1)
    ax.set_aspect('equal')
    ax.imshow(states, interpolation='none', cmap='RdPu');

import matplotlib.pyplot as plt

def plot_ECA(states, title=None, show_grid=False):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_axis_off()
    fig.suptitle(title)

    if show_grid:
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    else:
        ax.grid(False)

    ax.set_aspect('equal')
    ax.imshow(states, interpolation='none', cmap='binary');  # 'binary' is black and white

import matplotlib.pyplot as plt
import numpy as np

def plot_ECA(states, title=None, show_grid=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    fig.suptitle(title)

    ax.set_aspect('equal')
    ax.imshow(states, interpolation='none', cmap='binary')

    if show_grid:
        # Draw fine grid lines over the cells
        nrows, ncols = states.shape
        for y in range(nrows + 1):
            ax.axhline(y - 0.5, color='lightgrey', linewidth=0.5)
        for x in range(ncols + 1):
            ax.axvline(x - 0.5, color='lightgrey', linewidth=0.5)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_ECA(states, title=None, show_grid=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    fig.suptitle(title)

    ax.set_aspect('equal')
    ax.imshow(states, interpolation='none', cmap='binary')

    nrows, ncols = states.shape

    if show_grid:
        # Draw fine grid lines over the cells
        for y in range(nrows + 1):
            ax.axhline(y - 0.5, color='lightgrey', linewidth=0.5)
        for x in range(ncols + 1):
            ax.axvline(x - 0.5, color='lightgrey', linewidth=0.5)

    # Add a black frame around the image
    rect = patches.Rectangle(
        (-0.5, -0.5), ncols, nrows,
        linewidth=4, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_ECA(states, title=None, show_grid=False):
    nrows, ncols = states.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(title)

    # Draw the image
    ax.imshow(states, cmap='binary', interpolation='none',
              extent=(0, ncols, nrows, 0))

    ax.set_aspect('equal')

    # Configure ticks to match cell edges
    ax.set_xticks(np.arange(0, ncols + 1), minor=True)
    ax.set_yticks(np.arange(0, nrows + 1), minor=True)

    if show_grid:
        ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    else:
        ax.grid(False)

    # Hide major ticks and labels
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add black frame around the entire image
    rect = patches.Rectangle(
        (0, 0), ncols, nrows,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

    # Set limits to match exactly
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)
