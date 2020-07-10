import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.interpolate import griddata


def set_tex_fonts():
    plt.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cm'
    mpl.rcParams['mathtext.fontset'] = 'cm'


def get_slice(x, u, xloc=None, yloc=None, domain=None):
    if domain is None:
        x_coords = np.linspace(0, np.max(x[:, 0]), int(np.max(x[:, 0])))  # interpolate to 1 meter cells
        y_coords = np.linspace(0, np.max(x[:, 1]), int(np.max(x[:, 1])))
    else:
        x_coords = np.linspace(0, domain[0], domain[0])  # interpolate to 1 meter cells
        y_coords = np.linspace(0, domain[1], domain[1])

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_u = griddata(x[:, 0:2], u[:, :], (grid_x, grid_y), method='cubic')
    if xloc is None and yloc is not None:
        x_slice = grid_y[:,int(yloc)]
        u_slice = grid_u[:,int(yloc),:]
    elif xloc is not None and yloc is None:
        x_slice = grid_x[int(xloc),:]
        u_slice = grid_u[int(xloc), :,:]

    return x_slice, u_slice


def plot_contours(x, u, ax=None, levels=None,domain=None,type='velocity'):
    if domain is None:
        x_coords = np.linspace(0, np.max(x[:, 0]), int(np.max(x[:, 0])))  # interpolate to 1 meter cells
        y_coords = np.linspace(0, np.max(x[:, 1]), int(np.max(x[:, 1])))
    else:
        x_coords = np.linspace(0, domain[0], domain[0])  # interpolate to 1 meter cells
        y_coords = np.linspace(0, domain[1], domain[1])

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    if type == 'velocity' or type == 'force':
        grid_u = griddata(x[:, 0:2], u[:, :], (grid_x, grid_y), method='nearest')
        vnorm = np.sqrt(np.sum(grid_u**2,2))
    elif type == 'pressure':
        vnorm = griddata(x[:, 0:2], u, (grid_x, grid_y), method='linear')

    contour = ax.contourf(grid_x, grid_y, vnorm, levels=levels,extend='both')
    for c in contour.collections:
        c.set_edgecolor("face")
    ax.set_aspect("equal")
    return ax,contour


def rot(yaw):
    # cy = np.cos(yaw)
    # sy = np.sin(yaw)
    # rot = np.array([[cy, -sy], [sy, cy]])
    rot = np.array([
        [-np.sin(yaw), np.cos(yaw)],
        [-np.cos(yaw), -np.sin(yaw)]
        ])
    return rot


def plot_turbines(pos,radius,yaw,ax):
    yaw = np.deg2rad(yaw)

    for idx in range(len(pos)):
        blade = radius * rot(yaw[idx]).dot(np.array([0,1]).transpose())
        # pos = np.
        tip_0 = pos[idx] + blade
        tip_1 = pos[idx] - blade

        ax.plot([tip_0[0], tip_1[0]], [tip_0[1], tip_1[1]],'k-',lw=1)
    return ax

def plot_power(ax,time_sowfa, power_sowfa, time_wasp, power_wasp, power_floris):
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, 3))
    ax.plot(time_sowfa,power_sowfa,c=colours[0],ls='-.',lw=1,label="SOWFA")
    ax.plot(time_wasp, power_wasp,c=colours[1], ls='-', ms=1, lw=1,label="Control model")
    if power_floris is not None:
        ax.axhline(power_floris,c=colours[2],ls='--',lw=1,label="FLORIS")
    ax.set_xlim([time_wasp[0],time_wasp[-1]])
    ax.set_ylim(bottom=0)
    return ax

def plot_jacobian(djdm):
    fig = plt.figure(figsize=(4,3))
    grid = AxesGrid(fig,
                    rect=(0.1, 0.1, 0.80, 0.80),
                    nrows_ncols=(1, 1),
                    axes_pad=0.1,
                    cbar_mode='each',
                    cbar_location='right',
                    cbar_pad=0.2,
                    cbar_size=0.15
                    )
    vmax = np.abs(djdm).max()
    colours = grid[0].imshow(djdm,cmap='seismic',vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(colours, grid[0].cax)
    cb.solids.set_edgecolor("face")


def get_colours(n):
    return plt.cm.viridis(np.linspace(0.1, 0.9, n))

labels = {
    "umag": "$||\\mathbf{u}||$ (m s$^{-1}$)",
    "u": "$u$ (m s$^{-1}$)",
    "v": "$v$ (m s$^{-1}$)",
    "y": "$y$ (m)",
    "x": "$x$ (m)",
    "P": "$P$ (MW)",
    "t": "$t$ (s)",
    "yaw": "$\psi$ (rad)",
    "yaw_d": r"$\psi$ ($\degree$)"
}

if __name__ == '__main__':
    x = np.linspace(0,1,100)
    y = x**2
    fig,ax = plt.subplots(1,2,sharex='all',sharey='all')

    ax[0].plot(x,y)
    ax[0].set_title('some title name')
    ax[0].set_ylabel(labels['y'])
    ax[0].set_xlabel(labels['x'])
    ax[1].plot(x,-y)
    ax[1].set_title('title name')
    ax[1].set_xlabel(labels['x'])
    fig.savefig("figure.pdf", type='pdf')
    fig.savefig("figure.png", type='png',dpi=300)
    plt.show()
