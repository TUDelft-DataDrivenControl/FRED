from tools.data import *
from tools.plot import *

def main_contours():
    fig, ax = plt.subplots(1,3)

    vtk_file_u = "./results/two_U000030.vtu"
    vtk_file_p = "./results/two_p000030.vtu"
    vtk_file_f = "./results/two_f000030.vtu"

    x, u , tri  = read_vtk_unstructured(vtk_file_u)
    x, p, tri = read_vtk_unstructured(vtk_file_p)
    x, f, tri = read_vtk_unstructured(vtk_file_f)

    plot_contours(x, p, ax=ax[0], type='pressure')
    plot_contours(x, u, ax=ax[1], type='velocity')
    plot_contours(x, f, ax=ax[2], type='force')

def main_slice():
    fig,ax = plt.subplots(1,1, figsize=(5,3))
    fig.subplots_adjust(bottom=0.15,top=0.9)
    vtk_file_u = "./results/two_turbine_test/U000029.vtu"
    x, u, tri = read_vtk_unstructured(vtk_file_u)
    xs, us = get_slice(x,u,xloc=499)

    u_inf = 9.
    wi = 0.33 # ???
    radius = 89.
    xt = xs - 400.
    u_theory = u_inf * (1. - (2/np.pi)*wi*((np.pi/2) + np.arctan(xt/radius)))
    c = get_colours(2)
    ax.plot(xs,us[:,0],c=c[0],label="actual")
    ax.plot(xs,u_theory,c=c[1],label="theoretical")
    ax.set_xlabel(labels['t'])
    ax.set_ylabel(labels['u'])
    ax.legend()

def two_turbine_step():
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    fig.subplots_adjust(bottom=0.2,top=0.9)
    filename = "./results/two_log.csv"
    t,p = read_power_wasp(filename, 2)
    # t,y = read_yaw_wasp(filename, 2)
    c = get_colours(2)

    ax.plot(t, p[:, 0], c=c[0], label="$T_0$")
    ax.plot(t, p[:, 1], c=c[1], label="$T_1$")
    ax.set_ylim(0,6)
    ax.set_ylabel(labels['P'])
    ax.set_xlabel(labels['t'])
    ax.legend()

def single_turbine_grad_control():
    fig,ax = plt.subplots(2,1,figsize=(5,3),sharex='col')
    fig.subplots_adjust(bottom=0.2,top=0.9,left=0.15, right=0.95)
    filename = "./results/two_log.csv"
    t,p = read_power_wasp(filename, 1)
    t,y = read_yaw_wasp(filename, 1)
    c = get_colours(2)
    current_ax = ax[0]
    current_ax.plot(t, p, c=c[0], label="$T_0$")
    # ax.plot(t, p[:, 1], c=c[1], label="$T_1$")
    current_ax.set_ylim(0,8)
    current_ax.set_ylabel(labels['P'])

    current_ax = ax[1]
    current_ax.plot(t, y, c=c[0], label="$T_0$")
    current_ax.set_ylabel(labels['yaw'])
    current_ax.set_xlabel(labels['t'])
    current_ax.set_ylim(-0.5,0.5)
    current_ax.set_xlim(0,80)
    fig.align_ylabels()
    # ax.set_xlabel(labels['t'])
    # ax.legend()

def two_turbine_grad_control():
    fig,ax = plt.subplots(2,1,figsize=(5,3),sharex='col')
    fig.subplots_adjust(bottom=0.2,top=0.9,left=0.15, right=0.95)
    filename = "./results/two_log.csv"
    t,p = read_power_wasp(filename, 2)
    t,y = read_yaw_wasp(filename, 2)
    c = get_colours(2)
    current_ax = ax[0]
    current_ax.plot(t, p[:, 0], c=c[0], label="$T_0$")
    current_ax.plot(t, p[:, 1], c=c[1], label="$T_1$")
    current_ax.set_ylim(0,8)
    current_ax.set_ylabel(labels['P'])
    current_ax.legend()

    current_ax = ax[1]
    current_ax.plot(t, y[:, 0], c=c[0], label="$T_0$")
    current_ax.plot(t, y[:, 1], c=c[1], label="$T_1$")
    current_ax.set_ylabel(labels['yaw'])
    current_ax.set_xlabel(labels['t'])
    current_ax.set_ylim(-0.5,0.5)
    # current_ax.set_xlim(0,80)
    fig.align_ylabels()
    # ax.set_xlabel(labels['t'])
    # ax.legend()

def show_jacobian(djdm):
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
    max = np.abs(djdm).max()
    colours = grid[0].imshow(djdm,cmap='seismic',vmin=-max, vmax=max)
    cb = fig.colorbar(colours, grid[0].cax)
    cb.solids.set_edgecolor("face")



if __name__ == '__main__':
    # main_contours()
    main_slice()
    # two_turbine_step()
    # show_jacobian()
    # single_turbine_grad_control()
    # two_turbine_grad_control()
    plt.show()