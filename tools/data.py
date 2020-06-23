import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import floris.tools as wfct
import scipy.signal as sig
import pandas as pd

import controlmodel.conf as conf

def read_vtk_unstructured(filename_one):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename_one)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    if points is None:
        raise FileNotFoundError("The requested file does not exist: {}\n"
                                .format(filename_one))
    x = vtk_to_numpy(points.GetData())
    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4
    tri = np.take(triangles, [n for n in range(triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    n_arrays = reader.GetNumberOfPointArrays()
    for i in range(n_arrays):
        print(i, ":", reader.GetPointArrayName(i))

    u = vtk_to_numpy(data.GetPointData().GetArray(0))
    return x, u, tri


def read_vtk_sowfa(filename):
    """
    Reads SOWFA results .vtk file and returns coordinates of cell centres and velocity field as numpy arrays.
    :param filename:
    :return:
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)

    reader.Update()
    data = reader.GetOutput()
    u = vtk_to_numpy(data.GetCellData().GetArray(0))

    x = np.zeros_like(u)
    for idx in range(0, data.GetNumberOfCells()):
        # idx = 9231 # cell number
        p = vtk_to_numpy(data.GetCell(idx).GetPoints().GetData())
        p_center = (np.max(p, 0) + np.min(p, 0)) / 2
        x[idx, :] = p_center  # save center of point to coordinate list
    return x, u


def run_floris(pos=None, yaw=None, dir=None, domain=None):
    fi = wfct.floris_interface.FlorisInterface("config/two_turbine_steady.json")
    if pos is not None:
        pos = (pos[:, 0], pos[:, 1])

    fi.reinitialize_flow_field(layout_array=pos, wind_direction=dir, bounds_to_set=[0., 3000., 0., 3000., 0., 500.])

    fi.calculate_wake(yaw_angles=yaw)

    hor_plane = fi.get_hor_plane(x_bounds=[0, domain[0]], y_bounds=[0, domain[1]])

    x = hor_plane.df[['x1', 'x2']].to_numpy()
    u = hor_plane.df[['u', 'v']].to_numpy()
    p = np.array(fi.get_turbine_power())/1e6
    return x, u, p

def read_power_sowfa(filename):
    data = np.loadtxt(filename)
    num_turbines = int(np.max(data[:, 0]) + 1)
    time_data = data[0::num_turbines, 1] - data[0, 1]
    value_data = np.zeros((len(time_data), num_turbines))
    for idx in range(num_turbines):
        print(idx)
        value_data[:,idx] = data[idx::num_turbines,3]
    value_data = value_data / 1.225 / 1e6

    return time_data, value_data, num_turbines

def read_power_wasp(filename,num_turbines):
    df = pd.read_csv(filename)
    time_data = df['time'].to_numpy()
    value_data = df[['power_{:03d}'.format(x) for x in range(num_turbines)]].to_numpy() / 1e6
    return time_data, value_data

def read_yaw_wasp(filename,num_turbines):
    df = pd.read_csv(filename)
    time_data = df['time'].to_numpy()
    value_data = df[['yaw_{:03d}'.format(x) for x in range(num_turbines)]].to_numpy()
    return time_data, value_data

def filter_data(value_data):
    value_data_hat = np.zeros_like(value_data)
    for idx in range(value_data.shape[1]):
        value_data_hat[:, idx] = sig.savgol_filter(value_data[:, idx], 21, 3)
    return value_data_hat

def calculate_turbulence_intensity(u):
    """
    Calculate turbulence intensity from a turbulent precursor flow field u.
    """
    unorm = np.sqrt(np.sum(u**2,1))
    ti = np.std(unorm)/np.mean(unorm)
    return ti


def load_jacobian(turbine_idx=0,power_idx=None):
    if power_idx is None:
        power_idx = -1
    with open("./results/"+conf.par.simulation.name+"/djdm_P{:d}_T{:d}.npy".format(power_idx,turbine_idx), "rb") as f:
        dj_dm = np.load(f)
    return dj_dm


def load_lti_jacobian(turbine_idx=0):
    with open("./results/"+conf.par.simulation.name+"/djdm_LTI_P_T{:d}.npy".format(turbine_idx), "rb") as f:
        dj_dm = np.load(f)
    return dj_dm
