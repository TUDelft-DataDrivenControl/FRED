from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from tools.plot import *
import scipy.interpolate


def calculate_coefficients(turbine_file, tip_speed_ratio, pitch_angle, wind_speed):
    f = ParsedParameterFile(turbine_file)

    blade_radius = f['TipRad']
    num_blades = f['NumBl']

    blade_data = np.array([x.vals for x in f["BladeData"]])
    blade_data_columns = ["radius", "chord", "twist", "thickness", "userdef", "airfoil"]
    blade_df = pd.DataFrame(data=blade_data, columns=blade_data_columns)

    # calculate turbine blade segment lengths
    start_points = np.zeros_like(blade_df['radius'])
    end_points = np.zeros_like(start_points)
    start_points[1:] = blade_df['radius'][1:] - np.diff(blade_df['radius'] / 2)
    end_points[:-1] = start_points[1:]
    end_points[-1] = blade_radius
    blade_df['segment_length'] = end_points - start_points

    airfoil_list = [x.strip('"') for x in f["Airfoils"]]
    airfoil_df_list = []
    for airfoil in airfoil_list:
        airfoil_file = "./airfoilProperties/" + airfoil
        af = ParsedParameterFile(airfoil_file)
        airfoil_data = [x.vals for x in af['airfoilData']]
        airfoil_data_columns = ["alpha", "cl", "cd"]
        airfoil_df_list.append(pd.DataFrame(data=airfoil_data, columns=airfoil_data_columns))

    angular_velocity = wind_speed * tip_speed_ratio / blade_radius
    # axial_induction = 0.  # -
    blade_df['axial_induction'] = 0.
    axial_induction = blade_df['axial_induction']
    blade_df['tangential_induction'] = 0.

    # angular_velocity = 0.4  # rad.s^-1
    air_density = 1.225  # kg.m^-3
    # pitch_angle = 0.  # deg
    blade_df['old_axial_induction'] = 1.
    blade_df['old_tangential_induction'] = 1.

    iteration = 0
    while (np.max(np.abs(blade_df['axial_induction'] - blade_df['old_axial_induction'])) >= 1e-4) \
            and (np.max(np.abs(blade_df['tangential_induction'] - blade_df['old_tangential_induction'])) >= 1e-4)\
            and iteration < 25.:
        blade_df['old_axial_induction'] = blade_df['axial_induction'].values
        blade_df['old_tangential_induction'] = blade_df['tangential_induction'].values

        # v_in  = v_inf * (1 - a)
        blade_df['wind_velocity'] = wind_speed * (1 - blade_df['axial_induction'])

        # v_rot = r * w * (1-a')
        blade_df['rotational_velocity'] = blade_df['radius'] * angular_velocity * (1 + blade_df['tangential_induction'])

        # v_rel = sqrt( v_rot^2 + v_inf^2 )
        blade_df['relative_velocity'] = np.sqrt(blade_df['rotational_velocity'] ** 2 + blade_df['wind_velocity'] ** 2)

        # phi = atan( v_inf / v_rot )
        blade_df['inflow_angle'] = np.rad2deg(np.arctan(blade_df['wind_velocity'] / blade_df['rotational_velocity']))

        # alpha = phi - theta
        blade_df['angle_of_attack'] = blade_df['inflow_angle'] - blade_df['twist'] - pitch_angle

        # interpolate angle of attack for lift and drag coefficient
        blade_df['cl'] = [np.interp(a, airfoil_df_list[n]['alpha'], airfoil_df_list[n]['cl'])
                          for a, n in zip(blade_df['angle_of_attack'], blade_df['airfoil'].astype(int))]
        blade_df['cd'] = [np.interp(a, airfoil_df_list[n]['alpha'], airfoil_df_list[n]['cd'])
                          for a, n in zip(blade_df['angle_of_attack'], blade_df['airfoil'].astype(int))]

        # calculate axial and angular force coefficient.
        phi = np.deg2rad(blade_df['inflow_angle'])
        blade_df['cx'] = (blade_df['cl'] * np.cos(phi) + blade_df['cd'] * np.sin(phi))
        blade_df['cy'] = (blade_df['cl'] * np.sin(phi) - blade_df['cd'] * np.cos(phi))

        # update induction factors
        blade_df['chord_solidity'] = num_blades * (1 / (2 * np.pi)) * (blade_df['chord'] / blade_df['radius'])

        blade_df['axial_induction'] = 1 / ((4 * np.sin(phi) ** 2) / (blade_df['chord_solidity'] * blade_df['cx']) + 1)
        blade_df['tangential_induction'] = 1 / (
                    (4 * np.sin(phi) * np.cos(phi)) / (blade_df['chord_solidity'] * blade_df['cy']) - 1)

        iteration += 1
        print("iteration: {}".format(iteration))


    # calculate element-wise lift
    blade_df['lift'] = blade_df['cl'] * 0.5 * air_density * blade_df['relative_velocity'] ** 2 * \
                       blade_df['chord'] * blade_df['segment_length']
    blade_df['drag'] = blade_df['cd'] * 0.5 * air_density * blade_df['relative_velocity'] ** 2 * \
                       blade_df['chord'] * blade_df['segment_length']

    # calculate normal and driving force
    phi = np.deg2rad(blade_df['inflow_angle'])
    blade_df['axial_force'] = blade_df['lift'] * np.cos(phi) + \
                              blade_df['drag'] * np.sin(phi)
    blade_df['angular_force'] = blade_df['lift'] * np.sin(phi) - \
                                blade_df['drag'] * np.cos(phi)

    # calculate power per element
    blade_df['power'] = blade_df['angular_force'] * blade_df['rotational_velocity']

    total_force = num_blades * blade_df['axial_force'].sum()
    total_power = num_blades * blade_df['power'].sum()

    print("Total thrust: {:.2e} N".format(total_force))
    print("Total power : {:.2e} W".format(total_power))

    thrust_coefficient = total_force / (0.5 * air_density * np.pi * blade_radius ** 2 * wind_speed ** 2)
    power_coefficient = total_power / (0.5 * air_density * np.pi * blade_radius ** 2 * wind_speed ** 3)

    print("Thrust coefficient: {:.2f}".format(thrust_coefficient))
    print("Power coefficient : {:.2f}".format(power_coefficient))

    return thrust_coefficient, power_coefficient


def calculate_grid(pitch_array, tsr_array):
    pitch_grid, tsr_grid = np.meshgrid(pitch_array, tsr_array)

    ct_array = np.zeros_like(pitch_grid)
    cp_array = np.zeros_like(tsr_grid)

    wind_speed = 9.
    for idx0 in range(len(pitch_array)):
        for idx1 in range(len(tsr_array)):
            ct, cp = calculate_coefficients(turbine_file = "./turbineProperties/DTU10MWRef",
                                            tip_speed_ratio=tsr_array[idx1],
                                            pitch_angle=pitch_array[idx0],
                                            wind_speed=wind_speed)
            ct_array[idx1, idx0] = ct
            cp_array[idx1, idx0] = cp

    # cp_array[cp_array<0] = 0
    # ct_array[ct_array<0] = 0
    return pitch_grid, tsr_grid, ct_array, cp_array


def save_results(pitch_grid, tsr_grid, ct_array, cp_array):
    with open("DTU10MW_ct_cp", "wb") as f:
        np.save(f,[pitch_grid, tsr_grid, ct_array, cp_array] )

def load_results():
    with open("DTU10MW_ct_cp", "rb") as f:
        pitch_grid, tsr_grid, ct_array, cp_array = np.load(f)

    return pitch_grid, tsr_grid, ct_array, cp_array

def plot_coefficient_surface(pitch_grid, tsr_grid, ct_array, cp_array):
    fig = plt.figure(0)
    cp_array[cp_array<0] = 0.
    ct_array[ct_array<0] = 0.
    # cp_array = np.ma.masked_less_equal(cp_array, 0.)
    # fig = plt.figure(1)
    # cp_array = np.ma.masked_where(cp_array==0,cp_array)

    ax = []
    ax.append(fig.add_subplot(121, projection='3d'))
    current_ax = ax[0]
    current_ax.plot_surface(pitch_grid, tsr_grid, ct_array, cmap='viridis')
    current_ax.set_xlabel(labels['pitch'])
    current_ax.set_ylabel(labels['tsr'])
    current_ax.set_zlabel(labels['ct'])
    current_ax.set_zlim(0, current_ax.get_zlim()[1])

    ax.append(fig.add_subplot(122, projection='3d'))
    current_ax = ax[1]
    current_ax.plot_surface(pitch_grid, tsr_grid, cp_array, cmap='viridis')
    current_ax.set_xlabel(labels['pitch'])
    current_ax.set_ylabel(labels['tsr'])
    current_ax.set_zlabel(labels['cp'])
    current_ax.set_zlim(0, current_ax.get_zlim()[1])


def read_rosco_curves():
    filename = "Cp_Ct_Cq.DTU10MW.txt"
    with open(filename,"r") as f:
        datafile = f.readlines()
    for idx in range(len(datafile)):
        if "Pitch angle" in datafile[idx]:
            pitch_array = np.loadtxt(filename, skiprows=idx+1, max_rows=1)
        if "TSR vector" in datafile[idx]:
            tsr_array = np.loadtxt(filename, skiprows=idx+1, max_rows=1)
        if "Wind speed" in datafile[idx]:
            wind_speed = np.loadtxt(filename, skiprows=idx+1, max_rows=1)

        if "Power coefficient" in datafile[idx]:
            cp_array = np.loadtxt(filename, skiprows=idx+2, max_rows=len(tsr_array))
        if "Thrust coefficient" in datafile[idx]:
            ct_array = np.loadtxt(filename, skiprows=idx+2, max_rows=len(tsr_array))
        if "Torque coefficent" in datafile[idx]:
            cq_array = np.loadtxt(filename, skiprows=idx+2, max_rows=len(tsr_array))

    pitch_grid, tsr_grid = np.meshgrid(pitch_array, tsr_array)
    return pitch_grid, tsr_grid, ct_array, cp_array


def lookup_field(pitch_grid, tsr_grid, ct_array, cp_array):
    # construct function space
    sw_corner = Point(np.min(pitch_grid), np.min(tsr_grid))
    ne_corner = Point(np.max(pitch_grid), np.max(tsr_grid))
    (n_tsr, n_pitch) = pitch_grid.shape
    # set function in function space
    m = RectangleMesh(sw_corner, ne_corner, n_pitch+1, n_tsr+1)
    fe = FiniteElement("Lagrange", m.ufl_cell(), 1)
    fs = FunctionSpace(m, fe)

    # assign values to function
    dof_coords = fs.tabulate_dof_coordinates()

    ct = Function(fs)
    ct_interp = scipy.interpolate.interp2d(pitch_grid[0,:], tsr_grid[:,0], ct_array, kind='linear')
    ct_values = ct.vector().get_local()

    cp = Function(fs)
    cp_interp = scipy.interpolate.interp2d(pitch_grid[0,:], tsr_grid[:,0], cp_array, kind='linear')
    cp_values = cp.vector().get_local()
    for idx in range(len(dof_coords)):
        pitch, tsr = dof_coords[idx]
        ct_values[idx] = ct_interp(pitch, tsr)
        cp_values[idx] = cp_interp(pitch, tsr)
    ct.vector().set_local(ct_values)
    cp.vector().set_local(cp_values)

    # fv = VectorElement("Lagrange", m.ufl_cell(), 1)
    # fsv = FunctionSpace(m, fv)
    #
    # gct = project(grad(ct),fsv)


    # ct_file = File("ct.pvd")
    # cp_file = File("cp.pvd")
    # ct_file.write(ct)
    # cp_file.write(cp)

    return ct, cp


def evaluate_ct_cp(ct, cp, pitch, tsr):
    # pre-allocate numpy arrays for result storage
    ct_val = np.array([0.]) # make sure this is a float array
    cp_val = np.array([0.])
    x = np.array([pitch, tsr])

    # evaluate ct and cp for given pitch and tsr
    ct.eval(ct_val, x)
    cp.eval(cp_val, x)

    # or:
    # ct_val = ct(pitch, tsr)
    # cp_val = cp(pitch, tsr)
    return ct_val, cp_val


def point_eval_taylor_test(ct, cp):

    pitch = Constant(5.)
    tsr = Constant(8.)


    # ctval = ct(pitch, tsr)

    backend_ct = ct

    def get_ct(pitch, tsr):
        return backend_ct(pitch, tsr)

    #  http://www.dolfin-adjoint.org/en/latest/documentation/custom_functions.html
    from pyadjoint import Block

    class CtBlock(Block):
        def __init__(self, func, **kwargs):
            super(CtBlock, self).__init__()
            # self.func = func
            self.gradient = [project(backend_ct.dx(0), backend_ct.function_space()),
                             project(backend_ct.dx(1), backend_ct.function_space())]
            self.kwargs = kwargs
            self.add_dependency(func)

        def __str__(self):
            return "CtBlock"

        def recompute_component(self, inputs, block_variable, idx, prepared):
            return backend_ct(inputs[0], inputs[1])

        def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
            return self.gradient[idx](adj_inputs[0],adj_inputs[1])

    from pyadjoint.overloaded_function import overload_function
    get_ct = overload_function(get_ct, CtBlock)
    # ct = overload_function(ct.__call__, CtBlock)

    J = assemble(ct(pitch, tsr))
    m = Control(pitch)

    Jhat = ReducedFunctional(J, m)
    dJdm = Jhat.derivative()
    taylor_test(Jhat, m, m)


def coefficient_gradient_trial(ct, cp):

    pitch_val = Constant(10.)
    tsr_val = Constant(6.)

    def get_coefficient(func, pitch, tsr):
        return func(pitch, tsr)

    backend_get_coefficient = get_coefficient

    from pyadjoint import Block

    class CoefficientBlock(Block):
        def __init__(self,func, pitch, tsr, **kwargs):
            super(CoefficientBlock, self).__init__()
            self.kwargs = kwargs
            self.func = func
            self.add_dependency(pitch)
            self.add_dependency(tsr)
            degree = func.function_space().ufl_element().degree()
            family = func.function_space().ufl_element().family()
            mesh = func.function_space().mesh()
            if np.isin(family, ["CG", "Lagrange"]):
                self.V = FunctionSpace(mesh, "DG", degree - 1)
            else:
                raise NotImplementedError(
                    "Not implemented for other elements than Lagrange")

        def __str__(self):
            return "CoefficientBlock"

        def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
            # output = get_derivative(inputs[0], inputs[1], idx) * adj_inputs[0]
            grad_idx = project(self.func.dx(idx), self.V)
            output = grad_idx(inputs[0], inputs[1]) * adj_inputs[0]
            return output

        def recompute_component(self, inputs, block_variable, idx, prepared):
            return backend_get_coefficient(self.func, inputs[0], inputs[1])

    from pyadjoint.overloaded_function import overload_function

    get_coefficient = overload_function(get_coefficient, CoefficientBlock)

    controls = [pitch_val, tsr_val]

    ctv = get_coefficient(ct, pitch_val, tsr_val)
    cpv = get_coefficient(cp, pitch_val, tsr_val)
    J = (ctv*cpv) **2

    m = [Control(ctrl) for ctrl in controls]

    Jhat = ReducedFunctional(J, m)
    dJdm = Jhat.derivative()

    print("\nBeginning taylor test:\n")
    taylor_test(Jhat, controls, [Constant(0.01*np.random.rand()) for c in controls])

    print("end")



if __name__ == '__main__':
    # ct, cp = calculate_coefficients(turbine_file = "./turbineProperties/DTU10MWRef",
    #                                 tip_speed_ratio=10.,
    #                                 pitch=3.,
    #                                 wind_speed=9.)
    # pitch_grid, tsr_grid, ct_array, cp_array = calculate_grid(pitch_array=np.linspace(0, 45,16),
    #                                                           tsr_array=np.linspace(1,12,12))
    # save_results( pitch_grid, tsr_grid, ct_array, cp_array)
    # pitch_grid, tsr_grid, ct_array, cp_array = load_results()
    # plot_coefficient_surface(pitch_grid, tsr_grid, ct_array, cp_array)
    # plt.show()

    pitch_grid, tsr_grid, ct_array, cp_array = read_rosco_curves()
    # save_results(pitch_grid, tsr_grid, ct_array, cp_array)
    # plot_coefficient_surface(pitch_grid, tsr_grid, ct_array, cp_array)
    # plt.show()

    ct, cp = lookup_field(pitch_grid, tsr_grid, ct_array, cp_array)
    coefficient_gradient_trial(ct,cp)
    # point_eval_taylor_test(ct,cp)
    # ctv, cpv = evaluate_ct_cp(ct, cp, pitch=0., tsr=10.)
    # print("Ct: {:.2f}, Cp: {:.2f}".format(float(ctv), float(cpv)))
