import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from tools.plot import *

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

    pitch_grid,tsr_grid = np.meshgrid(pitch_array, tsr_array)
    return pitch_grid, tsr_grid, ct_array, cp_array

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
    save_results(pitch_grid, tsr_grid, ct_array, cp_array)
    plot_coefficient_surface(pitch_grid, tsr_grid, ct_array, cp_array)
    plt.show()