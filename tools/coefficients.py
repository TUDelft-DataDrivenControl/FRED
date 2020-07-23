import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

def parse_turbine_files():
    turbine_file = "./turbineProperties/DTU10MWRef"
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

    return blade_df, airfoil_df_list, num_blades, blade_radius

def calculate_thrust_coefficient(blade_df, airfoil_df_list, num_blades=3, blade_radius=60.):
    wind_speed = 9.  # m.s^-1
    # axial_induction = 0.  # -
    blade_df['axial_induction'] = 0.
    axial_induction = blade_df['axial_induction']
    blade_df['tangential_induction'] = 0.
    tip_speed_ratio = 10.
    angular_velocity = wind_speed * tip_speed_ratio / blade_radius
    # angular_velocity = 0.4  # rad.s^-1
    air_density = 1.225 # kg.m^-3
    pitch_angle = 3. # deg

    for idx in range(200):
        # v_in  = v_inf * (1 - a)
        blade_df['wind_velocity'] = wind_speed * (1-axial_induction)

        # v_rot = r * w * (1-a')
        blade_df['rotational_velocity'] = blade_df['radius']*angular_velocity * (1 + blade_df['tangential_induction'])

        # v_rel = sqrt( v_rot^2 + v_inf^2 )
        blade_df['relative_velocity'] = np.sqrt(blade_df['rotational_velocity']**2 + blade_df['wind_velocity']**2)

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
        blade_df['chord_solidity'] = num_blades * (1 / (2*np.pi)) * (blade_df['chord'] / blade_df['radius'])

        axial_induction = 1 / ((4*np.sin(phi)**2) / (blade_df['chord_solidity'] * blade_df['cx']) + 1)
        blade_df['tangential_induction'] = 1 / ((4*np.sin(phi)*np.cos(phi)) / (blade_df['chord_solidity'] * blade_df['cy']) -1)
        # b_ax = (blade_df['chord_solidity'] / (4*np.sin(phi)**2)) * \
        #     (blade_df['cx'] - (blade_df['chord_solidity'] / (4*np.sin(phi)**2)) * blade_df['cy']**2)
        # axial_induction = (b_ax / (1 + b_ax)).mean()
        #
        # b_tg = (blade_df['chord_solidity'] / (4*np.sin(phi)*np.cos(phi))) * blade_df['cy']
        # blade_df['tangential_induction'] = axial_induction / b_tg - 1


        #
        # print("Axial induction: {:.2f}".format(axial_induction))

    # calculate element-wise lift
    blade_df['lift'] = blade_df['cl'] * 0.5 * air_density * blade_df['relative_velocity']**2 *\
                       blade_df['chord'] * blade_df['segment_length']
    blade_df['drag'] = blade_df['cd'] * 0.5 * air_density * blade_df['relative_velocity']**2 *\
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

    thrust_coefficient = total_force / (0.5 * air_density * np.pi * blade_radius**2 * wind_speed**2)
    power_coefficient =  total_power / (0.5 * air_density * np.pi * blade_radius**2 * wind_speed**3)

    print("Thrust coefficient: {:.2f}".format(thrust_coefficient))
    print("Power coefficient : {:.2f}".format(power_coefficient))


if __name__ == '__main__':
    blade_df, airfoil_df_list, num_blades, blade_radius = parse_turbine_files()
    calculate_thrust_coefficient(blade_df, airfoil_df_list, num_blades ,blade_radius)
