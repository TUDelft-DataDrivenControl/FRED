import numpy as np
import control as ct
import matplotlib.pyplot as plt
import os
import scipy.interpolate
import logging
logger = logging.getLogger("tools.tsrtracker")
from tools.plot import labels

class Estimator:
    def __init__(self, num_turbines, sample_time):
        self._sampling_time = sample_time

        self._gamma = 10.
        self._rotor_speed_filtered = np.zeros((1, num_turbines))
        self._VwI = np.zeros((1, num_turbines))
        self._VwI_prev = np.zeros((1, num_turbines))

        self._wind_speed = np.zeros((1, num_turbines))
        self._wind_speed_prev = 12 * np.ones((1, num_turbines))  # why 12 in matlab?

        self._tsr = np.zeros((1, num_turbines))

        # initialise filter to zero
        self._filter_state = np.zeros((3, num_turbines))
        self._filter_state_prev = np.zeros((3, num_turbines))

        self._combined_filter = self._construct_rotor_speed_filter()

        self._turbine = TurbineParameters()

        self._measured_generator_torque = None
        self._measured_blade_pitch = None

    def _construct_rotor_speed_filter(self):
        cutoff_frequency = 5  # cut-off frequency
        low_pass_transfer_function = ct.tf([0, cutoff_frequency],
                                           [1, cutoff_frequency])
        low_pass_state_space = ct.tf2ss(low_pass_transfer_function)
        low_pass_state_space_discrete = ct.c2d(low_pass_state_space, self._sampling_time, 'tustin')

        beta_num = 0.0195  # notch parameter
        beta_den = 0.125  # notch parameter
        notch_frequency = 2 * np.pi * 0.62
        notch_transfer_function = ct.tf([1, 2 * beta_num * notch_frequency, notch_frequency ** 2],
                                        [1, 2 * beta_den * notch_frequency, notch_frequency ** 2])
        notch_state_space = ct.tf2ss(notch_transfer_function)
        notch_state_space_discrete = ct.c2d(notch_state_space, self._sampling_time, 'tustin')

        combined_filter_transfer_function = low_pass_state_space_discrete * notch_state_space_discrete
        return combined_filter_transfer_function

    def run_estimator(self, measured_rotor_speed, measured_generator_torque, measured_blade_pitch):
        self._measured_generator_torque = measured_generator_torque
        self._measured_blade_pitch = measured_blade_pitch
        self._filter_rotor_speed(measured_rotor_speed)
        self._estimate_wind_speed(measured_generator_torque, measured_blade_pitch)
        self._update_filter()

    def _filter_rotor_speed(self, measured_rotor_speed):
        self._filter_state = self._combined_filter.A * self._filter_state_prev + \
                             self._combined_filter.B * measured_rotor_speed
        self._rotor_speed_filtered = self._combined_filter.C * self._filter_state + \
                                     self._combined_filter.D * measured_rotor_speed

    def _estimate_wind_speed(self, measured_generator_torque, measured_blade_pitch):
        wt = self._turbine
        w = self._rotor_speed_filtered * np.pi / 30  # from rpm to rad.s^-1
        self._wind_speed = self._VwI_prev + self._gamma * w
        self._tsr = w * wt.rotor_radius / self._wind_speed

        self._VwI = self._VwI_prev + \
                    (self._sampling_time * self._gamma / wt.inertia_total) * \
                    (measured_generator_torque * wt.gearbox_ratio -
                     (wt.density * wt.rotor_area) / 2 * np.multiply(np.power(self._wind_speed, 3),
                                                                    wt.power_coefficient_lut(tsr=self._tsr,
                                                                                             pitch=measured_blade_pitch)) / w)

    def _update_filter(self):
        self._wind_speed_prev = self._wind_speed
        self._VwI_prev = self._VwI
        self._filter_state_prev = self._filter_state

    # def estimate_wind_speed(self, measurements):
    #
    # def update_estimator_values
    def get_filter(self):
        return self._combined_filter

    def plot_filter(self):
        w = np.linspace(0., 20., 100)
        f = self._combined_filter
        mag, phase, omega = f.freqresp(w)
        plt.plot(omega, mag.squeeze())
        plt.show()


class TurbineParameters:
    def __init__(self):
        # self.rotor_radius = 56.0
        # self.rotor_area = np.pi * self.rotor_radius ** 2
        #
        # self.gearbox_ratio = 113.25
        # self.density = 1.225
        #
        # # todo: look up correct values
        # self.inertia_generator = 132
        # self.inertia_blade = 4.45e6
        # self.inertia_total = 3 * self.inertia_blade + self.inertia_generator * self.gearbox_ratio ** 2

        self.rotor_radius = 89.2
        self.rotor_area = np.pi * self.rotor_radius ** 2

        self.gearbox_ratio = 50.0
        self.density = 1.225

        # todo: look up correct values
        self.inertia_generator = 1500.5
        self.inertia_hub = 325.6709e3
        self.inertia_blade = 45.64e6
        self.inertia_total = 3 * self.inertia_blade + self.inertia_hub + self.inertia_generator * self.gearbox_ratio ** 2
        pitch_grid, tsr_grid, cp_array = self._read_rosco_curves()
        self._cp_interpolator = scipy.interpolate.interp2d(pitch_grid[0, :], tsr_grid[:, 0], cp_array, kind='linear')

    def _read_rosco_curves(self):
        filename = os.path.join(os.path.dirname(__file__), "../config/Cp_Ct_Cq.DTU10MW.txt")
        with open(filename, "r") as f:
            datafile = f.readlines()
        for idx in range(len(datafile)):
            if "Pitch angle" in datafile[idx]:
                pitch_array = np.loadtxt(filename, skiprows=idx + 1, max_rows=1)
            if "TSR vector" in datafile[idx]:
                tsr_array = np.loadtxt(filename, skiprows=idx + 1, max_rows=1)

            if "Power coefficient" in datafile[idx]:
                cp_array = np.loadtxt(filename, skiprows=idx + 2, max_rows=len(tsr_array))

        pitch_grid, tsr_grid = np.meshgrid(pitch_array, tsr_array)
        return pitch_grid, tsr_grid, cp_array

    def power_coefficient_lut(self, tsr, pitch):
        # todo: implement LUT for cp(tsr, pitch)
        # cp1 = 21
        # cp2 = 125.21
        # cp3 = 9.8
        # cp4 = 0.0068
        # cp = np.multiply(np.exp(-cp1 / tsr), cp2 / tsr - cp3) + cp4 * tsr
        # cp = 2.
        if len(tsr.shape)==2:
            cp = np.zeros_like(tsr)
            for idx in range(max(tsr.shape)):
                cp[0,idx] = float(self._cp_interpolator(pitch[0,idx],tsr[0,idx]))
        else:
            cp = self._cp_interpolator(pitch,tsr)
        return cp


class TurbineModel:
    def __init__(self):
        self._parameters = TurbineParameters()
        self._sampling_time = 0.2

        self._rotor_speed = 1.
        self._rotor_speed_prev = 1.

        self._torque = 1000.
        self._pitch = 0.

        self._wind_speed = 10.

    def _aerodynamic_power(self):
        return (self._parameters.density * self._parameters.rotor_area) / 2 * np.power(self._wind_speed, 3) * \
               self._parameters.power_coefficient_lut(self._tsr(), self._pitch)

    def _tsr(self):
        return self._rotor_speed * self._parameters.rotor_radius / self._wind_speed

    def _update_rotor_speed(self):
        rotor_acceleration = -(1 / self._parameters.inertia_total) * \
                             (self._torque * self._parameters.gearbox_ratio -
                              self._aerodynamic_power() / self._rotor_speed_prev)
        self._rotor_speed = self._rotor_speed_prev + self._sampling_time * rotor_acceleration
        self._rotor_speed_prev = self._rotor_speed

    def run_time_step(self, wind_speed, torque, pitch):
        self._wind_speed = wind_speed
        self._torque = torque
        self._pitch = pitch
        self._update_rotor_speed()

    def get_rotor_speed(self):
        return self._rotor_speed * 30 / np.pi

    def get_torque(self):
        return self._torque

    # def set_torque(self, new_torque):
    #     self._torque = new_torque

    def get_pitch(self):
        return self._pitch


class TorqueController:
    def __init__(self, num_turbines, sample_time):
        self._torque_proportional_gain = -27338.
        self._torque_integrator_gain = -6134.

        self._torque_reference_base = None
        self._torque_reference_integrator = None
        self._torque_reference_proportional = None

        self._estimator = Estimator(num_turbines, sample_time)
        self._turbine = TurbineParameters()
        # todo: cut-in / rated rotorspeed

        # self._speed_torque_table = np.array([[0.0, 0.0],
        #                                [670.00, 0.0],
        #                                [700.00, 2542.00],
        #                                [874.20, 3972.00],
        #                                [1049.00, 5715.00],
        #                                [1224.00, 7758.00],
        #                                [1399.00, 10080.00],
        #                                [1490.00, 24380.00],
        #                                [10e3, 24380.00]])
        # //      gen speed (RPM) gen torque (N-m)
        self._speed_torque_table = np.array([[200.00, 0.0],
                                             [300.00, 90000.0],
                                             [405.00, 164025.0],
                                             [480.00, 164025.0]])
    def run_estimator(self, measured_rotor_speed, measured_generator_torque, measured_blade_pitch):
        logger.warning("Double-check rotorspeed units")
        measured_rotor_speed = measured_rotor_speed * 30 / np.pi
        self._estimator.run_estimator(measured_rotor_speed, measured_generator_torque, measured_blade_pitch)

    def generate_torque_reference(self, tsr_desired):
        wind_speed = self._estimator._wind_speed

        rotor_speed_reference = np.multiply(tsr_desired, wind_speed) * (30 / np.pi) / self._turbine.rotor_radius

        rotor_speed_error = rotor_speed_reference - self._estimator._rotor_speed_filtered

        self._torque_reference_base = np.interp(self._estimator._rotor_speed_filtered * self._turbine.gearbox_ratio,
                                                self._speed_torque_table[:, 0],
                                                self._speed_torque_table[:, 1])

        if self._torque_reference_integrator is None:
            self._torque_reference_integrator = self._estimator._measured_generator_torque - self._torque_reference_base \
                                                - self._torque_proportional_gain * rotor_speed_error

        self._torque_reference_integrator += self._torque_integrator_gain * rotor_speed_error * self._estimator._sampling_time

        self._torque_reference_proportional = self._torque_proportional_gain * rotor_speed_error
        logger.info(self._torque_reference_base)
        logger.info(self._torque_reference_proportional)
        logger.info(self._torque_reference_integrator)
        return self._torque_reference_base + self._torque_reference_integrator + self._torque_reference_proportional


def main():
    turbines = [TurbineModel()] #, TurbineModel(), TurbineModel()]
    sample_time = 0.2
    controller = TorqueController(len(turbines), sample_time=sample_time)
    estimator = controller._estimator

    measured_rotor_speed = np.zeros((1, len(turbines)))
    measured_generator_torque = np.zeros((1, len(turbines)))
    measured_blade_pitch = np.zeros((1, len(turbines)))

    torque_set_point = np.zeros((1, len(turbines)))
    pitch_set_point = np.zeros((1, len(turbines)))

    steps = 1000
    true_wind_speed = 9. * np.ones((steps, len(turbines)))
    true_wind_speed[:,0] += np.cos(np.arange(0,steps,1)/100)
    true_wind_speed += 0.5 * np.random.randn(steps, len(turbines))
    # true_wind_speed[:, 1] += 2.
    # true_wind_speed[:, 2] += 4.

    estimated_wind_speed = estimator._wind_speed.copy()
    filtered_rotor_speed = estimator._rotor_speed_filtered.copy()
    measured_rotor_speed_series = measured_rotor_speed.copy()
    estimated_tsr = estimator._tsr.copy()

    tsr_desired = 10. * np.ones_like(true_wind_speed)
    tsr_desired[:, 0] += np.round(np.cos(np.arange(0, steps, 1) / 50))
    torque_series = np.zeros_like(tsr_desired)
    for step in range(steps):

        # iterate and gather measurements
        for idx in range(len(turbines)):
            turbines[idx].run_time_step(wind_speed=true_wind_speed[step, idx], torque=torque_set_point[0, idx],
                                        pitch=pitch_set_point[0, idx])
            measured_rotor_speed[0, idx] = turbines[idx].get_rotor_speed()
            measured_generator_torque[0, idx] = turbines[idx].get_torque()
            measured_blade_pitch[0, idx] = turbines[idx].get_pitch()

        measured_rotor_speed_series = np.concatenate((measured_rotor_speed_series, measured_rotor_speed.copy()))

        # estimate wind speed
        controller.run_estimator(measured_rotor_speed, measured_generator_torque, measured_blade_pitch)

        estimated_wind_speed = np.concatenate((estimated_wind_speed, estimator._wind_speed.copy()))
        filtered_rotor_speed = np.concatenate((filtered_rotor_speed, estimator._rotor_speed_filtered.copy()))
        estimated_tsr = np.concatenate((estimated_tsr, estimator._tsr.copy()))

        # control turbines
        torque_set_point = controller.generate_torque_reference(tsr_desired[step, :])
        torque_series[step,:] = torque_set_point
        print(torque_set_point)
        # print(torque_set_point)
        # for idx in range(len(turbines)):
        #     turbines[idx].set_torque(generator_torque_reference[0,idx])

        # record output wind speed

    fig, ax = plt.subplots(2,2, sharex='all',figsize=(10,6))
    ax = ax.ravel()
    ax[0].set_ylabel(labels['u'])
    ax[0].plot(np.arange(1, steps + 1, 1)*sample_time, true_wind_speed, "-", lw=1,c='k',alpha=0.2)
    ax[0].plot(np.arange(0, steps + 1, 1)*sample_time, estimated_wind_speed, lw=1, c="k")

    ax[1].set_ylabel(labels['w'])
    ax[1].plot(np.arange(0, steps + 1, 1)*sample_time, measured_rotor_speed_series, "-",lw=1,c='k', alpha=0.2)
    ax[1].plot(np.arange(0, steps + 1, 1)*sample_time, filtered_rotor_speed, lw=1,c='k')

    actual_tsr = np.divide(measured_rotor_speed_series[1:, :] * np.pi / 30 * turbines[0]._parameters.rotor_radius,
                           true_wind_speed)
    ax[2].set_ylabel(labels['tsr'])
    ax[2].plot(np.arange(1, steps + 1, 1) * sample_time, actual_tsr, lw=1, c='k', alpha=0.2,label='actual')
    ax[2].plot(np.arange(1, steps + 1, 1)*sample_time, tsr_desired, 'k:',lw=1,label='reference')
    ax[2].plot(np.arange(0, steps + 1, 1)*sample_time, estimated_tsr,lw=1, c="k",label='estimate')
    ax[3].set_ylabel(labels['torque'])
    ax[3].plot(np.arange(1, steps + 1, 1)*sample_time, torque_series, 'k-', lw=1)


    # ax[-1].set_xlabel('discrete time step (-)')
    ax[-1].set_xlabel(labels['t'])
    ax[0].set_xlim(0,steps*sample_time)
    ax[0].set_ylim(7,11)
    ax[1].set_ylim(8,12)
    ax[2].set_ylim(8,12)
    ax[2].legend(loc=4)
    ax[-2].set_xlabel(labels['t'])
    fig.tight_layout()
    fig.savefig("../figures/tsrtracker.png",dpi=600, format="png")
    # ax[2].plot(np.arange(0, 10001, 1),rotor_speed_series)
    plt.show()


if __name__ == '__main__':
    main()
