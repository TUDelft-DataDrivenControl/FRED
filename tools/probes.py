import numpy as np
import matplotlib.pyplot as plt


def write_probe_file():
    domain = (2000., 1000.)
    num_probes = (41, 21)
    hub_height = 119.0
    probe_positions_x = np.linspace(0,domain[0],num_probes[0])
    probe_positions_y = np.linspace(0,domain[1],num_probes[1])

    probe_position_array = np.zeros((num_probes[0]*num_probes[1],3))

    filename = "probe_field"
    probe_name = "probe0"

    def write_header():
        header_lines = [
            "{:s}\r\n".format(filename),
            "{\r\n",
            "type probes;\r\n",
            'functionObjectLibs ("libsampling.so");\r\n',
            "name {:s}\r\n".format(probe_name),
            "outputControl timeStep;\r\n",
            "outputInterval 1;\r\n",
            "surfaceFormat vtk;\r\n",
            "fields \r\n (\r\n U \r\n );\r\n",
            "\r\n",
            "probeLocations\r\n",
            "(\r\n"
        ]
        for line in header_lines:
            file.write(line)

    def write_probes():
        n = 0
        for px in probe_positions_x:
            for py in probe_positions_y:
                probe_position_array[n, :] = [px, py, hub_height]
                probe_position_string = "({:8.0f} {:8.0f} {:8.0f})".format(px, py, hub_height)
                file.write(probe_position_string)
                file.write("\r\n")
                print(probe_position_string)
                n += 1

    def write_tail():
        file.write("); \r\n")
        file.write("}\r\n")

    with open(filename,"w") as file:
        write_header()
        write_probes()
        write_tail()

    plt.figure()
    plt.plot(probe_position_array[:,0], probe_position_array[:,1], '.')
    plt.show()


def read_probe_data(filename="data/U", plot_probes=False):
    with open(filename, "r") as file:
        data = file.readlines()

        probe_positions = []
        time_series = []
        data_series = []

        for data_line in data:
            # Gather probe positions
            if data_line.startswith("# Probe"):
                probe_positions.append([int(x) for x in data_line.rsplit("(")[1].strip(")\n").split(" ")])

            # Gather probe measurements
            elif not data_line.startswith("#"):
                d = data_line.replace(")", "").split("(")
                time_series.append(float(d[0]))
                data_series.append([[float(x) for x in p.split(' ')[0:3]] for p in d[1:]])

        probe_position_array = np.array(probe_positions)
        time_array = np.array(time_series)
        if time_array[0] >= 20000:
            time_array -= 20000
        data_array = np.array(data_series)

        # Probe measurement scatter plot
        if plot_probes:
            plt.figure()
            plt.scatter(probe_position_array[:,0], probe_position_array[:,1],cmap='viridis',c=np.linalg.norm(data_array[-1,:,:],ord=2,axis=1),vmin=3,vmax=10)
            for idx in range(len(probe_position_array)):
                plt.text(probe_position_array[idx,0],
                         probe_position_array[idx,1],
                         "{:d}".format(idx),
                         size='small')
            plt.show()

    return probe_position_array, time_array, data_array



if __name__ == '__main__':
    # write_probe_file()
    p,t,d = read_probe_data("../data/U",plot_probes=True)