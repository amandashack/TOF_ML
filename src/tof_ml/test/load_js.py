import json
import h5py

# Load the JSON file
with open('spectra_info.json', 'r') as json_file:
    spectra_info = json.load(json_file)

# Define the base path
base_path = r'C:\Users\proxi\Documents\coding\simulations\TOF_simulation\simion_output\collection_efficiency'


def load_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        # Extract data
        data = f['data'][:]

        # Extract dimensions
        keys = list(f.keys())
        keys.remove('data')  # Remove the 'data' key

        # Determine the correct order of dimensions by matching lengths
        axes = {key: f[key][:] for key in keys}
        dims = [None] * len(data.shape)
        coords = {}

        for key, axis in axes.items():
            axis_len = axis.shape[0]
            for i, dim_len in enumerate(data.shape):
                if axis_len == dim_len:
                    dims[i] = key
                    coords[key] = axis
                    break

        # Check if all dimensions are assigned
        if None in dims:
            raise ValueError("Could not match all dimensions with the data shape.")

        # Extract attributes
        attrs = dict(f.attrs)

    # Create DataArray
    data_array = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs=attrs
    )

    return data_array


def load_data_from_json(json_file, retardation=None, voltage1=None, voltage2=None):
    with open(json_file, 'r') as file:
        data = json.load(file)

    def filter_data(item):
        if retardation is not None:
            if isinstance(retardation, str):
                if (retardation == 'neg' and item['retardation'] >= 0) or (
                        retardation == 'pos' and item['retardation'] <= 0):
                    return False
            elif isinstance(retardation, list):
                if item['retardation'] not in retardation:
                    return False

        if voltage1 is not None:
            if isinstance(voltage1, str):
                if (voltage1 == 'neg' and item['voltage1'] >= 0) or (voltage1 == 'pos' and item['voltage1'] <= 0):
                    return False
            elif isinstance(voltage1, list):
                if item['voltage1'] not in voltage1:
                    return False

        if voltage2 is not None:
            if isinstance(voltage2, str):
                if (voltage2 == 'neg' and item['voltage2'] >= 0) or (voltage2 == 'pos' and item['voltage2'] <= 0):
                    return False
            elif isinstance(voltage2, list):
                if item['voltage2'] not in voltage2:
                    return False

        return True

    filtered_data = [item for item in data if filter_data(item)]

    return filtered_data


# Example usage
json_file = '../simulations/simulation_data.json'
retardation_filter = 'neg'  # Can be 'neg', 'pos', or a list of values
#voltage1_filter = [4.4376]  # Can be 'neg', 'pos', or a list of values
#voltage2_filter = None  # Can be 'neg', 'pos', or a list of values

filtered_data = load_data_from_json(json_file, retardation=retardation_filter)
print(filtered_data)
