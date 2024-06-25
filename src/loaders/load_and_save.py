import xarray as xr
import os
import h5py


def save_xarray(data_array, directory, filename, method='h5'):
    """
    Save an xarray.DataArray to an HDF5 file.

    Parameters:
    - data_array: xarray.DataArray, the data array to save.
    - directory: str, the directory where the file will be saved.
    - filename: str, the name of the file (without extension).

    The function will create the directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, f"{filename}.h5")
    if method == "netcdf":
        data_array.to_netcdf(filepath, format='NETCDF4', engine='netcdf4')
    else:
        with h5py.File(filepath, 'w') as f:
            for dim in data_array.dims:
                f.create_dataset(dim, data=data_array[dim].values)
            f.create_dataset('data', data=data_array.values)
            for attr, value in data_array.attrs.items():
                f.attrs[attr] = value


def load_xarray(directory, filename, method='h5'):
    """
    Load an xarray.DataArray from an HDF5 file.

    Parameters:
    - directory: str, the directory where the file is located.
    - filename: str, the name of the file (without extension).

    Returns:
    - data_array: xarray.DataArray, the loaded data array.
    """
    filepath = os.path.join(directory, f"{filename}.h5")
    if method == 'netcdf':
        return xr.open_dataarray(filepath, engine='netcdf4')
    else:
        with h5py.File(filepath, 'r') as f:
            # Load dimensions
            coords = {dim: f[dim][:] for dim in reversed(f.keys()) if dim != 'data'}

            # Load the data values
            data = f['data'][:]

            # Load attributes
            attrs = {attr: f.attrs[attr] for attr in f.attrs}

        # Create the xarray DataArray
        xarr = xr.DataArray(data, coords=coords, dims=list(coords.keys()), attrs=attrs)
        return xarr


def save_to_h5(array, filename):
    """
    Save a NumPy array to an HDF5 file.

    Parameters:
    array (np.ndarray): The NumPy array to save.
    filename (str): The name of the HDF5 file to save the array in.
    """
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('data', data=array)
    print(f"Data saved to {filename}")


def load_from_h5(filename):
    """
    Load a NumPy array from an HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file to load the array from.

    Returns:
    np.ndarray: The loaded NumPy array.
    """
    with h5py.File(filename, 'r') as h5f:
        array = h5f['data'][:]
    print(f"Data loaded from {filename}")
    return array

