import numpy as np
import h5py
import json

class H5FileWriter:
    def __init__(self, output_path: str, group_key: np.uint32, metadata: dict):
        """
        :param output_path: The file path where the new H5 file will be written.
        :param group_key: The 32-bit integer group key.
        :param metadata: A dictionary containing metadata (e.g. access_events, parameters, etc.).
        """
        self.output_path = output_path
        self.group_key = group_key
        self.metadata = metadata

    def write(self, data: np.ndarray):
        with h5py.File(self.output_path, 'w') as f:
            # Create a group with the group key as its name.
            group_name = str(self.group_key)
            grp = f.create_group(group_name)

            # Write the data into a dataset within this group.
            grp.create_dataset('data', data=data)

            # Write the metadata into attributes.
            # For example, store the access event log as a JSON string.
            grp.attrs['access_events'] = json.dumps(self.metadata.get("access_events", []))
            for key, value in self.metadata.items():
                if key != "access_events":
                    # If the value is not a string, convert it.
                    grp.attrs[key] = str(value)

