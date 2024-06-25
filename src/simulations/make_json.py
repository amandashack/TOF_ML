import os
import re
import json
import h5py

def parse_filename(filename):
    pattern = re.compile(r"sim_(neg|pos)_R(-?\d+)_(neg|pos)_(-?\d+\.\d+)_(neg|pos)_(-?\d+\.\d+)_(\d+)\.h5")
    match = pattern.match(filename)
    if not match:
        return None

    sign_map = {'neg': -1, 'pos': 1}

    retardation_sign = sign_map[match.group(1)]
    retardation_value = int(match.group(2))
    mid1_ratio_sign = sign_map[match.group(3)]
    mid1_ratio_value = float(match.group(4))
    mid2_ratio_sign = sign_map[match.group(5)]
    mid2_ratio_value = float(match.group(6))
    kinetic_energy = int(match.group(7))

    return {
        'retardation': retardation_sign * retardation_value,
        'mid1_ratio': mid1_ratio_sign * mid1_ratio_value,
        'mid2_ratio': mid2_ratio_sign * mid2_ratio_value,
        'kinetic_energy': kinetic_energy
    }

def create_json_from_filenames(base_directory, output_json):
    data = []

    for subdir, _, files in os.walk(base_directory):
        for filename in files:
            if filename.endswith(".h5"):
                parsed_data = parse_filename(filename)
                if parsed_data:
                    full_path = os.path.join(subdir, filename)
                    data.append({
                        'filename': full_path,
                        **parsed_data
                    })

    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def read_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        data = {
            'initial_ke': f['data1']['initial_ke'][:],
            'initial_elevation': f['data1']['initial_elevation'][:],
            'x_tof': f['data1']['x'][:],
            'y_tof': f['data1']['y'][:],
            'tof_values': f['data1']['tof'][:],
            'final_elevation': f['data1']['final_elevation'][:],
            'final_ke': f['data1']['final_ke'][:],
        }
    return data

def write_h5_file(filename, data):
    with h5py.File(filename, 'w') as f:
        for group_name, group_info in data.items():
            grp = f.create_group(group_name)
            for dataset_name, dataset_values in group_info['data'].items():
                grp.create_dataset(dataset_name, data=dataset_values)
            grp.attrs['mid1_ratio'] = group_info['mid1_ratio']
            grp.attrs['mid2_ratio'] = group_info['mid2_ratio']
            grp.attrs['kinetic_energy'] = group_info['kinetic_energy']

def organize_data(json_file_path):
    with open(json_file_path, 'r') as file:
        file_info = json.load(file)

    organized_data = {}
    for entry in file_info:
        # Check if the entry contains 'mid1_ratio' to avoid processing combined files
        if 'mid1_ratio' not in entry:
            continue

        retardation = entry['retardation']
        mid1_ratio = entry['mid1_ratio']
        mid2_ratio = entry['mid2_ratio']
        kinetic_energy = entry['kinetic_energy']
        filename = entry['filename']

        group_name = f"{mid1_ratio}_{mid2_ratio}_{kinetic_energy}"
        if retardation not in organized_data:
            organized_data[retardation] = {
                'path': os.path.dirname(filename),  # Store the path for each retardation value
                'data': {}
            }

        data = read_h5_file(filename)
        organized_data[retardation]['data'][group_name] = {
            'data': data,
            'mid1_ratio': mid1_ratio,
            'mid2_ratio': mid2_ratio,
            'kinetic_energy': kinetic_energy
        }

    return organized_data

def create_new_h5_files(organized_data):
    new_files = []
    for retardation, info in organized_data.items():
        sign = "neg" if retardation < 0 else "pos"
        ret_value = abs(retardation)
        new_filename = os.path.join(info['path'], f"sim_{sign}_R{ret_value}.h5")
        write_h5_file(new_filename, info['data'])
        new_files.append({
            'filename': new_filename,
            'retardation': retardation,
            'combined': True
        })
    return new_files

def update_json_with_new_files(original_json, new_files):
    with open(original_json, 'r') as file:
        data = json.load(file)

    data.extend(new_files)

    with open(original_json, 'w') as file:
        json.dump(data, file, indent=4)

# Paths
base_dir = r'C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency'
json_file_path = 'simulation_data.json'  # Update this to the path of your JSON file

# Create or update the JSON file with all filenames
create_json_from_filenames(base_dir, json_file_path)

# Organize data and create new H5 files
organized_data = organize_data(json_file_path)
new_files = create_new_h5_files(organized_data)

# Update JSON with new files
update_json_with_new_files(json_file_path, new_files)

print("New H5 files created and JSON file updated successfully.")


