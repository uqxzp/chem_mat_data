"""
This experiment module converts the QM9 dataset from figshare into the ChemMatData graph format.
The dataset consists of two bzip2-compressed files containing XYZ files for molecules.
We download, extract, and parse the XYZ files locally using RDKit.
"""
import os
import bz2
import tarfile
import requests
import pandas as pd
from typing import Dict, List
import rdkit.Chem as Chem

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param DOWNLOAD_URLS:
#       List of URLs to download the dataset files from figshare.
DOWNLOAD_URLS: List[str] = [
    'https://ndownloader.figshare.com/files/3195389',
    'https://ndownloader.figshare.com/files/3195398'
]
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name.
DATASET_NAME: str = 'qm9'

__TESTING__ = False

BASE_EXPERIMENT = os.path.join(os.path.dirname(__file__), 'create_graph_datasets__from_xyz.py')
if os.path.exists(BASE_EXPERIMENT):
    experiment = Experiment.extend(
        BASE_EXPERIMENT,
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals(),
    )
else:
    # Fallback to standalone Experiment
    experiment = Experiment(
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals(),
    )


def load_local_xyz_dataset(folder_path: str) -> pd.DataFrame:
    """
    Load XYZ dataset from a local folder containing .xyz files.
    Parses each .xyz file using RDKit and returns a DataFrame.
    """
    data = []
    total_files = 0
    parsed_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.xyz'):
            total_files += 1
            filepath = os.path.join(folder_path, filename)
            try:
                mol = Chem.MolFromXYZFile(filepath)
                if mol is not None:
                    data.append({'mol': mol, 'filename': filename})
                    parsed_count += 1
                else:
                    experiment.log(f'Failed to parse {filename}: RDKit returned None')
            except Exception as e:
                experiment.log(f'Error parsing {filename}: {e}')
    
    success_rate = parsed_count / total_files if total_files > 0 else 0
    if success_rate < 0.8:
        raise ValueError(f'Parsing success rate too low: {success_rate:.2%} ({parsed_count}/{total_files})')
    
    experiment.log(f'Parsed {parsed_count}/{total_files} XYZ files successfully')
    return pd.DataFrame(data)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download, extract, and load the dataset from the provided URLs.
    """
    e.log('Downloading and extracting dataset...')
    
    extract_path = os.path.join(e.path, 'extracted')
    os.makedirs(extract_path, exist_ok=True)
    
    for url in e.DOWNLOAD_URLS:
        e.log(f'Downloading {url}...')
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the bz2 file
        filename = url.split('/')[-1] + '.bz2'
        bz2_path = os.path.join(e.path, filename)
        with open(bz2_path, 'wb') as f:
            f.write(response.content)
        
        # Extract bz2
        with bz2.BZ2File(bz2_path, 'rb') as f_in:
            extracted_data = f_in.read()
        
        # Check if extracted data is a tar file
        if extracted_data.startswith(b'\x75\x73\x74\x61\x72'):  # ustar signature
            import io
            tar = tarfile.open(fileobj=io.BytesIO(extracted_data))
            tar.extractall(extract_path)
        else:
            # Assume it's a single file or handle as needed; for QM9, likely tar inside bz2
            pass
    
    # Now, load from extract_path
    df = load_local_xyz_dataset(extract_path)
    
    index_data_map: Dict[int, dict] = {}
    for idx, row in df.iterrows():
        index_data_map[idx] = row.to_dict()
    
    return index_data_map


experiment.run_if_main()