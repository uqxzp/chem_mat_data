# Streaming Datasets

The ``chem_mat_data`` package provides an alternative approach to loading datasets through *streaming datasets* which load data lazily on-demand rather than loading entire datasets into memory at once. This approach is particularly beneficial when working with large datasets or in memory-constrained environments.

## Motivation

While the pre-processed graph datasets (see [Loading Datasets](api_datasets.md)) offer convenience by providing ready-to-use graph representations, they require loading the entire dataset into memory. For large datasets with hundreds of thousands of molecules, this can consume significant RAM and may not be feasible on consumer hardware or in shared computing environments.

Streaming datasets address these challenges through several key advantages:

**Low Memory Footprint**: Only the data currently being processed needs to be in memory. A streaming dataset can process arbitrarily large datasets - even those with millions of molecules - on hardware with limited RAM. This is because molecules are loaded and processed one at a time (or in small batches) rather than loading the entire dataset upfront.

**Reduced Download Size**: The streaming ``GraphDataset`` is based on compressed raw representations (SMILES strings or XYZ files) rather than pre-computed graph structures. These raw formats are significantly more compact than full graph representations, resulting in smaller file downloads and reduced bandwidth usage.

**Flexible Processing**: Since molecules are converted from raw representations to graphs on-the-fly, you can easily apply custom featurization by providing your own processing class. This eliminates the need to download multiple versions of the same dataset with different feature encodings.

**Immediate Training**: With streaming datasets, model training can begin immediately as data is loaded and processed in the background. There is no waiting period for the entire dataset to be loaded into memory before training starts.

**Support for Multiple Formats**: Streaming datasets work with both SMILES-based datasets (for connectivity only) and XYZ-based datasets (with 3D geometry), providing flexibility for different types of molecular modeling tasks.

These benefits make streaming datasets the recommended approach for working with large-scale datasets, experimenting with custom features, or training in resource-constrained environments.

## SmilesDataset

The ``SmilesDataset`` class provides streaming access to raw molecular datasets in SMILES format. It reads the CSV file containing SMILES strings and target properties line-by-line, yielding one molecule at a time without loading the entire dataset into memory.

This is the most memory-efficient way to work with molecular datasets and is particularly useful when you need access to the raw SMILES representations, for example when working with molecular fingerprints or other SMILES-based machine learning methods.

```python
from chem_mat_data.dataset import SmilesDataset

# Create streaming dataset
dataset = SmilesDataset(
    dataset='aqsoldb',
    smiles_column='smiles',
    target_columns=['Solubility']
)

# Iterate through molecules one at a time
for smiles, properties in dataset:
    print(f"SMILES: {smiles}")
    print(f"Properties: {properties}")
    break  # Just show first example
```

The ``SmilesDataset`` automatically downloads and caches the raw CSV file if it doesn't exist locally. Each iteration yields a tuple of ``(smiles_string, properties_array)`` where the properties are returned as a numpy array.

## XyzDataset

The ``XyzDataset`` class provides streaming access to molecular datasets in XYZ format with 3D coordinate information. Unlike SMILES datasets which only contain molecular connectivity, XYZ datasets include the full 3D geometry of molecules - making them essential for tasks that require spatial information such as conformer generation, force field training, or 3D-QSAR methods.

XYZ datasets are stored as "xyz_bundle" folders containing individual .xyz files (one per molecule) and an optional ``meta.csv`` file with target properties and metadata. The dataset streams through these files one at a time, yielding molecular geometry and properties without loading the entire dataset into memory.

```python
from chem_mat_data.dataset import XyzDataset

# Create streaming XYZ dataset
dataset = XyzDataset(
    dataset='qm9',
    parser_cls='qm9',  # Specify parser for QM9 format
    target_columns=['U0', 'U', 'H']
)

# Iterate through molecules one at a time
for xyz_data, properties in dataset:
    print(f"Atoms: {xyz_data['num_atoms']}")
    print(f"Positions shape: {xyz_data['positions'].shape}")  # (N, 3)
    print(f"Atomic numbers: {xyz_data['atomic_numbers']}")
    print(f"Properties: {properties}")
    break  # Just show first example
```

The ``XyzDataset`` automatically downloads and caches the xyz_bundle folder if it doesn't exist locally. Each iteration yields a tuple of ``(xyz_data, properties_array)`` where:

- ``xyz_data`` is a dictionary containing:
    - ``positions``: numpy array of shape (N, 3) with 3D atomic coordinates
    - ``atomic_numbers``: numpy array of shape (N,) with atomic numbers
    - ``symbols``: numpy array of shape (N,) with atom symbols (e.g., 'C', 'O', 'N')
    - ``num_atoms``: integer count of atoms in the molecule
- ``properties`` is a numpy array with target values

!!! info "XYZ Format Variations"

    Different datasets use slightly different XYZ file formats to encode additional information. The ``parser_cls`` parameter specifies which parser to use when loading a specific dataset. Use ``cmdata info <dataset-name>`` to check which parser is required for a particular dataset. Common parsers include 'default', 'qm9', and 'hopv15'.

## GraphDataset

The ``GraphDataset`` class extends the streaming approach to provide on-the-fly conversion from raw molecular representations (SMILES strings or XYZ structures) to graph dict representations. Each molecule is processed into the same graph dict format used by the pre-processed datasets (see [Graph Representation](graph_representation.md)), but the processing happens as you iterate through the dataset rather than being pre-computed.

``GraphDataset`` automatically detects whether to load data as a ``SmilesDataset`` or ``XyzDataset`` based on what's available for the specified dataset. This means you can use the same code for both SMILES-based and XYZ-based datasets, and the appropriate loader will be selected automatically.

### Sequential Processing

By default, or when setting ``num_workers=0``, the ``GraphDataset`` operates in sequential mode. This processes one molecule at a time in a truly streaming manner with minimal memory overhead:

```python
from rich.pretty import pprint
from chem_mat_data.dataset import GraphDataset

# Create streaming graph dataset from SMILES (sequential mode)
dataset = GraphDataset(
    dataset='aqsoldb',
    num_workers=0,  # Sequential processing
    smiles_column='smiles',
    target_columns=['Solubility']
)

# Iterate through graphs one at a time
for graph in dataset:
    pprint(graph)
    break  # Just show first example
```

Sequential processing is optimal for typical molecular datasets where RDKit graph generation is relatively fast (around 0.5ms per molecule). The simplicity of sequential processing avoids the overhead of multiprocessing coordination, making it the fastest option for most use cases.

### Working with XYZ Datasets

``GraphDataset`` works seamlessly with XYZ-based datasets, automatically detecting and using ``XyzDataset`` when appropriate. This is particularly useful for datasets like QM9 that provide 3D geometries:

```python
from chem_mat_data.dataset import GraphDataset

# Create streaming graph dataset from XYZ data
# GraphDataset automatically detects this is an XYZ dataset
dataset = GraphDataset(
    dataset='qm9',
    parser_cls='qm9',  # Required for XYZ datasets
    target_columns=['U0', 'U', 'H'],
    num_workers=0
)

# Iterate through graphs generated from 3D structures
for graph in dataset:
    # Same graph dict format as SMILES-based datasets
    print(f"Nodes: {len(graph['node_indices'])}")
    print(f"Node features shape: {graph['node_attributes'].shape}")
    break
```

The automatic detection follows a simple rule: it first tries to load as a ``SmilesDataset`` (checking for .csv file), and if that fails, it attempts to load as an ``XyzDataset`` (checking for .xyz_bundle folder). This transparent handling means you don't need to explicitly specify the dataset type.

### Parallel Processing

For complex molecules or datasets where processing time per molecule is significant, the ``GraphDataset`` supports parallel processing using multiple worker processes:

```python
from chem_mat_data.dataset import GraphDataset

# Create streaming graph dataset with parallel processing
dataset = GraphDataset(
    dataset='aqsoldb',
    num_workers=4,      # Use 4 parallel worker processes
    buffer_size=100,    # Number of molecules to queue ahead
    smiles_column='smiles',
    target_columns=['Solubility']
)

# Iterate through graphs with parallel processing
for graph in dataset:
    # Graphs are processed in parallel while maintaining order
    pass
```

The parallel mode uses Python's multiprocessing to bypass the Global Interpreter Lock (GIL), enabling true CPU parallelism. The dataset maintains the original molecule order even though workers may complete processing at different rates. This is achieved through an internal priority queue that reorders results before yielding them.

!!! info "When to Use Parallel Mode"

    Parallel processing introduces overhead from process coordination, queue operations, and data serialization. For typical small molecules, sequential mode is actually faster. Use parallel mode when:

    - Processing complex molecules (e.g., large polymers, proteins)
    - Using custom processing classes with CPU-intensive computations
    - Working with datasets where processing time exceeds 2ms per molecule
    - You have 4 or more CPU cores available for better amortization of overhead

### Custom Processing

The ``GraphDataset`` accepts a custom ``processing_class`` parameter to define your own featurization. This is particularly powerful for streaming datasets since you can experiment with different feature encodings without re-downloading the dataset:

```python
from chem_mat_data.dataset import GraphDataset
from chem_mat_data.processing import MoleculeProcessing

# Use custom processing class (see Custom Pre-Processing docs)
class CustomProcessing(MoleculeProcessing):
    # Define custom node/edge attributes here
    pass

dataset = GraphDataset(
    dataset='aqsoldb',
    processing_class=CustomProcessing,
    num_workers=0
)

for graph in dataset:
    # Graphs use your custom feature encoding
    pass
```

For details on creating custom processing classes, see [Custom Pre-Processing](custom_pre_processing.md).

## ShuffleDataset

Machine learning training typically requires data to be shuffled to prevent the model from learning spurious patterns based on data ordering. The ``ShuffleDataset`` wrapper provides approximate shuffling of streaming datasets using a shuffle buffer algorithm.

### How Shuffle Buffering Works

The shuffle buffer algorithm provides randomization without loading the entire dataset into memory by maintaining a fixed-size buffer from which items are randomly selected and yielded, with each yielded item being replaced by the next item from the stream. The buffer size controls the trade-off between shuffling quality and memory usage - larger buffers provide better global shuffling (with ``buffer_size >= dataset_size`` giving perfect shuffling), while smaller buffers provide approximate shuffling with lower memory consumption. For most machine learning applications, a buffer size of 5,000-10,000 provides good shuffling quality while maintaining reasonable memory usage.

### Usage in Training

The ``ShuffleDataset`` is particularly useful for training *graph neural networks* where shuffled mini-batches are required:

```python
from chem_mat_data.dataset import GraphDataset, ShuffleDataset

# Create base streaming dataset
base_dataset = GraphDataset(
    dataset='aqsoldb',
    num_workers=4
)

# Wrap with shuffle buffer
shuffled_dataset = ShuffleDataset(
    dataset=base_dataset,
    buffer_size=5000,  # Shuffle buffer of 5000 molecules
    seed=42            # Fixed seed for reproducibility
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for graph in shuffled_dataset:
        # Each epoch iterates through all molecules in shuffled order
        # With same seed, each epoch uses identical shuffle order
        train_step(graph)
```

!!! info "Reproducibility"

    Setting a fixed ``seed`` ensures that each iteration through the ``ShuffleDataset`` produces the same shuffle order. This is important for reproducible machine learning experiments. If ``seed=None``, each iteration will produce a different random shuffle.

### Integration with Deep Learning Frameworks

While the streaming datasets are framework-agnostic, they work seamlessly with PyTorch DataLoaders and similar abstractions:

```python
from chem_mat_data.dataset import GraphDataset, ShuffleDataset
from chem_mat_data.graph import graph_dict_to_pyg

# Create shuffled streaming dataset
dataset = ShuffleDataset(
    GraphDataset(dataset='aqsoldb', num_workers=4),
    buffer_size=5000,
    seed=42
)

# Training loop with on-the-fly conversion to PyG format
for graph_dict in dataset:
    # Convert to PyTorch Geometric format
    pyg_graph = graph_dict_to_pyg(graph_dict)

    # Use with your GNN model
    output = model(pyg_graph)
    loss = criterion(output, target)
    # ...
```

For details on converting graph dicts to framework-specific formats, see [Graph Representation](graph_representation.md).

## Use Cases

### When to Use Streaming Datasets

Streaming datasets are recommended when:

- **Large Datasets**: Working with datasets containing more than 100,000 molecules where memory consumption is a concern
- **Limited Hardware**: Training on consumer hardware or shared infrastructure with limited RAM
- **Custom Features**: Experimenting with different featurization approaches without re-downloading datasets
- **Immediate Prototyping**: Quick iteration where you want to start training immediately without waiting for full dataset loading
- **3D Geometry Data**: Working with XYZ datasets that include spatial coordinates for conformer-dependent predictions

### When to Use Pre-Processed Datasets

Pre-processed datasets (loaded via ``load_graph_dataset``) are better when:

- **Small Datasets**: For datasets under 50,000 molecules that easily fit in memory
- **Multiple Epochs**: Training for many epochs where the one-time loading cost is amortized
- **Random Access**: Algorithms that require index-based access to specific molecules
- **Standardization**: When consistent, pre-computed features across research groups are important

### Choosing Between SMILES and XYZ

The choice between ``SmilesDataset`` and ``XyzDataset`` depends on your modeling needs:

- **Use SMILES** for:
    - 2D molecular property prediction (e.g., toxicity, solubility, bioactivity)
    - Graph-based methods where only connectivity matters
    - Maximum memory efficiency (SMILES strings are extremely compact)
    - Datasets without 3D structural information

- **Use XYZ** for:
    - 3D molecular property prediction (e.g., binding energies, conformer stability)
    - Force field development and quantum chemistry applications
    - Methods that require spatial information (e.g., 3D-QSAR, pharmacophore modeling)
    - Datasets like QM9 that provide valuable geometric information

When using ``GraphDataset``, you don't need to explicitly choose - it automatically detects which format is available for your dataset and uses the appropriate loader.

In practice, streaming datasets and pre-processed datasets are complementary approaches. The ``chem_mat_data`` package provides both options so you can choose the approach that best fits your specific use case and computational environment.
