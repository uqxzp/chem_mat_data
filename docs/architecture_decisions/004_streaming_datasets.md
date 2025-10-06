# Streaming Datasets

## Status

implemented

## Context

The ChemMatData package provides molecular datasets for machine learning applications, particularly for training graph neural networks. As datasets grow larger (100k+ molecules), several challenges emerge:

**Memory Constraints**: Loading entire datasets into memory becomes problematic. A single molecular graph with features can require several kilobytes, meaning a 100k molecule dataset can easily consume multiple gigabytes of RAM. This limits the ability to work with large datasets on consumer hardware or to train multiple models concurrently on shared infrastructure.

**Processing Bottlenecks**: Converting molecular representations (SMILES strings or XYZ structures) to graph representations using RDKit is CPU-intensive (~0.5ms per molecule). For large datasets, processing all molecules upfront creates a significant startup delay before training can begin. Additionally, if users want to experiment with different featurization approaches, reprocessing the entire dataset each time is inefficient.

**Training Requirements**: Modern deep learning frameworks (PyTorch, JAX) use data loaders that iterate through datasets multiple times during training. These frameworks are designed to work efficiently with streaming data, loading and processing batches on-demand rather than requiring the entire dataset in memory.

**Multiple Data Formats**: Molecular datasets come in different formats:
- SMILES strings represent 2D connectivity (compact, text-based)
- XYZ files include 3D atomic coordinates (essential for conformer-dependent properties)

Supporting both formats while maintaining a unified API is important for flexibility across different modeling tasks.

**Use Case Flexibility**: Different use cases have different requirements:
- Quick prototyping benefits from fast iteration with small subsets
- Large-scale experiments need to handle datasets that exceed available RAM
- Custom featurization pipelines require processing flexibility without re-downloading datasets
- 3D property prediction requires access to geometric information from XYZ datasets

The existing approach of providing pre-processed datasets (ADR 001) works well for convenience but doesn't address these scalability challenges for large datasets, custom processing needs, or flexible format support.

## Decision

Implement a streaming dataset architecture with four main classes in `chem_mat_data/dataset.py`:

### 1. SmilesDataset
A lazy-loading dataset that streams SMILES strings and properties from CSV files line-by-line using Python's `csv.DictReader`. Only one row is loaded into memory at a time, providing minimal memory footprint. Used for 2D molecular datasets where only connectivity matters.

### 2. XyzDataset
A lazy-loading dataset that streams 3D molecular structures from XYZ file bundles. Iterates through individual .xyz files (one per molecule) stored in an xyz_bundle folder, yielding atomic coordinates, atomic numbers, and properties. Supports multiple XYZ format parsers (default, qm9, hopv15) to handle different XYZ file specifications. Essential for tasks requiring 3D geometric information.

### 3. GraphDataset
Extends the streaming approach to convert raw molecular representations (from either SmilesDataset or XyzDataset) into graph dict representations on-the-fly. Automatically detects which dataset type to use based on what's available (tries .csv first, then .xyz_bundle). Supports two processing modes:

**Sequential Mode (num_workers=0)**:
- True streaming: processes one molecule at a time
- Minimal memory usage
- Single-threaded processing (~2000 mol/s for typical molecules)
- Best for memory-constrained environments or simple molecules

**Parallel Mode (num_workers>0)**:
- Multi-process architecture for CPU-bound parallelism
- Producer thread reads raw data (SMILES/XYZ) and distributes work to worker processes
- Worker processes convert raw data to Mol objects then to graphs in parallel (bypasses Python GIL)
- Collector thread reorders results to maintain dataset order
- Results buffer allows main iterator to suspend at yield without deadlock
- Better throughput for complex molecules or CPU-intensive custom processing

**Auto-Detection Architecture**:
- Uses try-except approach: attempts SmilesDataset first, falls back to XyzDataset
- Stores dataset_type string ('smiles' or 'xyz') for worker initialization
- Workers receive raw data (SMILES strings or XYZ dicts) and convert in parallel
- Transparent to users - same API regardless of underlying format

### 4. ShuffleDataset
Wrapper that provides approximate shuffling using a fixed-size shuffle buffer. This enables training with shuffled data without loading the entire dataset into memory. The buffer size controls the trade-off between shuffling quality and memory usage.

### Architecture Highlights

**Deadlock Prevention**: The parallel mode uses a sophisticated producer-collector thread architecture where a separate collector thread continuously drains the result queue. This prevents deadlock when the main iterator suspends at yield, ensuring worker processes never block on a full queue.

**Order Preservation**: Workers may complete molecules out of order (due to varying complexity), but the collector thread uses a priority queue (heap) to reorder results before yielding them, maintaining the original dataset sequence.

**Queue Design**:
- Input queue: bounded (provides backpressure)
- Result queue: unbounded (critical for deadlock prevention)
- Results buffer: thread-local queue (low overhead, no pickling)

**Integration**: Datasets work seamlessly with the existing `ensure_dataset()` function, automatically downloading and caching raw CSV files when needed.

## Consequences

### Advantages

**Memory Efficiency**: Streaming enables working with arbitrarily large datasets on limited hardware. Only the data currently being processed needs to be in memory, not the entire dataset. This is particularly important for cloud environments where memory is a cost factor.

**Multiple Format Support**: Unified API for both SMILES-based and XYZ-based datasets enables seamless handling of 2D connectivity and 3D geometric information. Auto-detection makes format handling transparent to users.

**Flexible Processing**: Users can apply custom featurization by subclassing `MoleculeProcessing` and providing it to GraphDataset. The streaming approach means custom processing happens on-demand rather than requiring reprocessing of stored datasets. Works identically for both SMILES and XYZ sources.

**Faster Startup**: Training can begin immediately as data is streamed and processed on-the-fly, rather than waiting for the entire dataset to be loaded and processed upfront.

**True Parallelism**: Multi-process architecture bypasses Python's GIL, enabling true CPU parallelism for graph processing. This scales effectively with CPU cores for complex molecules or custom processing. Raw data format (SMILES strings or XYZ dicts) minimizes pickling overhead.

**Composability**: The wrapper pattern (ShuffleDataset) allows building more complex data pipelines by composing simple components, similar to PyTorch's dataset philosophy.

**API Consistency**: Streaming datasets implement the standard Python iterator protocol (`__iter__` and `__len__`), making them compatible with existing code that expects iterable datasets. Same interface regardless of underlying format.

### Disadvantages

**Complexity**: The parallel processing implementation is significantly more complex than simple in-memory loading, particularly the deadlock prevention architecture with multiple threads and processes. Supporting multiple dataset formats (SMILES and XYZ) adds additional complexity with the auto-detection mechanism. This increases maintenance burden and makes debugging more challenging.

**Parallel Overhead**: For simple molecules, parallel processing is actually slower than sequential due to overhead (~0.6ms per molecule for pickling, queue operations, and thread coordination vs ~0.5ms processing time). The complexity of the parallel code only pays off for complex molecules or CPU-intensive custom processing.

**No Random Access**: Streaming datasets don't support index-based access (e.g., `dataset[42]`). This is incompatible with some training frameworks or use cases that require random access patterns. However, this is mitigated by the ShuffleDataset wrapper for common training scenarios.

**Repeated Iteration Overhead**: Each iteration through the dataset requires re-reading the raw data (CSV or XYZ files) and re-processing molecules. For training that requires multiple epochs, this processing cost is paid repeatedly. In contrast, pre-processed datasets only pay the cost once during download.

**Format-Specific Parameters**: XyzDataset requires additional parameters (parser_cls) that aren't needed for SmilesDataset. This creates some API inconsistency, though GraphDataset handles this through auto-detection and parameter forwarding.

**Less Discoverable**: Users familiar with the pre-processed datasets (ADR 001) may not immediately discover the streaming alternative. Documentation and examples are critical to guide users toward the appropriate solution for their use case, including when to use SMILES vs XYZ formats.

**Processing Variance**: Unlike pre-processed datasets where all users get identical featurization, streaming datasets with custom processing classes can lead to less comparable results across different experiments or research groups.
