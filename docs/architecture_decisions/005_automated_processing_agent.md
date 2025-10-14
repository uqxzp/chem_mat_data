# Automated Dataset Processing Agent

## Status

planning

## Context

Currently, there are two major bottlenecks in the extension of the ChemMatDatabase:
1. The discovery of new datasets in the scientific literature
2. The processing of the *very* different file formats into a unified format

The bottleneck is especially the second part of this - the processing of the strange datasets formats.
In the ChemMatData package most of this processing complexity is already encapsulated in the `create_graph_datasets.py` 
module which handles most of the processing. The only thing required to create a new dataset is to inherit 
from that base experiment `create_graph_datasets__{dataset_name}.py` and to overwrite a function which 
provides the dataset in an intermediate format for the rest of the processing functionality to work on.

## Decision

To address the substantial manual effort, it has been decided to employ an AI Coding Agent to alleviate 
as much of this as possible. In the end, this processing capability is meant to become part of the 
management interface `cmmanage` like this:

```python
# Given the online source for the original publication about the dataset
cmmanage agent process https://link.to/some/dataset/source/online

# Or locally the downloaded datasets already
cmmanage agent process /local/folder/containing/files
```