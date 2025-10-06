# ``cache`` - Interacting with the Local File System Cache

The ``chem_mat_data`` package can be used to download various datasets from a remote file share server. 
To make sure that datasets only need to be actually downloaded once, the package uses a local file system 
*cache*. Whenever a dataset is downloaded for the first time, the corresponding files are placed in the cache 
and if the same dataset is requested again, it is simply copied from the cache rather than downloaded from 
the remote server.

The ``cmdata`` command line interface provides the ``cache`` command group to interact with this cache 
folder:

```bash
cmdata cache --help
```

## Viewing Cache Information

To view detailed information about the cache including its location, total size, and statistics, you can use the ``cache info`` command:

```bash
cmdata cache info
```

This will display comprehensive information about the cache directory, including the cache path, total size in MB or GB, the number of datasets stored, and the total number of files in the cache.

## Viewing the Cache Content

To view all the files that are currently stored in the cache, you can use the ``cache list`` command:

```bash
cmdata cache list
```

This will print a list view of all the files that are currently stored in the cache directory, divided
by the raw and processed dataset files.

## Removing a Specific Dataset

To remove a specific dataset from the cache, you can use the ``cache remove`` command:

```bash
cmdata cache remove "dataset_name"
```

By default, this removes the ``.mpack`` (processed) version of the dataset. You can specify a different format using the ``--type`` option:

```bash
cmdata cache remove "dataset_name" --type csv
```

This command will remove only the specified dataset file from the cache, leaving all other cached datasets intact.

## Resetting the Cache

If you wich to reset the cache for some reason, you can use the ``cache clear`` command:

```bash
cmdata cache clear
```

This command will delete all the files in the cache folder.