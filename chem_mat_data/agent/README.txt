Add to config file to enable LLM access to local files:


"tools": {
  "read": true,
  "list": true,
  "glob": true,
  "grep": true,
  "write": false,
  "edit": false,
  "patch": false,
  "bash": false
}

How to use:

1. cmmanage agent process <link>
2. cmmanage agent generate
3. python chem_mat_data/agent/artifacts/scripts/<name>_generated.py 

Paper used for testing: https://www.nature.com/articles/s41467-020-16201-z