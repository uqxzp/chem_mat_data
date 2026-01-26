Add to config file to enable LLM access to local files:

"permission": {
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "bash": "allow",
    "write": "allow",
    "crawlfetch": "allow",
    "webfetch": "deny",
    "websearch": "deny"
  }

How to use:

0. opencode serve --port 54321 --print-logs
1. cmmanage agent process <link>
2. cmmanage agent generate
3. python chem_mat_data/agent/artifacts/scripts/<name>_generated.py 