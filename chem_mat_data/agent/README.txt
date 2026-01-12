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

0. opencode serve --port 54321 --print-logs
1. cmmanage agent process <link>
2. cmmanage agent generate
3. python chem_mat_data/agent/artifacts/scripts/<name>_generated.py 

-----

Performance on dataset (corresponding manual script removed during script generation)

EASY 

AqSolDB 
- https://www.nature.com/articles/s41597-019-0151-1
# Download: GPT 5.1 Codex Mini, Processing: not working (tried Grok Code, GPT 5.1 Mini Codex, GPT 5.1 Codex, GPT 5.1, Gemini 3 Flash (broken), Claude Haiku 4.5 (rushed & invalid))

ESOL
- https://pubs.acs.org/doi/10.1021/ci034243x
# GPT 5.1 Codex Mini

MEDIUM

Compas 1x 
- https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00503
# Download: GPT 5.1 Codex Mini, Processing: Grok Code (sometimes)

Half Life
- https://chemrxiv.org/engage/chemrxiv/article-details/64be471cb605c6803b425da6
# Grok Code/GPT 5.1 Codex Mini

Open Melting Point
- https://www.nature.com/articles/npre.2011.6229.1
# Download: Grok Code, GPT 5.1 Codex Mini; Processing: not working (tried Grok Code, GPT 5.1 Mini Codex, GPT 5.1 Codex, GPT 5.1) 

qm9 (Hard)
- https://www.nature.com/articles/sdata201422
# Grok Code (sometimes)

Hopv15 (Hard)
- https://chemrxiv.org/engage/chemrxiv/article-details/64be471cb605c6803b425da6
# Download: GPT 5.1 Codex Mini, Processing: not working (tried, Grok Code, GPT 5.1 Mini Codex)