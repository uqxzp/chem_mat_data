Crawlfetch: https://github.com/unclecode/crawl4ai
MCP File Downloader: github.com/abs222222/mcp-file-downloader
Websearch: https://github.com/ghoulr/opencode-websearch-cited

Add to config file:

    "mcp": {
        "file_downloader": {
            "type": "local",
            "command": ["node", "<absolute path>/chem_mat_data/.mcp-file-downloader/download-server.js"],
            "enabled": true,
            "timeout": 15000
        }
    },
    "provider": {
        "openai": {
            "options": {
                "websearch_cited": {
                    "model": "gpt-5.2"
                }
            }
        }
    },
    "plugin": [
        "opencode-websearch-cited@1.2.0"
    ],
    "agent": {
        "dataset-links": {
            "description": "Find dataset download links, then return JSON only.",
            "steps": 24,
            "permission": {
                "websearch_*": "allow",
                "crawlfetch": "allow",
                "webfetch": "deny",
                "file_downloader_*": "allow",
                "bash": "deny",
                "read": "deny",
                "glob": "deny",
                "grep": "deny",
                "list": "deny",
                "edit": "deny",
                "patch": "deny",
                "codesearch": "deny",
                "skill": "deny",
                "todowrite": "deny",
                "todoread": "deny",
                "question": "deny",
                "task": {
                    "*": "deny"
                }
            }
        }
    }

How to use:

0. opencode serve --port 54321 --print-logs
1. cmmanage agent process <link>
2. cmmanage agent generate
3. python chem_mat_data/agent/artifacts/scripts/<name>_generated.py 