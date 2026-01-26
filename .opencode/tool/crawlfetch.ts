import { tool } from "@opencode-ai/plugin";

function uniqKeepOrder(items: string[]) {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const x of items) {
    const k = x.trim();
    if (!k) continue;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  return out;
}

function extractUrls(text: string) {
  // URL extractor for markdown/html-ish text
  const re = /\bhttps?:\/\/[^\s<>()\]]+/g;
  const matches = text.match(re) ?? [];
  // strip trailing punctuation that often sticks in markdown
  return uniqKeepOrder(matches.map(u => u.replace(/[),.;\]]+$/g, "")));
}

function scoreUrl(u: string) {
  const s = u.toLowerCase();

  // Repository / dataset hosts
  const hostBoost =
    (s.includes("figshare.com") ? 50 : 0) +
    (s.includes("zenodo.org") ? 50 : 0) +
    (s.includes("osf.io") ? 35 : 0) +
    (s.includes("dataverse") ? 30 : 0) +
    (s.includes("kaggle.com") ? 25 : 0) +
    (s.includes("github.com") ? 20 : 0) +
    (s.includes("huggingface.co") ? 20 : 0) +
    (s.includes("drive.google.com") ? 15 : 0) +
    (s.includes("dropbox.com") ? 15 : 0);

  // Strong “this is the data” signals
  const intentBoost =
    (s.includes("supplement") || s.includes("supporting") || s.includes("si") ? 18 : 0) +
    (s.includes("download") ? 16 : 0) +
    (s.includes("dataset") || s.includes("data-set") ? 14 : 0) +
    (s.includes("files") ? 8 : 0) +
    (s.includes(".zip") || s.includes(".tar") || s.includes(".gz") || s.includes(".7z") ? 12 : 0) +
    (s.includes(".csv") || s.includes(".tsv") || s.includes(".json") || s.includes(".sdf") || s.includes(".mol") ? 10 : 0) +
    (s.includes(".xlsx") ? 6 : 0) +
    (s.includes(".pdf") ? 2 : 0);

  // Mild penalty for “not data”
  const penalty =
    (s.includes("twitter.com") || s.includes("x.com") ? 40 : 0) +
    (s.includes("facebook.com") ? 40 : 0) +
    (s.includes("linkedin.com") ? 30 : 0) +
    (s.includes("mailto:") ? 100 : 0) +
    (s.includes("javascript:") ? 100 : 0);

  return hostBoost + intentBoost - penalty;
}

function evidenceSnippets(md: string) {
  const needles = [
    "data availability",
    "availability of data",
    "code availability",
    "supplementary",
    "supporting information",
    "supplemental",
    "dataset",
    "download",
    "figshare",
    "zenodo",
    "osf",
    "dataverse",
    "github",
  ];

  const lines = md.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const low = line.toLowerCase();
    if (needles.some(n => low.includes(n))) {
      // include tiny context window
      const prev = lines[i - 1] ?? "";
      const next = lines[i + 1] ?? "";
      const block = [prev, line, next].filter(Boolean).join("\n");
      out.push(block.trim());
      if (out.length >= 12) break;
    }
  }
  return uniqKeepOrder(out);
}

export default tool({
  description:
    "Fetch a URL using Crawl4AI (JS-rendered), then return ranked candidate dataset links + evidence snippets + truncated markdown.",

  args: {
    url: tool.schema.string().describe("URL to crawl"),
    maxChars: tool.schema.number().optional().describe("Max characters of markdown to include (default 20000)"),
    maxLinks: tool.schema.number().optional().describe("Max candidate links to return (default 40)"),
  },

  async execute({ url, maxChars, maxLinks }) {
    const limit = maxChars ?? 20000;
    const linkLimit = maxLinks ?? 40;

    // Crawl4AI markdown
    const mdFull = await Bun.$`crwl crawl ${url} -o markdown`.text();

    const urls = extractUrls(mdFull);
    const ranked = urls
      .map(u => ({ u, score: scoreUrl(u) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, linkLimit);

    const evid = evidenceSnippets(mdFull);

    const md = mdFull.length > limit
      ? mdFull.slice(0, limit) + `\n\n---\n[truncated] Returned first ${limit} chars of markdown.`
      : mdFull;

    const topLinksText = ranked.length
      ? ranked.map(x => `- (${x.score}) ${x.u}`).join("\n")
      : "(none found)";

    const evidText = evid.length
      ? evid.map((b, i) => `### Evidence ${i + 1}\n${b}`).join("\n\n")
      : "(none found)";

    return [
      `# CrawlFetch result`,
      `URL: ${url}`,
      ``,
      `## Top candidate links`,
      topLinksText,
      ``,
      `## Evidence snippets`,
      evidText,
      ``,
      `## Page markdown (truncated)`,
      md,
    ].join("\n");
  },
});
