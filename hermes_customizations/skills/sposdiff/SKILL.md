---
name: sposdiff
description: Analyze Google and Yandex SEO position exports found in this project's files directory.
---

# SEO position difference analysis

The `/sposdiff` command accepts no arguments.

Before running anything, inspect the open project's `files\` directory. Count
only regular files with extensions CSV, TSV, TXT, XLS, or XLSX; ignore hidden
files such as `.gitkeep`.

Stop immediately and return only an error if:

- no such files exist;
- more than two exist;
- a file basename is not exactly `google` or `yandex`; or
- there are two files for the same engine.

If one valid file or both valid files exist, run exactly:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" analyze
```

Report the resulting JSON. If it succeeds, use the native `read_file` tool to
read only `insights.md` in the output directory. Never use a terminal command
to read this file: the terminal can corrupt UTF-8 Cyrillic text.

Use the cloud DeepSeek model only now. Write a fluent, professional final
answer entirely in Russian. Do not translate word by word, reproduce raw tool
output, print metric identifiers, or mention this workflow, DeepSeek, files,
CSV, or column names. Explain the local evidence using this exact order:

1. A two-sentence verdict: uniform, cluster-specific, or mixed.
2. The largest global CTR-visibility loss first. Never replace this with a
   cluster anomaly: global impact and cluster anomaly answer different questions.
3. The three strongest loss clusters: category, human-readable transition,
   actual rate, global rate, and the reason the cluster stands out. State
   whether a cluster is a large contributor to the global loss or a local
   outlier, or both.
4. The three strongest growth clusters, if present.
5. A concise list of manual checks. Treat a disappearance as a separate event.

Do not recalculate metrics, invent causes, use frequency to prioritize a
result, send source exports or any additional CSV files to the cloud, or
continue after an error. Do not download or look up embedding models: this
workflow loads only `..\models\multilingual-e5-small\` in offline mode.
