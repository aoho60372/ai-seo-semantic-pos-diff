# Position Dynamics Workflow

Work only in this project directory. This workflow is independent from
any neighboring SEO workflow and must not read or alter its files.

The only accepted source files are `files\google.csv` and/or `files\yandex.csv`
(CSV, TSV, XLS, and XLSX are supported). Each table must contain `query`,
`freq`, `pos_cur`, and `pos_prev`. `0` means the phrase is absent from the
tracked organic results.

For files with 16 or more phrases, require a complete local model in
`..\models\multilingual-e5-small\`. The workflow is offline-first:
never enable an online fallback or download model files during a task.

For `/sposdiff`, first inspect `files\`. Stop with an error if no supported
source file is present or if more than two are present. Stop with an error if a
source name is not exactly `google` or `yandex` before its extension. Otherwise
run exactly:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" analyze
```

Read and report only the resulting JSON and `outputs\<run>\insights.md`.
Never report `pos_prev > 0, pos_cur = 0` as an ordinary numeric decline: it is
a disappearance. A `0 -> positive` event is an appearance. `freq` is reported
only as a cluster sum; it must not alter deltas, verdicts, ranking, or claims.
Do not infer traffic, revenue, indexing status, penalties, or causes.
