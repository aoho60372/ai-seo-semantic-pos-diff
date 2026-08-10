"""Offline-first analysis of separate Google and Yandex position exports."""
from __future__ import annotations

import argparse, json, os, re, shutil, sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

REQUIRED = ("query", "freq", "pos_cur", "pos_prev")
ENGINES = ("google", "yandex")
EXTENSIONS = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "multilingual-e5-small"
DEFAULT_CTR_CURVE = PROJECT_DIR / "configs" / "ctr_curve.json"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    error = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp866"):
        try: return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc: error = exc
    raise ValueError(f"Cannot read {path.name}: {error}")


def input_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir(): raise ValueError(f"Files directory does not exist: {directory}")
    files = sorted(x for x in directory.iterdir() if x.is_file() and not x.name.startswith(".") and x.suffix.lower() in EXTENSIONS)
    if not files: raise ValueError("No input files found in files/. Expected google.* and/or yandex.*.")
    if len(files) > 2: raise ValueError("More than two input files found: " + ", ".join(x.name for x in files))
    result: dict[str, Path] = {}
    for file in files:
        engine = file.stem.lower()
        if engine not in ENGINES: raise ValueError(f"Unsupported input name {file.name}. Use google.* or yandex.*.")
        if engine in result: raise ValueError(f"More than one input file supplied for {engine}.")
        result[engine] = file
    return result


def validate(raw: pd.DataFrame, max_position: int) -> pd.DataFrame:
    frame = raw.copy(); frame.columns = [str(x).strip().lower() for x in frame.columns]
    missing = [x for x in REQUIRED if x not in frame]
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = frame.loc[:, REQUIRED].copy(); frame["query"] = frame["query"].astype("string").fillna("").str.strip()
    if (frame["query"] == "").any(): raise ValueError("Column query contains empty values.")
    if frame["query"].duplicated().any(): raise ValueError("Duplicate queries found; resolve them before analysis.")
    frame["freq"] = pd.to_numeric(frame["freq"], errors="coerce")
    if frame["freq"].isna().any() or (frame["freq"] < 0).any(): raise ValueError("Column freq must be non-negative numeric values.")
    for column in ("pos_cur", "pos_prev"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any() or (values < 0).any() or (values > max_position).any():
            raise ValueError(f"Column {column} must be integer positions from 0 to {max_position}.")
        frame[column] = values.astype("int64")
    return frame


def load_local_model(model_dir: Path, device: str):
    if not model_dir.is_dir():
        raise ValueError(f"Local embedding model is missing: {model_dir}. Download it before analysis; online model lookup is disabled.")
    required = ("config.json", "modules.json", "model.safetensors")
    absent = [name for name in required if not (model_dir / name).is_file()]
    if absent: raise ValueError(f"Local model is incomplete ({', '.join(absent)} missing): {model_dir}")
    import torch
    from sentence_transformers import SentenceTransformer
    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if chosen == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA was requested but is unavailable.")
    return SentenceTransformer(str(model_dir), device=chosen, local_files_only=True), chosen


def embeddings(frame: pd.DataFrame, model, work_dir: Path, batch_size: int) -> np.memmap:
    dim = model.get_sentence_embedding_dimension(); path = work_dir / "embeddings.f32"
    matrix = np.memmap(path, dtype="float32", mode="w+", shape=(len(frame), dim))
    texts = frame["query"].astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batch = ["query: " + text for text in texts[start:start + batch_size]]
        matrix[start:start + len(batch)] = model.encode(batch, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        matrix.flush()
    return matrix


def cluster(frame: pd.DataFrame, model, used_device: str, work_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if len(frame) < args.min_cluster_size * 2: return np.array(["All queries"] * len(frame), object), "single cluster: dataset is small"
    matrix = embeddings(frame, model, work_dir, args.embedding_batch_size)
    count = max(2, min(args.max_clusters, int(np.ceil(len(frame) / args.target_cluster_size))))
    model = MiniBatchKMeans(n_clusters=count, random_state=42, n_init=3, max_iter=100, batch_size=min(4096, len(frame)))
    labels = model.fit_predict(matrix)
    del matrix
    return np.array([f"Cluster {x + 1:03d}" for x in labels], object), f"local E5 embeddings on {used_device}; MiniBatchKMeans ({count} clusters)"


def names(frame: pd.DataFrame, codes: np.ndarray) -> pd.Series:
    result = {}
    for code in sorted(set(codes)):
        if code == "All queries": result[code] = code; continue
        counts: dict[str, int] = {}
        for query in frame.loc[codes == code, "query"].astype(str):
            for token in re.findall(r"[\w-]{3,}", query.lower()): counts[token] = counts.get(token, 0) + 1
        top = [x for x, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]
        result[code] = f"{code}: {' / '.join(top)}"
    return pd.Series([result[x] for x in codes], index=frame.index)


def load_ctr_curve(path: Path, engine: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    curve = data.get(engine, data["default"])
    required = [str(position) for position in range(1, 11)] + ["11_plus", "0"]
    missing = [key for key in required if key not in curve]
    if missing: raise ValueError(f"CTR curve is missing: {', '.join(missing)}")
    return {key: float(value) for key, value in curve.items()}


def ctr(position: int, curve: dict[str, float]) -> float:
    return curve.get(str(position), curve["11_plus"]) if position > 0 else curve["0"]


def decorate(frame: pd.DataFrame, curve: dict[str, float]) -> pd.DataFrame:
    result = frame.copy(); previous, current = result.pos_prev, result.pos_cur; same = (previous > 0) & (current > 0)
    result["event"] = np.select([(previous > 0) & (current == 0), (previous == 0) & (current > 0), same & (current < previous), same & (current > previous)], ["disappeared", "appeared", "improved", "worsened"], default="stable")
    result["delta"] = np.where(same, previous - current, np.nan)
    result["ctr_prev"] = previous.map(lambda position: ctr(position, curve))
    result["ctr_cur"] = current.map(lambda position: ctr(position, curve))
    result["relative_ctr_change"] = result.ctr_cur - result.ctr_prev
    result["prev_band"] = result.pos_prev.map(position_band); result["cur_band"] = result.pos_cur.map(position_band)
    return result


def position_band(position: int) -> str:
    if position == 0: return "0 (absent)"
    if position == 1: return "1"
    if position <= 3: return "2-3"
    if position <= 5: return "4-5"
    if position <= 10: return "6-10"
    return "11+"


def flows(frame: pd.DataFrame) -> pd.DataFrame:
    order = ["1", "2-3", "4-5", "6-10", "11+", "0 (absent)"]
    table = pd.crosstab(frame.prev_band, frame.cur_band).reindex(index=order, columns=order, fill_value=0)
    return table.rename_axis("previous_position_band").reset_index()


def global_impact(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    change = frame.relative_ctr_change
    affected = change < 0 if direction == "loss" else change > 0
    if not affected.any(): return pd.DataFrame(columns=["prev_band", "cur_band", "phrases", "relative_ctr_impact", "share_of_direction_pct"])
    grouped = frame.loc[affected].assign(relative_ctr_impact=change.loc[affected].abs()).groupby(["prev_band", "cur_band"], as_index=False).agg(phrases=("query", "size"), relative_ctr_impact=("relative_ctr_impact", "sum"))
    grouped["share_of_direction_pct"] = (grouped.relative_ctr_impact / grouped.relative_ctr_impact.sum() * 100).round(2)
    return grouped.sort_values("relative_ctr_impact", ascending=False)


def cluster_contribution(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    change = frame.relative_ctr_change
    affected = change < 0 if direction == "loss" else change > 0
    if not affected.any(): return pd.DataFrame(columns=["prev_band", "cur_band", "cluster", "cluster_freq_sum", "phrases", "relative_ctr_impact", "transition_contribution_pct"])
    cluster_freq = frame.groupby("cluster").freq.sum()
    result = frame.loc[affected].assign(relative_ctr_impact=change.loc[affected].abs()).groupby(["prev_band", "cur_band", "cluster"], as_index=False).agg(phrases=("query", "size"), relative_ctr_impact=("relative_ctr_impact", "sum"))
    result["cluster_freq_sum"] = result.cluster.map(cluster_freq).astype("int64")
    result = result[["prev_band", "cur_band", "cluster", "cluster_freq_sum", "phrases", "relative_ctr_impact"]]
    totals = result.groupby(["prev_band", "cur_band"]).relative_ctr_impact.transform("sum")
    result["transition_contribution_pct"] = (result.relative_ctr_impact / totals * 100).round(2)
    return result.sort_values(["relative_ctr_impact", "phrases"], ascending=False)


def anomaly_table(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    previous, current = frame.pos_prev, frame.pos_cur
    source_zones = [("1", previous == 1, 100), ("2_3", previous.between(2, 3), 75), ("4_5", previous.between(4, 5), 50), ("6_10", previous.between(6, 10), 30)]
    target_zones = [("2_3", current.between(2, 3), 10), ("4_5", current.between(4, 5), 20), ("6_10", current.between(6, 10), 40), ("11_plus", (current == 0) | (current >= 11), 100)]
    definitions = []
    if direction == "loss":
        source_index = {"1": 0, "2_3": 1, "4_5": 2, "6_10": 3}
        target_index = {"2_3": 1, "4_5": 2, "6_10": 3, "11_plus": 4}
        for source, eligible, source_weight in source_zones:
            for target, affected, target_weight in target_zones:
                if target_index[target] > source_index[source]: definitions.append((f"{source}_to_{target}", source, target, eligible, affected, source_weight, target_weight))
    else:
        growth_sources = [("11_plus", (previous == 0) | (previous >= 11)), ("6_10", previous.between(6, 10)), ("4_5", previous.between(4, 5)), ("2_3", previous.between(2, 3))]
        growth_targets = [("6_10", current.between(6, 10), 40), ("4_5", current.between(4, 5), 50), ("2_3", current.between(2, 3), 75), ("1", current == 1, 100)]
        source_index = {"11_plus": 4, "6_10": 3, "4_5": 2, "2_3": 1}; target_index = {"6_10": 3, "4_5": 2, "2_3": 1, "1": 0}
        for source, eligible in growth_sources:
            for target, affected, target_weight in growth_targets:
                if target_index[target] < source_index[source]: definitions.append((f"{source}_to_{target}", source, target, eligible, affected, 1, target_weight))
    rows = []
    for metric, source, target, eligible, affected, source_weight, target_weight in definitions:
        overall_eligible, overall_affected = int(eligible.sum()), int((eligible & affected).sum())
        if not overall_eligible: continue
        baseline = overall_affected / overall_eligible
        for cluster, group in frame.groupby("cluster", sort=False):
            mask = group.index
            exposure = int(eligible.loc[mask].sum()); actual = int((eligible.loc[mask] & affected.loc[mask]).sum())
            if not exposure: continue
            rate = actual / exposure; expected = exposure * baseline; excess = actual - expected
            flagged = exposure >= 30 and actual >= 5 and rate - baseline >= .05 and rate >= baseline * 1.5
            dropouts = int((eligible.loc[mask] & affected.loc[mask] & (current.loc[mask] == 0)).sum()) if direction == "loss" else 0
            ctr_impact = float((group.loc[eligible.loc[mask] & affected.loc[mask], "relative_ctr_change"].abs()).sum())
            global_ctr_per_eligible = float((frame.loc[eligible & affected, "relative_ctr_change"].abs()).sum()) / overall_eligible
            excess_ctr = ctr_impact - exposure * global_ctr_per_eligible
            rows.append({"cluster":cluster,"cluster_freq_sum":int(group.freq.sum()),"metric":metric,"source_band":source,"target_band":target,"phrases":len(group),"eligible_phrases":exposure,"affected_phrases":actual,"affected_rate_pct":round(rate*100,2),"global_rate_pct":round(baseline*100,2),"rate_lift":round(rate / baseline,2) if baseline else np.nan,"excess_phrases":round(excess,1),"relative_ctr_impact":round(ctr_impact,4),"excess_relative_ctr_impact":round(excess_ctr,4),"priority_score":round(max(excess_ctr, 0),4),"dropouts_within_transition":dropouts,"problematic": "yes" if flagged else "no"})
    return pd.DataFrame(rows).sort_values(["problematic", "priority_score", "rate_lift"], ascending=[False, False, False])


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby("cluster", sort=False):
        comparable = group.delta.dropna(); row = {"cluster": name, "phrases":len(group), "freq_sum":group.freq.sum(), "comparable_phrases":len(comparable), "median_delta":comparable.median() if len(comparable) else np.nan}
        for event in ("improved", "worsened", "stable", "appeared", "disappeared"):
            row[event] = int((group.event == event).sum()); row[event + "_share_pct"] = round(row[event] / len(group) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["disappeared", "phrases"], ascending=[False, False])


def insight(engine: str, frame: pd.DataFrame, losses: pd.DataFrame, growth: pd.DataFrame, global_losses: pd.DataFrame, global_growth: pd.DataFrame) -> tuple[dict, str]:
    gone, appeared = int((frame.event == "disappeared").sum()), int((frame.event == "appeared").sum())
    loss_hits, growth_hits = losses[losses.problematic == "yes"], growth[growth.problematic == "yes"]
    verdict = "смешанная картина: выделены кластеры с аномальными изменениями" if len(loss_hits) or len(growth_hits) else "изменения похожи на равномерное движение по кластерам"
    lines = [f"## {engine.title()}", f"- Всего фраз: {len(frame)}; пропало: {gone}; появилось: {appeared}.", f"- Вердикт: {verdict}.", "- Главные общие потери кликовой видимости:"]
    for _, row in global_losses.head(3).iterrows(): lines.append(f"  - из {row.prev_band} в {row.cur_band}: {int(row.phrases)} фраз; относительная потеря CTR {row.relative_ctr_impact:.2f} п.п. ({row.share_of_direction_pct}% всех потерь).")
    lines.append("- Проблемные падения по кластерам:")
    for _, row in loss_hits.head(8).iterrows(): lines.append(f"  - {row.cluster}: {transition_text(row.metric)}, затронуто {int(row.affected_phrases)} из {int(row.eligible_phrases)} ({row.affected_rate_pct}%; фон {row.global_rate_pct}%; превышение {row.rate_lift}x; полных выпадений {int(row.dropouts_within_transition)}).")
    if not len(loss_hits): lines.append("  - Не выделено.")
    lines.append("- Главный общий рост кликовой видимости:")
    for _, row in global_growth.head(3).iterrows(): lines.append(f"  - из {row.prev_band} в {row.cur_band}: {int(row.phrases)} фраз; относительный рост CTR {row.relative_ctr_impact:.2f} п.п. ({row.share_of_direction_pct}% всего роста).")
    lines.append("- Аномальный рост по кластерам:")
    for _, row in growth_hits.head(8).iterrows(): lines.append(f"  - {row.cluster}: {transition_text(row.metric)}, затронуто {int(row.affected_phrases)} из {int(row.eligible_phrases)} ({row.affected_rate_pct}%; фон {row.global_rate_pct}%; превышение {row.rate_lift}x).")
    if not len(growth_hits): lines.append("  - Не выделено.")
    return {"search_engine":engine,"phrases":len(frame),"freq_sum":frame.freq.sum(),"appeared":appeared,"disappeared":gone,"loss_anomalies":len(loss_hits),"growth_anomalies":len(growth_hits),"verdict":verdict}, "\n".join(lines)


def transition_text(metric: str) -> str:
    source_labels = {"1": "1-го места", "2_3": "позиций 2-3", "4_5": "позиций 4-5", "6_10": "позиций 6-10", "11_plus": "позиций ниже топ-10"}
    target_labels = {"1": "1-е место", "2_3": "позиции 2-3", "4_5": "позиции 4-5", "6_10": "позиции 6-10", "11_plus": "позиции ниже топ-10"}
    source, target = metric.split("_to_", 1)
    return f"переход с {source_labels[source]} на {target_labels[target]}"


def analyze(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parent; sources = input_files(Path(args.files_dir).resolve() if args.files_dir else root / "files")
    prepared = {engine: validate(read_table(path), args.max_position) for engine, path in sources.items()}
    local_model = used_device = None
    if any(len(frame) >= args.min_cluster_size * 2 for frame in prepared.values()): local_model, used_device = load_local_model(Path(args.model_dir), args.device)
    output = Path(args.output_dir).resolve() if args.output_dir else root / "outputs" / f"position-diff-{datetime.now():%Y%m%d-%H%M%S}"; output.mkdir(parents=True, exist_ok=False)
    rows, texts, results, methods = [], ["# Инсайты по динамике позиций", ""], {}, {}
    try:
        for engine in ENGINES:
            if engine not in prepared: continue
            work = output / ".work" / engine; work.mkdir(parents=True)
            curve = load_ctr_curve(Path(args.ctr_curve), engine)
            frame = prepared[engine]; codes, method = cluster(frame, local_model, used_device or "n/a", work, args); frame["cluster_code"] = codes; frame["cluster"] = names(frame, codes); frame["cluster_freq_sum"] = frame.groupby("cluster")["freq"].transform("sum").astype("int64"); frame = decorate(frame, curve); summary = summarize(frame); losses = anomaly_table(frame, "loss"); growth = anomaly_table(frame, "growth"); flow = flows(frame); global_losses = global_impact(frame, "loss"); global_growth = global_impact(frame, "growth"); loss_contributions = cluster_contribution(frame, "loss"); growth_contributions = cluster_contribution(frame, "growth"); row, text = insight(engine, frame, losses, growth, global_losses, global_growth)
            rows.append(row); texts += [text, ""]; results[engine] = (frame, summary, losses, growth, flow, global_losses, global_growth, loss_contributions, growth_contributions); methods[engine] = method
        metadata = {"generated_at":datetime.now().isoformat(timespec="seconds"),"inputs":{x:str(y) for x,y in sources.items()},"model_dir":str(Path(args.model_dir)),"ctr_curve":str(Path(args.ctr_curve)),"network":"disabled; local_files_only=true","clustering":methods,"frequency_policy":"freq is reported only as cluster sum"}
        with pd.ExcelWriter(output / "position_dynamics.xlsx", engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Executive summary", index=False)
            for engine, (frame, summary, losses, growth, flow, global_losses, global_growth, loss_contributions, growth_contributions) in results.items():
                flow.to_excel(writer, sheet_name=f"{engine.title()} flows", index=False); global_losses.to_excel(writer, sheet_name=f"{engine.title()} global loss", index=False); global_growth.to_excel(writer, sheet_name=f"{engine.title()} global growth", index=False); loss_contributions.to_excel(writer, sheet_name=f"{engine.title()} loss contributions", index=False); growth_contributions.to_excel(writer, sheet_name=f"{engine.title()} growth contributions", index=False); losses.to_excel(writer, sheet_name=f"{engine.title()} loss anomalies", index=False); growth.to_excel(writer, sheet_name=f"{engine.title()} growth anomalies", index=False); summary.to_excel(writer, sheet_name=f"{engine.title()} clusters", index=False); frame.to_excel(writer, sheet_name=f"{engine.title()} phrases", index=False)
        for engine, (frame, summary, losses, growth, flow, global_losses, global_growth, loss_contributions, growth_contributions) in results.items():
            summary.to_csv(output / f"{engine}_cluster_summary.csv", index=False, encoding="utf-8-sig"); losses.to_csv(output / f"{engine}_loss_anomalies.csv", index=False, encoding="utf-8-sig"); growth.to_csv(output / f"{engine}_growth_anomalies.csv", index=False, encoding="utf-8-sig"); flow.to_csv(output / f"{engine}_position_flows.csv", index=False, encoding="utf-8-sig"); global_losses.to_csv(output / f"{engine}_global_loss.csv", index=False, encoding="utf-8-sig"); global_growth.to_csv(output / f"{engine}_global_growth.csv", index=False, encoding="utf-8-sig"); loss_contributions.to_csv(output / f"{engine}_loss_contributions.csv", index=False, encoding="utf-8-sig"); growth_contributions.to_csv(output / f"{engine}_growth_contributions.csv", index=False, encoding="utf-8-sig"); frame.to_csv(output / f"{engine}_phrases_enriched.csv", index=False, encoding="utf-8-sig")
        (output / "insights.md").write_text("\n".join(texts) + "\nЧастотность приведена только справочно и не влияет на выводы.\n", encoding="utf-8"); (output / "run.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"status":"complete","output":str(output),"workbook":str(output / "position_dynamics.xlsx"),"engines":list(results),"clustering":methods},ensure_ascii=False)); return output
    finally: shutil.rmtree(output / ".work", ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); cmd = sub.add_parser("analyze")
    cmd.add_argument("--files-dir"); cmd.add_argument("--output-dir"); cmd.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR)); cmd.add_argument("--ctr-curve", default=str(DEFAULT_CTR_CURVE)); cmd.add_argument("--max-position",type=int,default=100); cmd.add_argument("--min-cluster-size",type=int,default=8); cmd.add_argument("--embedding-batch-size",type=int,default=256); cmd.add_argument("--target-cluster-size",type=int,default=5000); cmd.add_argument("--max-clusters",type=int,default=80); cmd.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    args = parser.parse_args()
    try: analyze(args)
    except Exception as exc: print(json.dumps({"status":"error","error":str(exc)},ensure_ascii=False),file=sys.stderr); return 2
    return 0

if __name__ == "__main__": raise SystemExit(main())
