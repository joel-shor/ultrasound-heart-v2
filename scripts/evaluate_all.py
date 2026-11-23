#!/usr/bin/env python3
"""Run multi-model fusion + Evaluation 1 (Binary) + Evaluation 2 (3-State Triage) & archive to GCS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heartsounds.eval.ensemble import build_all_ensembles_and_ablations  # noqa: E402
from heartsounds.eval.metrics import evaluate_binary_suite  # noqa: E402
from heartsounds.eval.plots import (  # noqa: E402
    plot_confusion_and_subtypes,
    plot_roc_and_pr_frontiers,
    plot_triage_coverage_tradeoff,
)
from heartsounds.eval.triage import evaluate_triage_suite  # noqa: E402
from heartsounds.storage.gcs import (  # noqa: E402
    CURATED_GCS_ROOT,
    DEFAULT_GCP_PROJECT,
    upload_directory_to_gcs,
)


def _discover_model_runs(
    runs_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    oof_map: dict[str, pd.DataFrame] = {}
    tst_map: dict[str, pd.DataFrame] = {}
    quar_map: dict[str, pd.DataFrame] = {}

    candidates = [
        ("logreg_dsp", runs_root / "family1_dsp" / "logreg_dsp"),
        ("xgboost_dsp", runs_root / "family1_dsp" / "xgboost_dsp"),
        ("lightgbm_dsp", runs_root / "family1_dsp" / "lightgbm_dsp"),
        ("gbdt_dsp_blend", runs_root / "family1_dsp" / "gbdt_dsp_blend"),
        ("waveform_mil", runs_root / "waveform_mil_5fold"),
        ("spec_tempogram_mil", runs_root / "spec_tempogram_mil_5fold"),
    ]
    for name, folder in candidates:
        oof_pq = folder / "oof_predictions.parquet"
        tst_pq = folder / "test_predictions.parquet"
        if oof_pq.is_file() and tst_pq.is_file():
            oof_map[name] = pd.read_parquet(oof_pq)
            tst_map[name] = pd.read_parquet(tst_pq)
            q_pq = folder / "quarantine_predictions.parquet"
            if q_pq.is_file():
                quar_map[name] = pd.read_parquet(q_pq)
    return oof_map, tst_map, quar_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate standalone models and fused ensembles.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "v2_curated")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "outputs" / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "evaluation")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--upload-gcs", action="store_true")
    parser.add_argument("--gcs-dest", default=f"{CURATED_GCS_ROOT}/predictions")
    parser.add_argument("--gcp-project", default=DEFAULT_GCP_PROJECT)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_parquet(args.data_dir / "manifest.parquet")
    dsp_df = pd.read_parquet(args.data_dir / "dsp_features.parquet")
    quality_by_id = dsp_df.set_index("recording_id")["quality_r_peak_mean"].to_dict()
    uri_by_id = manifest_df.set_index("recording_id")["primary_raw_gcs_uri"].to_dict()

    oof_all, tst_all, quar_all = _discover_model_runs(args.runs_dir)
    if not oof_all:
        raise RuntimeError(f"No completed model runs discovered under {args.runs_dir}")
    print(f"[eval] Discovered standalone models: {list(oof_all.keys())}", flush=True)

    # Primary representative from each distinct model family for clean multi-family fusion
    primary_keys = [
        k for k in ("gbdt_dsp_blend", "waveform_mil", "spec_tempogram_mil") if k in oof_all
    ]
    primary_oof = {k: oof_all[k] for k in primary_keys}
    primary_tst = {k: tst_all[k] for k in primary_keys}

    ensembles, quar_blend = build_all_ensembles_and_ablations(
        primary_oof, primary_tst, quarantine_tables=quar_all
    )
    for ens_name, ens_out in ensembles.items():
        oof_all[ens_name] = ens_out.oof_df
        tst_all[ens_name] = ens_out.test_df

    eval1_reports: dict[str, Any] = {}
    eval2_reports: dict[str, Any] = {}
    triage_oof_cols: dict[str, np.ndarray] = {}
    triage_tst_cols: dict[str, np.ndarray] = {}

    for m_name in oof_all:
        odf = oof_all[m_name].sort_values("recording_id").reset_index(drop=True)
        tdf = tst_all[m_name].sort_values("recording_id").reset_index(drop=True)
        oof_all[m_name] = odf
        tst_all[m_name] = tdf

        ev1 = evaluate_binary_suite(odf, tdf, n_boot=args.n_boot)
        eval1_reports[m_name] = ev1

        center_th = ev1["oof_operating_points"]["max_youden_j"]["threshold"]
        q_oof = odf["recording_id"].map(quality_by_id).to_numpy(dtype=float)
        q_tst = tdf["recording_id"].map(quality_by_id).to_numpy(dtype=float)

        q_df = (
            quar_blend
            if (m_name == "fusion_3family_logit_blend" and quar_blend is not None)
            else quar_all.get(m_name)
        )
        q_qual = (
            q_df["recording_id"].map(quality_by_id).to_numpy(dtype=float)
            if q_df is not None
            else None
        )
        ev2, st_oof, st_tst = evaluate_triage_suite(
            odf,
            tdf,
            center_threshold=center_th,
            oof_quality=q_oof,
            test_quality=q_tst,
            quarantine_df=q_df,
            quarantine_quality=q_qual,
        )
        eval2_reports[m_name] = ev2
        triage_oof_cols[m_name] = st_oof
        triage_tst_cols[m_name] = st_tst

    # Choose champion model strictly by OOF 0.5*ROC_AUC + 0.5*PR_AUC
    champion = max(
        eval1_reports.keys(),
        key=lambda k: (
            0.5 * eval1_reports[k]["oof_global"]["roc_auc"]
            + 0.5 * eval1_reports[k]["oof_global"]["pr_auc"]
        ),
    )
    print(f"[eval] Selected OOF champion model: {champion}", flush=True)

    full_report = {
        "champion_model_by_oof": champion,
        "ensemble_weights": {k: v.weights for k, v in ensembles.items()},
        "evaluation_1_binary_100pct_coverage": eval1_reports,
        "evaluation_2_clinical_3state_triage": eval2_reports,
    }
    (args.out_dir / "evaluation_report.json").write_text(
        json.dumps(full_report, indent=2) + "\n", encoding="utf-8"
    )

    # Export inspectable master prediction spreadsheets linking back to gs://data-raw-dualregion/
    for split_tag, table_dict, triage_dict in (
        ("oof", oof_all, triage_oof_cols),
        ("test", tst_all, triage_tst_cols),
    ):
        ref = table_dict[champion].copy()
        master = pd.DataFrame(
            {
                "recording_id": ref["recording_id"],
                "primary_raw_gcs_uri": ref["recording_id"].map(uri_by_id),
                "split": ref["split"],
                "patient_group": ref["patient_group"],
                "y_true": ref["y_true"],
                "curated_subtype": ref["curated_subtype"],
                "dsp_quality_r_peak": ref["recording_id"].map(quality_by_id).round(5),
            }
        )
        for m_name, m_df in table_dict.items():
            master[f"prob_{m_name}"] = m_df["y_prob"].round(6)
        champ_th = eval1_reports[champion]["oof_operating_points"]["max_youden_j"]["threshold"]
        master["champion_binary_pred_max_youden_j"] = (
            master[f"prob_{champion}"] >= champ_th
        ).astype(int)
        master["champion_triage_10pct_yellow"] = triage_dict[champion]

        master.to_csv(args.out_dir / f"{split_tag}_predictions.csv", index=False)
        master.to_parquet(args.out_dir / f"{split_tag}_predictions.parquet", index=False)

    # Generate diagnostic plots
    plot_keys = [
        k
        for k in (*primary_keys, "fusion_3family_logit_blend", "fusion_3family_stacker")
        if k in tst_all
    ]
    plot_roc_and_pr_frontiers(
        {k: tst_all[k] for k in plot_keys}, args.out_dir / "eval1_roc_pr_frontiers.png"
    )
    plot_triage_coverage_tradeoff(
        eval2_reports[champion], champion, args.out_dir / "eval2_triage_coverage_tradeoff.png"
    )
    champ_th = eval1_reports[champion]["oof_operating_points"]["max_youden_j"]["threshold"]
    plot_confusion_and_subtypes(
        tst_all[champion],
        champ_th,
        champion,
        args.out_dir / "eval1_champion_confusion_and_subtypes.png",
    )

    if args.upload_gcs:
        uploaded = upload_directory_to_gcs(args.out_dir, args.gcs_dest, project=args.gcp_project)
        print(f"[eval] Uploaded {len(uploaded)} evaluation files to {args.gcs_dest}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
