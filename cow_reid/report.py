from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .utils import bool_mask


STYLE = """
body { font-family: Inter, system-ui, sans-serif; margin: 28px; color: #17211b; background: #f4f7f5; }
h1, h2 { color: #173c2d; }
.notice { padding: 14px 16px; background: #fff3cd; border-left: 5px solid #d59b00; margin: 18px 0; }
.cluster { background: white; border: 1px solid #d8e2dc; border-radius: 12px; padding: 16px; margin: 16px 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 12px; }
.pair { background: white; border: 1px solid #d8e2dc; border-radius: 12px; padding: 14px; margin: 14px 0; }
.pair-images { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }
.card { border: 1px solid #dfe8e2; border-radius: 8px; padding: 10px; background: #fbfdfb; }
.card img { width: 100%; height: auto; border-radius: 6px; }
.meta { font-family: ui-monospace, monospace; font-size: 12px; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid #d8e2dc; padding: 7px; text-align: left; font-size: 13px; }
th { background: #e8f1eb; }
"""


def _relative_image(path_value: object, report_path: Path) -> str:
    path = Path(str(path_value))
    if not path.exists():
        return ""
    return os.path.relpath(path, report_path.parent).replace(os.sep, "/")


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "review.html"
    tracklets = pd.read_csv(run_dir / "tracklets.csv")
    assignments_path = run_dir / "pseudo_id_assignments.csv"
    assignments = pd.read_csv(assignments_path) if assignments_path.exists() else pd.DataFrame()
    valid = tracklets.loc[bool_mask(tracklets["valid"])].copy()
    if not assignments.empty:
        valid = valid.merge(assignments, on=["tracklet_id", "session_id"], how="left")
    else:
        valid["pseudo_id"] = "UNCLUSTERED"
        valid["similarity_to_cluster"] = 0.0
        valid["needs_review"] = True

    labels_path = run_dir / "labels_template.csv"
    if not labels_path.exists():
        valid[["tracklet_id", "pseudo_id", "session_id"]].assign(
            cow_id="", confirmed=False, notes=""
        ).to_csv(labels_path, index=False)

    sections: list[str] = []
    for pseudo_id, group in valid.groupby("pseudo_id", dropna=False):
        cards: list[str] = []
        for _, row in group.sort_values(["session_id", "start_s"]).iterrows():
            image_src = _relative_image(row.get("contact_path", ""), report_path)
            image_html = f'<img src="{html.escape(image_src)}" alt="tracklet contact sheet">' if image_src else "<em>No image</em>"
            cards.append(
                "<div class='card'>"
                f"{image_html}"
                f"<div class='meta'><b>{html.escape(str(row['tracklet_id']))}</b><br>"
                f"session={html.escape(str(row['session_id']))}<br>"
                f"time={float(row['start_s']):.1f}–{float(row['end_s']):.1f}s<br>"
                f"quality={float(row['mean_quality']):.3f}, frames={int(row['n_best_frames'])}<br>"
                f"cluster_similarity={float(row.get('similarity_to_cluster', 0.0)):.3f}</div></div>"
            )
        sections.append(
            f"<section class='cluster'><h2>{html.escape(str(pseudo_id))} — {len(group)} tracklet</h2>"
            f"<div class='cards'>{''.join(cards)}</div></section>"
        )

    pair_html = ""
    candidate_path = run_dir / "candidate_pairs.csv"
    if candidate_path.exists():
        try:
            pairs = pd.read_csv(candidate_path)
        except EmptyDataError:
            pairs = pd.DataFrame()
        if not pairs.empty:
            lookup = valid.set_index("tracklet_id", drop=False)
            seen: set[tuple[str, str]] = set()
            visual_pairs: list[str] = []
            for _, pair in pairs.sort_values("cosine_similarity", ascending=False).iterrows():
                left_id = str(pair["query_tracklet_id"])
                right_id = str(pair["candidate_tracklet_id"])
                key = tuple(sorted((left_id, right_id)))
                if key in seen or left_id not in lookup.index or right_id not in lookup.index:
                    continue
                seen.add(key)
                cards: list[str] = []
                for tracklet_id in (left_id, right_id):
                    row = lookup.loc[tracklet_id]
                    image_src = _relative_image(row.get("contact_path", ""), report_path)
                    image_html = (
                        f'<img src="{html.escape(image_src)}" alt="candidate tracklet">'
                        if image_src
                        else "<em>No image</em>"
                    )
                    cards.append(
                        "<div class='card'>"
                        f"{image_html}<div class='meta'><b>{html.escape(tracklet_id)}</b><br>"
                        f"session={html.escape(str(row['session_id']))}</div></div>"
                    )
                visual_pairs.append(
                    "<section class='pair'>"
                    f"<h3>Candidate similarity: {float(pair['cosine_similarity']):.3f}</h3>"
                    f"<div class='pair-images'>{''.join(cards)}</div></section>"
                )
                if len(visual_pairs) >= 40:
                    break
            pair_html = (
                "<h2>Visual cross-session candidate pairs</h2>"
                "<p>These are ranked suggestions only. Matching coat patterns must be confirmed manually.</p>"
                + "".join(visual_pairs)
                + "<h2>Candidate table</h2>"
                + pairs.head(100).to_html(index=False, escape=True)
            )

    document = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Cow Re-ID Review</title><style>{STYLE}</style></head>
<body><h1>Cow Re-ID tracklet review</h1>
<div class="notice"><b>Important:</b> PSEUDO IDs are appearance-based hypotheses, not verified farm Cow_ID values.
Confirm them with a human, ear tag, or RFID before operational use.</div>
<p>Valid tracklets: <b>{len(valid)}</b> · Candidate pseudo identities: <b>{valid['pseudo_id'].nunique()}</b></p>
<p>Edit <code>labels_template.csv</code>, or launch the optional Streamlit review application.</p>
{''.join(sections)}{pair_html}</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path
