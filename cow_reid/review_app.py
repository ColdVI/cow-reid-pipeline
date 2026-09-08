from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from cow_reid.audit import inspect_label_consistency
from cow_reid.extract import APPEARANCE_CONSISTENCY_FLAG_THRESHOLD
from cow_reid.review_state import load_excluded_tracklets
from cow_reid.identity import (
    accept_pair_as_same,
    assign_tracklets,
    assign_tracklets_checked,
    identity_candidates,
    identity_catalog,
    load_identity_labels,
    load_pair_reviews,
    next_cow_id,
    normalize_cow_id,
    record_pair_as_different,
    reject_identity_candidate,
    rename_identity,
    save_identity_labels,
    save_pair_review,
    unassign_tracklets,
)
from cow_reid.posture import import_posture_scores, load_posture_observations, posture_history
from cow_reid.utils import bool_mask
from cow_reid.debug_overlay import render_debug_evidence
from cow_reid.media import enrich_tracks_with_recording_time
from cow_reid.review_state import record_tracklet_state


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default=os.environ.get("COW_REID_RUN"))
    args, _ = parser.parse_known_args()
    if not args.run:
        parser.error("--run is required (or set COW_REID_RUN)")
    return args


def _image_path(row: pd.Series) -> Path | None:
    value = str(row.get("contact_path", ""))
    path = Path(value)
    return path if value and path.exists() else None


def _show_track(row: pd.Series, cow_id: str = "", similarity: float | None = None) -> None:
    path = _image_path(row)
    if path is not None:
        st.image(str(path), width="stretch")
    else:
        st.info("Bu tracklet için contact.jpg bulunamadı.")
    details = [
        str(row.get("tracklet_id", "")),
        f"video={row.get('source_filename', row.get('video_id', ''))}",
        f"session={row.get('session_id', '')}",
        f"time={float(row.get('start_s', 0.0)):.1f}s",
    ]
    if cow_id:
        details.append(f"cow_id={cow_id}")
    if similarity is not None:
        details.append(f"similarity={similarity:.3f}")
    st.code("\n".join(details))


def _show_video(row: pd.Series, run_dir: Path, cow_id: str = "", status: str = "UNKNOWN", similarity: float | None = None, key: str = "video") -> None:
    try:
        clip = render_debug_evidence(row, run_dir / "debug_evidence", cow_id=cow_id, status=status, similarity=similarity)
        st.video(str(clip), autoplay=False, loop=True)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Video geçişi hazırlanamadı: {exc}")
    recorded = str(row.get("recording_start", ""))
    st.caption(
        f"{row.get('source_filename', row.get('video_id', ''))} · {recorded or 'zaman bilinmiyor'} · "
        f"{float(row.get('start_s', 0)):.1f}–{float(row.get('end_s', 0)):.1f}s · {row.get('tracklet_id', '')}"
    )
    path = _image_path(row)
    if path is not None:
        with st.expander("Yardımcı torso/contact sheet"):
            st.image(str(path), width="stretch")


def _video_review_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Video geçişiyle insan kontrolü")
    st.caption("Karar bir frame'e değil, kutu ve trajectory ile gösterilen bütün tracklet geçişine uygulanır.")
    catalog = identity_catalog(labels)
    if catalog.empty:
        st.info("Henüz aranabilir COW kimliği yok.")
        return
    query = st.text_input("İnek ara", placeholder="COW_0012", key="video_cow_search").strip()
    choices = catalog["cow_id"].astype(str).tolist()
    normalized_query = normalize_cow_id(query)
    matches = [
        value for value in choices
        if not query or value == normalized_query or query.upper() in value.upper()
    ]
    if not matches:
        st.warning(f"{query} ile eşleşen kimlik bulunamadı.")
        return
    cow_id = st.selectbox("Kimlik", matches, key="video_cow_choice")
    enriched = enrich_tracks_with_recording_time(tracks, run_dir).drop_duplicates("tracklet_id").set_index("tracklet_id")
    member_ids = labels.loc[bool_mask(labels["confirmed"], default=False) & labels["cow_id"].astype(str).eq(cow_id), "tracklet_id"].astype(str)
    members = enriched.loc[enriched.index.intersection(member_ids)].reset_index()
    if members.empty:
        st.warning("Bu kimliğe bağlı video geçişi bulunamadı.")
        return
    members["_time"] = pd.to_datetime(members.get("recording_start"), errors="coerce")
    members = members.sort_values(["_time", "start_s"], na_position="last")
    days = int(members["_time"].dt.date.nunique())
    metrics = st.columns(3); metrics[0].metric("Video geçişi", len(members)); metrics[1].metric("Gerçek gün", days); metrics[2].metric("Kimlik", cow_id)
    options = members.index.tolist()
    selected_index = st.selectbox(
        "İzlenecek geçiş",
        options,
        format_func=lambda idx: f"{members.loc[idx, '_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(members.loc[idx, '_time']) else 'zaman bilinmiyor'} · {members.loc[idx, 'source_filename']} · {members.loc[idx, 'start_s']:.1f}s",
        key="member_video",
    )
    selected = members.loc[selected_index]
    _show_video(selected, run_dir, cow_id, "HUMAN_CONFIRMED", key="member")
    action_cols = st.columns(3)
    if action_cols[0].button("Emin değilim", key=f"unsure_member_{selected['tracklet_id']}"):
        record_tracklet_state(run_dir, str(selected["tracklet_id"]), "unsure"); st.toast("Tracklet emin değilim kuyruğuna alındı.")
    if action_cols[1].button("Tracklet hatalı", key=f"bad_member_{selected['tracklet_id']}"):
        record_tracklet_state(run_dir, str(selected["tracklet_id"]), "tracklet_error"); st.toast("Tracker düzeltme kuyruğuna alındı.")
    split_at = action_cols[2].number_input("Bölme zamanı (sn)", min_value=float(selected["start_s"]), max_value=float(selected["end_s"]), value=float((selected["start_s"] + selected["end_s"]) / 2), key=f"split_time_{selected['tracklet_id']}")
    if action_cols[2].button("Geçişi böl", key=f"split_member_{selected['tracklet_id']}"):
        record_tracklet_state(run_dir, str(selected["tracklet_id"]), "split_requested", split_at_s=float(split_at)); st.toast("Bölme isteği tracker kuyruğuna alındı.")

    st.divider(); st.markdown("#### Benzer video geçişini karşılaştır")
    try:
        candidates = identity_candidates(run_dir, labels, cow_id, top_k=8, exclude_assigned=True, exclude_seen_sessions=False)
    except ValueError as exc:
        st.warning(str(exc)); return
    if candidates.empty:
        st.info("Karşılaştırılacak atanmış olmayan aday bulunamadı."); return
    candidate_index = st.selectbox(
        "Aday geçiş", candidates.index.tolist(),
        format_func=lambda idx: f"similarity={candidates.loc[idx, 'similarity']:.3f} · {candidates.loc[idx, 'source_filename']} · {candidates.loc[idx, 'start_s']:.1f}s",
        key="candidate_video",
    )
    candidate = candidates.loc[candidate_index]
    if str(candidate["tracklet_id"]) in enriched.index:
        candidate_id = str(candidate["tracklet_id"])
        candidate = enriched.loc[candidate_id].copy()
        candidate["tracklet_id"] = candidate_id
        candidate["similarity"] = float(candidates.loc[candidate_index, "similarity"])
    left, right = st.columns(2)
    with left: st.markdown(f"**Referans: {cow_id}**"); _show_video(selected, run_dir, cow_id, "HUMAN_CONFIRMED", key="reference")
    with right: st.markdown("**Model adayı**"); _show_video(candidate, run_dir, cow_id, "PREDICTED", float(candidate["similarity"]), key="candidate")
    same, different, unsure, bad = st.columns(4)
    candidate_id = str(candidate["tracklet_id"]); reference_id = str(selected["tracklet_id"])
    if same.button("Aynı inek", type="primary", width="stretch"):
        try:
            updated, _ = assign_tracklets_checked(run_dir, labels, [candidate_id], cow_id, source="video_tracklet_review")
            save_identity_labels(updated, run_dir); save_pair_review(run_dir, reference_id, candidate_id, "same", cow_id, float(candidate["similarity"])); st.rerun()
        except ValueError as exc: st.error(str(exc))
    if different.button("Farklı inek", width="stretch"):
        try:
            record_pair_as_different(run_dir, labels, reference_id, candidate_id, float(candidate["similarity"])); st.rerun()
        except ValueError as exc: st.error(str(exc))
    if unsure.button("Emin değilim", width="stretch"):
        save_pair_review(run_dir, reference_id, candidate_id, "unsure", similarity=float(candidate["similarity"])); st.rerun()
    if bad.button("Tracklet hatalı", width="stretch"):
        record_tracklet_state(run_dir, candidate_id, "tracklet_error"); st.rerun()


def _canonical_pair(left: object, right: object) -> str:
    return "||".join(sorted((str(left), str(right))))


def _unassigned_best_guess_queue(run_dir: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per still-unassigned tracklet: its single best-guess match, highest confidence first.

    This is the model's full proposal for the remaining backlog, not just the tiny
    slice that clears the auto-confirm bar — the human's job here is to rubber-stamp
    or correct each guess in order, not to search for candidates from scratch.
    """
    path = run_dir / "candidate_pairs.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        pairs = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if pairs.empty or "rank" not in pairs:
        return pd.DataFrame()
    top1 = pairs.loc[pairs["rank"] == 1].rename(columns={"query_tracklet_id": "tracklet_id"})
    unassigned_ids = set(labels.loc[~bool_mask(labels["confirmed"], default=False), "tracklet_id"].astype(str))
    queue = top1.loc[top1["tracklet_id"].astype(str).isin(unassigned_ids)].copy()
    if queue.empty:
        return queue
    queue["pair_key"] = [
        _canonical_pair(left, right)
        for left, right in zip(queue["tracklet_id"], queue["candidate_tracklet_id"])
    ]
    reviews = load_pair_reviews(run_dir)
    reviewed_keys = {
        _canonical_pair(left, right)
        for left, right in zip(reviews["left_tracklet_id"], reviews["right_tracklet_id"])
    }
    queue = queue.loc[~queue["pair_key"].isin(reviewed_keys)]
    return queue.sort_values("cosine_similarity", ascending=False).drop_duplicates("tracklet_id").reset_index(drop=True)


def _identity_choice(catalog: pd.DataFrame, proposed: str, key: str) -> str | None:
    existing = catalog["cow_id"].astype(str).tolist() if not catalog.empty else []
    options = [f"Yeni kimlik oluştur: {proposed}", *existing]
    normalized_proposed = normalize_cow_id(proposed)
    default_index = options.index(normalized_proposed) if normalized_proposed in existing else 0
    choice = st.selectbox("Hedef kimlik", options, index=default_index, key=key)
    return None if choice.startswith("Yeni kimlik oluştur:") else normalize_cow_id(choice)


def _candidate_review_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Model önerisiyle toplu triyaj")
    st.caption(
        "Model, kalan her atanmamış tracklet için en iyi tahminini bulup en yüksek güvenden "
        "başlayarak sırayla önüne getiriyor. Sıfırdan aday aramıyorsun — önerilen çifti onaylıyor "
        "ya da düzeltiyorsun, sistem otomatik bir sonrakine geçiyor."
    )
    lookup = tracks.drop_duplicates("tracklet_id").set_index("tracklet_id")
    queue = _unassigned_best_guess_queue(run_dir, labels)
    if not queue.empty:
        queue = queue.loc[
            queue["tracklet_id"].astype(str).isin(lookup.index.astype(str))
            & queue["candidate_tracklet_id"].astype(str).isin(lookup.index.astype(str))
        ]
    if queue.empty:
        st.success("Sırada model önerisi olan tracklet kalmadı.")
        return
    st.metric("Sırada kalan", len(queue))
    pair = queue.iloc[0]
    left_id = str(pair["tracklet_id"])
    right_id = str(pair["candidate_tracklet_id"])
    label_lookup = labels.set_index("tracklet_id")
    right_cow = normalize_cow_id(label_lookup.loc[right_id, "cow_id"]) if right_id in label_lookup.index else ""
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("**Atanmamış tracklet**")
        _show_track(lookup.loc[left_id])
    with right_column:
        st.markdown(f"**Model önerisi{f' — {right_cow}' if right_cow else ' (o da atanmamış)'}**")
        _show_track(lookup.loc[right_id], right_cow, similarity=float(pair["cosine_similarity"]))

    geometry_similarity = pair.get("geometry_similarity")
    if pd.notna(geometry_similarity) and pair["cosine_similarity"] >= 0.9 and geometry_similarity < 0.5:
        st.warning(
            f"⚠️ Görünüm benzerliği yüksek (sim={pair['cosine_similarity']:.3f}) ama benek "
            f"konumu benzerliği düşük (geo={geometry_similarity:.3f}) — desen ters/simetrik "
            "olabilir (ör. sol-alt/sağ-üst vs sağ-üst/sol-alt). Onaylamadan önce iki görseldeki "
            "benek konumlarını dikkatlice karşılaştır."
        )

    same_button_label = f"✓ Onayla — {right_cow}" if right_cow else "✓ Onayla — aynı inek (yeni kimlik)"
    same_column, different_column, skip_column = st.columns(3)
    with same_column:
        if st.button(same_button_label, type="primary", width="stretch"):
            try:
                _, resolved = accept_pair_as_same(
                    run_dir, labels, left_id, right_id, similarity=float(pair["cosine_similarity"])
                )
                st.toast(f"{left_id} → {resolved}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with different_column:
        if st.button("✕ Farklı inek", width="stretch"):
            try:
                record_pair_as_different(
                    run_dir, labels, left_id, right_id, similarity=float(pair["cosine_similarity"])
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with skip_column:
        if st.button("Atla / emin değilim", width="stretch"):
            save_pair_review(run_dir, left_id, right_id, "unsure", similarity=float(pair["cosine_similarity"]))
            st.rerun()


def _seed_cluster_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Pseudo kümeden kimlik başlat")
    assignments_path = run_dir / "pseudo_id_assignments.csv"
    if not assignments_path.exists():
        st.info("Bu run içinde pseudo kimlik kümesi yok.")
        return
    assignments = pd.read_csv(assignments_path)
    confirmed_cow_by_tracklet = (
        labels.loc[bool_mask(labels["confirmed"], default=False)]
        .set_index("tracklet_id")["cow_id"]
        .astype(str)
    )
    assignments["_cow_id"] = assignments["tracklet_id"].astype(str).map(confirmed_cow_by_tracklet)

    def _summarize(group: pd.DataFrame) -> pd.Series:
        linked = sorted(group["_cow_id"].dropna().unique())
        return pd.Series(
            {
                "tracklets": len(group),
                "sessions": group["session_id"].nunique(),
                "yeni": int(group["_cow_id"].isna().sum()),
                "linked_cow": linked[0] if linked else "",
            }
        )

    groups = assignments.groupby("pseudo_id").apply(_summarize, include_groups=False)
    only_new = st.checkbox("Sadece yeni tracklet içeren kümeleri göster", value=True)
    view = groups.loc[groups["yeni"] > 0] if only_new else groups
    if view.empty:
        st.success("Yeni tracklet içeren pseudo küme kalmadı.")
        return
    view = view.sort_values(["yeni", "tracklets", "sessions"], ascending=False)
    pseudo_id = st.selectbox(
        "Pseudo küme",
        view.index.tolist(),
        format_func=lambda value: (
            f"{value} · {int(view.loc[value, 'tracklets'])} tracklet · "
            f"{int(view.loc[value, 'yeni'])} yeni"
            + (f" · → {view.loc[value, 'linked_cow']}" if view.loc[value, "linked_cow"] else "")
        ),
    )
    member_ids = assignments.loc[assignments["pseudo_id"].eq(pseudo_id), "tracklet_id"].astype(str).tolist()
    lookup = tracks.drop_duplicates("tracklet_id").set_index("tracklet_id")
    selected: list[str] = []
    columns = st.columns(min(3, max(1, len(member_ids))))
    for index, tracklet_id in enumerate(member_ids):
        if tracklet_id not in lookup.index:
            continue
        with columns[index % len(columns)]:
            _show_track(lookup.loc[tracklet_id], confirmed_cow_by_tracklet.get(tracklet_id, ""))
            if st.checkbox("Kimliğe dahil et", value=True, key=f"seed_{pseudo_id}_{tracklet_id}"):
                selected.append(tracklet_id)
    catalog = identity_catalog(labels)
    proposed = view.loc[pseudo_id, "linked_cow"] or next_cow_id(labels)
    requested = _identity_choice(catalog, proposed, "seed_target_identity")
    if st.button("Seçilen geçişleri kimliğe ata", type="primary", disabled=not selected):
        updated, resolved = assign_tracklets(labels, selected, requested, source="pseudo_cluster_review")
        save_identity_labels(updated, run_dir)
        st.toast(f"{len(selected)} geçiş {resolved} kimliğine eklendi.")
        st.rerun()


def _identity_gallery_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("İnek odaklı hızlı etiketleme")
    st.caption(
        "Bir COW seç: önce ona bağlı tüm görüntüleri gör, sonra yalnızca modelin en yakın birkaç "
        "atanmamış adayını kabul et veya reddet. Binlerce çifti dolaşman gerekmiyor."
    )
    catalog = identity_catalog(labels)
    if catalog.empty:
        st.info("Henüz COW kimliği oluşturulmadı. Aday eşleştirme veya pseudo küme sekmesinden ilk kimliği başlat.")
        return
    cow_id = st.selectbox("İnek kimliği", catalog["cow_id"].astype(str).tolist(), key="gallery_cow")
    audit, _, _ = inspect_label_consistency(labels, load_pair_reviews(run_dir), load_excluded_tracklets(run_dir))
    audit_row = audit.loc[audit["cow_id"].eq(cow_id)] if not audit.empty else audit
    identity_conflicted = bool(not audit_row.empty and audit_row.iloc[0]["status"] == "conflict")
    if identity_conflicted:
        st.error(
            f"{cow_id} içinde önceki aynı/farklı kararları çelişiyor. Yeni görüntü eklemeden önce "
            "Çelişki denetimi sekmesinde düzelt."
        )
    stats = catalog.loc[catalog["cow_id"].eq(cow_id)].iloc[0]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Tracklet", int(stats["tracklets"]))
    metric_columns[1].metric("Video", int(stats["videos"]))
    metric_columns[2].metric("Session", int(stats["sessions"]))

    lookup = tracks.drop_duplicates("tracklet_id").set_index("tracklet_id")
    members = labels.loc[
        bool_mask(labels["confirmed"], default=False) & labels["cow_id"].astype(str).eq(cow_id)
    ].sort_values(["session_id", "tracklet_id"])
    st.markdown("#### Bu kimliğe bağlı görüntüler")
    remove_ids: list[str] = []
    columns = st.columns(min(3, max(1, len(members))))
    for index, (_, member) in enumerate(members.iterrows()):
        tracklet_id = str(member["tracklet_id"])
        if tracklet_id not in lookup.index:
            continue
        with columns[index % len(columns)]:
            _show_track(lookup.loc[tracklet_id], cow_id)
            if st.checkbox("Kimlikten çıkar", key=f"remove_{cow_id}_{tracklet_id}"):
                remove_ids.append(tracklet_id)
    if st.button("Seçilenleri kimlikten çıkar", disabled=not remove_ids):
        updated = unassign_tracklets(labels, remove_ids)
        save_identity_labels(updated, run_dir)
        st.toast(f"{len(remove_ids)} geçiş kimlikten çıkarıldı.")
        st.rerun()

    with st.expander("Geçici kimliği gerçek çiftlik numarasıyla değiştir"):
        renamed = st.text_input("Yeni kimlik", value=cow_id, key=f"rename_{cow_id}")
        if st.button("Kimliği yeniden adlandır", key=f"rename_button_{cow_id}"):
            try:
                updated = rename_identity(labels, cow_id, renamed)
                save_identity_labels(updated, run_dir)
                st.toast(f"{cow_id} → {normalize_cow_id(renamed)}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("#### Diğer videolardan benzer geçişleri getir")
    left, right = st.columns(2)
    top_k = left.slider("Gösterilecek aday", min_value=3, max_value=12, value=6, step=1)
    exclude_seen = right.checkbox("İneğin zaten görüldüğü session'ları dışla", value=True)
    try:
        candidates = identity_candidates(
            run_dir,
            labels,
            cow_id,
            top_k=top_k,
            exclude_assigned=True,
            exclude_seen_sessions=exclude_seen,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return
    if candidates.empty:
        st.info("Filtrelere uyan atanmamış benzer tracklet bulunamadı.")
        return
    st.caption("Her aday için karar ver, sonuna kadar tıklamayı bekletme — hepsini tek seferde uygularsın.")
    decisions: dict[str, str] = {}
    columns = st.columns(3)
    for index, candidate in candidates.iterrows():
        tracklet_id = str(candidate["tracklet_id"])
        with columns[index % 3]:
            _show_track(candidate, similarity=float(candidate["similarity"]))
            decisions[tracklet_id] = st.radio(
                "Karar",
                ["İncelenmedi", "Aynı inek", "Farklı inek"],
                key=f"decision_{cow_id}_{tracklet_id}",
                horizontal=True,
                label_visibility="collapsed",
            )
    add_ids = [tracklet_id for tracklet_id, decision in decisions.items() if decision == "Aynı inek"]
    reject_ids = [tracklet_id for tracklet_id, decision in decisions.items() if decision == "Farklı inek"]
    if st.button(
        f"Kararları uygula ({len(add_ids)} ekle · {len(reject_ids)} reddet)",
        type="primary",
        disabled=(not add_ids and not reject_ids) or identity_conflicted,
    ):
        try:
            if add_ids:
                updated, _ = assign_tracklets_checked(
                    run_dir, labels, add_ids, cow_id, source="identity_similarity_search"
                )
                save_identity_labels(updated, run_dir)
            for tracklet_id in reject_ids:
                reject_identity_candidate(run_dir, cow_id, tracklet_id)
            st.toast(f"{len(add_ids)} geçiş {cow_id} kimliğine eklendi, {len(reject_ids)} reddedildi.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def _new_identity_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Atanmamış bir geçişten yeni inek başlat")
    confirmed = bool_mask(labels["confirmed"], default=False) & labels["cow_id"].fillna("").ne("")
    unassigned = labels.loc[~confirmed, "tracklet_id"].astype(str)
    available = tracks.loc[tracks["tracklet_id"].astype(str).isin(unassigned)].copy()
    if available.empty:
        st.success("Atanmamış geçiş kalmadı.")
        return
    sort_columns = [column for column in ("mean_quality", "start_s") if column in available]
    if sort_columns:
        available = available.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    lookup = available.drop_duplicates("tracklet_id").set_index("tracklet_id")
    tracklet_id = st.selectbox(
        "Başlangıç tracklet'ı",
        lookup.index.astype(str).tolist(),
        format_func=lambda value: (
            f"{value} · {lookup.loc[value].get('source_filename', lookup.loc[value].get('video_id', ''))} "
            f"· {float(lookup.loc[value].get('start_s', 0.0)):.1f}s"
        ),
    )
    _show_track(lookup.loc[tracklet_id])
    proposed = next_cow_id(labels)
    st.info(f"Bu geçiş yeni {proposed} kimliği olacak. Sonra İnek odaklı sekmede diğer videolar aranacak.")
    if st.button(f"Yeni {proposed} oluştur", type="primary"):
        updated, resolved = assign_tracklets_checked(
            run_dir, labels, [tracklet_id], None, source="manual_identity_seed"
        )
        save_identity_labels(updated, run_dir)
        st.toast(f"{resolved} oluşturuldu.")
        st.rerun()


def _conflict_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Etiket çelişkilerini onar")
    audit, conflicts, _ = inspect_label_consistency(labels, load_pair_reviews(run_dir), load_excluded_tracklets(run_dir))
    bad = audit.loc[audit["status"].eq("conflict")]
    if bad.empty:
        st.success("Doğrudan aynı/farklı çelişkisi bulunmadı.")
        if not audit.empty:
            st.dataframe(audit, width="stretch", hide_index=True)
        return
    st.error(
        f"{len(bad)} kimlik eğitimden otomatik olarak karantinaya alındı. Görsellere bakıp aşağıdaki "
        "çelişkili kararları düzelt."
    )
    st.dataframe(bad, width="stretch", hide_index=True)
    cow_id = st.selectbox("Onarılacak kimlik", bad["cow_id"].astype(str).tolist())
    member_ids = labels.loc[
        bool_mask(labels["confirmed"], default=False) & labels["cow_id"].astype(str).eq(cow_id),
        "tracklet_id",
    ].astype(str).tolist()
    with st.expander(f"{cow_id} grubunu sıfırdan ayır ({len(member_ids)} tracklet)"):
        st.warning(
            "Bu işlem yalnızca bu COW grubunun atamalarını kaldırır; görüntüleri veya geçmiş "
            "aynı/farklı kararlarını silmez. labels.csv otomatik yedeklenir."
        )
        confirmation = st.checkbox(
            f"{cow_id} grubunu atanmamış hale getirmeyi onaylıyorum", key=f"reset_confirm_{cow_id}"
        )
        if st.button(
            f"{cow_id} grubunu sıfırla", disabled=not confirmation, key=f"reset_identity_{cow_id}"
        ):
            save_identity_labels(unassign_tracklets(labels, member_ids), run_dir)
            st.toast(f"{cow_id} içindeki {len(member_ids)} tracklet atanmamış hale getirildi.")
            st.rerun()
    relevant = conflicts.loc[
        conflicts["cow_id"].eq(cow_id) | conflicts["other_cow_id"].eq(cow_id)
    ].reset_index(drop=True)
    if relevant.empty:
        return
    conflict_index = st.selectbox(
        "Çelişkili karar",
        relevant.index.tolist(),
        format_func=lambda index: (
            f"{relevant.loc[index, 'conflict_type']} · "
            f"{relevant.loc[index, 'left_tracklet_id']} ↔ {relevant.loc[index, 'right_tracklet_id']}"
        ),
    )
    conflict = relevant.loc[conflict_index]
    left_id = str(conflict["left_tracklet_id"])
    right_id = str(conflict["right_tracklet_id"])
    lookup = tracks.drop_duplicates("tracklet_id").set_index("tracklet_id")
    columns = st.columns(2)
    with columns[0]:
        if left_id in lookup.index:
            _show_track(lookup.loc[left_id], cow_id)
    with columns[1]:
        if right_id in lookup.index:
            _show_track(lookup.loc[right_id], cow_id)

    if conflict["conflict_type"] == "different_pair_inside_identity":
        st.write("Hangisi doğru?")
        same_col, left_col, right_col = st.columns(3)
        with same_col:
            if st.button("Aslında aynıydılar", type="primary", width="stretch"):
                save_pair_review(run_dir, left_id, right_id, "same", cow_id=cow_id)
                st.rerun()
        with left_col:
            if st.button("Solu kimlikten çıkar", width="stretch"):
                save_identity_labels(unassign_tracklets(labels, [left_id]), run_dir)
                st.rerun()
        with right_col:
            if st.button("Sağı kimlikten çıkar", width="stretch"):
                save_identity_labels(unassign_tracklets(labels, [right_id]), run_dir)
                st.rerun()
    else:
        st.warning("Bu çift aynı işaretlenmiş ama iki ayrı COW altında kalmış.")
        if st.button(f"İki kimliği {cow_id} altında birleştir", type="primary"):
            try:
                accept_pair_as_same(run_dir, labels, left_id, right_id, requested_cow_id=cow_id)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def _quality_flags_tab(run_dir: Path, tracks: pd.DataFrame, labels: pd.DataFrame) -> None:
    st.subheader("Şüpheli izler")
    st.caption(
        "Bir tracklet'in kendi kareleri arasında büyük görünüm sıçraması var mı diye otomatik "
        "kontrol (extract.py::_intra_track_appearance_consistency). Genelde ya tracker'ın iki "
        "farklı hayvanı tek ize birleştirmesi (id-switch), ya da aşırı açı değişimi. Bu liste "
        "yalnız insan gözünden geçmesi için — hiçbir satır otomatik olarak dışlanmaz veya "
        "kimliğe bağlanmaz."
    )
    if "quality_flags" not in tracks.columns:
        st.info("Bu run eski bir extract.py sürümüyle çıkarılmış, quality_flags kolonu yok.")
        return
    flagged = tracks.loc[tracks["quality_flags"].fillna("").str.contains("large_appearance_change")].copy()
    if flagged.empty:
        st.success("Şüpheli görünüm sıçraması bulunmadı.")
        return
    reviews_path = run_dir / "tracklet_reviews.csv"
    reviewed = set(pd.read_csv(reviews_path)["tracklet_id"].astype(str)) if reviews_path.exists() else set()
    flagged["already_reviewed"] = flagged["tracklet_id"].astype(str).isin(reviewed)
    flagged = flagged.sort_values("appearance_consistency")
    st.metric("Şüpheli iz", f"{(~flagged['already_reviewed']).sum()} / {len(flagged)} incelenmemiş")
    pending = flagged.loc[~flagged["already_reviewed"]]
    if pending.empty:
        st.success("Tüm şüpheli izler zaten incelenmiş/dışlanmış.")
        return
    row = pending.iloc[0]
    st.caption(
        f"appearance_consistency={float(row['appearance_consistency']):.3f} "
        f"(eşik={APPEARANCE_CONSISTENCY_FLAG_THRESHOLD}, düşük=daha şüpheli)"
    )
    _show_track(row)
    cols = st.columns(3)
    if cols[0].button("Tracklet hatalı (iki hayvan karışmış)", type="primary", key=f"qflag_bad_{row['tracklet_id']}"):
        record_tracklet_state(run_dir, str(row["tracklet_id"]), "tracklet_error", notes="auto-flagged: large_appearance_change")
        st.rerun()
    if cols[1].button("Sorun yok, geçerli", key=f"qflag_ok_{row['tracklet_id']}"):
        record_tracklet_state(run_dir, str(row["tracklet_id"]), "valid", notes="human-cleared: large_appearance_change")
        st.rerun()
    if cols[2].button("Emin değilim", key=f"qflag_unsure_{row['tracklet_id']}"):
        record_tracklet_state(run_dir, str(row["tracklet_id"]), "unsure", notes="auto-flagged: large_appearance_change")
        st.rerun()


def _posture_tab(run_dir: Path, labels: pd.DataFrame) -> None:
    st.subheader("Kamburluk geçmişi")
    st.caption(
        "Keypoint hattı daha sonra tracklet_id ile skor verdiğinde burada aynı COW kimliğinin zaman içindeki ölçümleri birikir."
    )
    with st.expander("Keypoint skor CSV sözleşmesi"):
        st.code(
            "tracklet_id,frame_idx,arch_score,model_name,model_version\n"
            "vid_xxx_t0001,125,0.31,cow_keypoints,v1\n"
            "vid_xxx_t0001,126,0.34,cow_keypoints,v1",
            language="csv",
        )
        st.write("Skor sütunu `arch_score`, `posture_score`, `hunch_score` veya `score` olabilir.")

    uploaded = st.file_uploader("Keypoint skor CSV yükle", type=["csv"])
    if uploaded is not None and st.button("Skorları içe aktar"):
        import_dir = run_dir / "posture_imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        target = import_dir / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        try:
            output = import_posture_scores(run_dir, target)
            st.toast(f"Skorlar {output.name} dosyasına aktarıldı.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    observations = load_posture_observations(run_dir)
    if observations.empty:
        st.info("Henüz keypoint/kamburluk skoru içe aktarılmadı.")
        return
    catalog = identity_catalog(labels)
    available = sorted(set(observations["cow_id"].dropna().astype(str)).intersection(catalog["cow_id"].astype(str)))
    if not available:
        st.warning("Skorlar var ancak skor verilen tracklet'lar henüz bir COW kimliğine bağlanmamış.")
        return
    cow_id = st.selectbox("Geçmişi gösterilecek inek", available, key="posture_cow")
    history = posture_history(run_dir, cow_id)
    if history.empty:
        st.info("Bu inek için ölçüm yok.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Ölçüm", len(history))
    metrics[1].metric("Ortalama skor", f"{history['arch_score_mean'].mean():.3f}")
    metrics[2].metric("En yüksek skor", f"{history['arch_score_mean'].max():.3f}")
    latest_delta = history["delta_from_baseline"].dropna()
    metrics[3].metric("Son baseline farkı", "—" if latest_delta.empty else f"{latest_delta.iloc[-1]:+.3f}")
    chart = history.set_index("observation_order")[["arch_score_mean", "personal_baseline"]]
    st.line_chart(chart, x_label="Ölçüm sırası", y_label="Kamburluk skoru")
    st.dataframe(
        history[
            [
                "observed_at",
                "session_id",
                "tracklet_id",
                "arch_score_mean",
                "arch_score_p90",
                "arch_score_std",
                "n_scored_frames",
                "personal_baseline",
                "delta_from_baseline",
                "model_version",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    run_dir = Path(_args().run).resolve()
    st.set_page_config(page_title="Cow Identity Workbench", layout="wide")
    st.title("Cow Identity Workbench")
    st.warning(
        "COW_0001 gibi kimlikler şimdilik görsel olarak doğrulanan geçici kimliklerdir. "
        "Gerçek çiftlik numarası geldiğinde aynı kayıt yeniden adlandırılabilir."
    )
    if not (run_dir / "tracklets.csv").exists():
        st.error(f"tracklets.csv bulunamadı: {run_dir}")
        return
    tracks = pd.read_csv(run_dir / "tracklets.csv")
    if "valid" in tracks:
        tracks = tracks.loc[bool_mask(tracks["valid"], default=False)].copy()
    labels = load_identity_labels(run_dir)
    catalog = identity_catalog(labels)
    assigned_tracklets = int(bool_mask(labels["confirmed"], default=False).sum())
    header = st.columns(4)
    header[0].metric("Geçerli tracklet", len(tracks))
    header[1].metric("Oluşturulan COW", len(catalog))
    header[2].metric("Atanmış tracklet", assigned_tracklets)
    header[3].metric("Atanmamış", max(0, len(labels) - assigned_tracklets))

    tabs = st.tabs(
        [
            "Video ile inek kontrolü",
            "Model önerisiyle toplu triyaj",
            "Torso ile hızlı tarama",
            "Yeni kimlik başlat",
            "Çelişki denetimi",
            "Şüpheli izler",
            "Kamburluk geçmişi",
        ]
    )
    with tabs[0]:
        _video_review_tab(run_dir, tracks, labels)
    with tabs[1]:
        _candidate_review_tab(run_dir, tracks, labels)
    with tabs[2]:
        _identity_gallery_tab(run_dir, tracks, labels)
    with tabs[3]:
        _new_identity_tab(run_dir, tracks, labels)
    with tabs[4]:
        _conflict_tab(run_dir, tracks, labels)
    with tabs[5]:
        _quality_flags_tab(run_dir, tracks, labels)
    with tabs[6]:
        _posture_tab(run_dir, labels)


if __name__ == "__main__":
    main()
