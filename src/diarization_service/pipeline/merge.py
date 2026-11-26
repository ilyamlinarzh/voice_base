from bisect import bisect_right


def merge_segments_with_words(
    diarization_segments: list[dict],
    words: list[dict],
) -> list[dict]:
    if not diarization_segments:
        return []
    sorted_segs = sorted(diarization_segments, key=lambda s: s["start"])
    seg_starts = [s["start"] for s in sorted_segs]
    seg_ends = [s["end"] for s in sorted_segs]

    sorted_words = sorted(words, key=lambda w: w["start"])

    seg_word_map: dict[int, list[dict]] = {i: [] for i in range(len(sorted_segs))}
    for word in sorted_words:
        idx = bisect_right(seg_starts, word["start"]) - 1
        if idx < 0:
            continue
        seg = sorted_segs[idx]
        if word["start"] < seg["end"]:
            seg_word_map[idx].append(word)

    final_map: dict[int, dict] = {}
    for new_idx, original_seg in enumerate(sorted_segs):
        seg_words = seg_word_map[new_idx]
        text = " ".join(w["text"] for w in seg_words)
        final_map[id(original_seg)] = {
            "text": text,
            "words": seg_words,
        }

    result = []
    for seg in diarization_segments:
        info = final_map.get(id(seg), {"text": "", "words": []})
        result.append({
            **seg,
            "text": info["text"],
            "words": info["words"],
        })
    return result