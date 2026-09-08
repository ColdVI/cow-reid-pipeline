from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import bbox_iou


def _center(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    missed: int = 0


class CentroidTracker:
    """Small deterministic tracker used by the offline motion fallback."""

    def __init__(self, frame_width: int, max_distance: float = 0.22, max_missed: int = 8):
        self.frame_width = max(frame_width, 1)
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: dict[int, _Track] = {}

    def _new(self, box: Sequence[float]) -> int:
        track_id = self.next_id
        self.next_id += 1
        box_array = np.asarray(box, dtype=np.float32)
        center = _center(box_array)
        self.tracks[track_id] = _Track(track_id, box_array, center, np.zeros(2, dtype=np.float32))
        return track_id

    def update(self, boxes: Sequence[Sequence[float]]) -> list[int]:
        boxes_array = [np.asarray(box, dtype=np.float32) for box in boxes]
        if not boxes_array:
            for track_id in list(self.tracks):
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]
            return []
        if not self.tracks:
            return [self._new(box) for box in boxes_array]

        track_ids = list(self.tracks)
        cost = np.full((len(track_ids), len(boxes_array)), 1e6, dtype=np.float32)
        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted = track.center + track.velocity
            for col, box in enumerate(boxes_array):
                center = _center(box)
                distance = float(np.linalg.norm(center - predicted) / self.frame_width)
                if distance <= self.max_distance:
                    cost[row, col] = distance + 0.25 * (1.0 - bbox_iou(track.box, box))

        rows, cols = linear_sum_assignment(cost)
        assigned_detections: dict[int, int] = {}
        assigned_tracks: set[int] = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if cost[row, col] >= 1e5:
                continue
            track_id = track_ids[row]
            track = self.tracks[track_id]
            new_center = _center(boxes_array[col])
            track.velocity = 0.7 * track.velocity + 0.3 * (new_center - track.center)
            track.center = new_center
            track.box = boxes_array[col]
            track.missed = 0
            assigned_detections[col] = track_id
            assigned_tracks.add(track_id)

        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]

        output: list[int] = []
        for index, box in enumerate(boxes_array):
            output.append(assigned_detections.get(index) or self._new(box))
        return output


class MotionDetector:
    """Fixed-camera foreground fallback. It is intentionally conservative."""

    def __init__(self, frame_width: int, frame_height: int, config: dict):
        self.width = frame_width
        self.height = frame_height
        self.min_area = float(config.get("min_area", 0.012)) * frame_width * frame_height
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(config.get("history", 250)),
            varThreshold=float(config.get("var_threshold", 28)),
            detectShadows=False,
        )
        self.tracker = CentroidTracker(
            frame_width,
            max_distance=float(config.get("max_distance", 0.22)),
            max_missed=int(config.get("max_missed", 8)),
        )

    def detect_and_track(self, frame: np.ndarray) -> tuple[list[list[float]], list[int], list[float]]:
        mask = self.subtractor.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or w < 0.10 * self.width or h < 0.15 * self.height:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        boxes.sort(key=lambda box: box[0])
        ids = self.tracker.update(boxes)
        return boxes, ids, [0.5] * len(boxes)

