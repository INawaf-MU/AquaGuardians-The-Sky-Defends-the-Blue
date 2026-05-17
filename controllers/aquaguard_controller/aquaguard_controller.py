"""AquaGuard Webots drone controller.

This controller uses a simple OpenCV contour detector to find likely
waste/plastic objects in the drone camera image, remembers unique detections,
and prints a final report when the scan ends.

The detector is intentionally simple for a university Webots simulation. The
`WasteDetector.detect()` method is the best place to replace the logic with a
YOLO or custom trained model later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "This controller requires OpenCV and NumPy. Install them for Webots' "
        "Python interpreter with: python -m pip install opencv-python numpy"
    ) from exc

from controller import Robot


# ----------------------------- Configuration -----------------------------

SCAN_DURATION_SECONDS = 120.0
FRAME_PROCESS_INTERVAL_SECONDS = 0.25

# Tune these for your Webots camera distance/object scale.
MIN_CONTOUR_AREA = 80
MAX_CONTOUR_AREA_RATIO = 0.35
MIN_CONFIDENCE = 0.35

# Duplicate matching thresholds. GPS distance is in Webots world units.
DUPLICATE_GPS_DISTANCE = 1.5
DUPLICATE_IMAGE_CENTER_DISTANCE = 70.0

# Nearby-detection grouping for the optional most-polluted-area report.
POLLUTION_GROUP_DISTANCE = 4.0

CAMERA_DEVICE_NAMES = (
    "camera",
    "front camera",
    "camera_front",
    "Camera",
    "cam",
)

GPS_DEVICE_NAMES = (
    "gps",
    "GPS",
    "global position sensor",
)


# ----------------------------- Data Models -----------------------------

BoundingBox = Tuple[int, int, int, int]  # x, y, width, height
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass
class WasteCandidate:
    """A single visual detection from the current camera frame."""

    bbox: BoundingBox
    center: Point2D
    confidence: float
    contour_area: float


@dataclass
class WasteRecord:
    """A unique object remembered across frames."""

    object_id: str
    gps_position: Point3D
    detection_time: float
    bbox: BoundingBox
    image_center: Point2D
    confidence: float
    zone: Optional[str] = None


# ----------------------------- Device Helpers -----------------------------


def get_first_available_device(robot: Robot, names: Sequence[str]):
    """Return the first Webots device that exists from a list of candidate names."""

    for name in names:
        try:
            device = robot.getDevice(name)
            if device is not None:
                return device, name
        except Exception:
            # Webots raises if a device name does not exist.
            continue
    return None, None


def camera_image_to_bgr(camera) -> "np.ndarray":
    """Convert the Webots camera BGRA byte image into an OpenCV BGR image."""

    width = camera.getWidth()
    height = camera.getHeight()
    raw_image = camera.getImage()

    bgra = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def gps_to_tuple(gps) -> Point3D:
    """Read the GPS position as a tuple so it is safe to store in memory."""

    values = gps.getValues()
    return float(values[0]), float(values[1]), float(values[2])


def distance_2d(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_3d(a: Point3D, b: Point3D) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


# ----------------------------- Detection -----------------------------


class WasteDetector:
    """Simple contour-based plastic/waste detector.

    This is deliberately replaceable. A future trained model can return the
    same `WasteCandidate` objects from this class without changing the memory
    or report code.
    """

    def __init__(
        self,
        min_area: int = MIN_CONTOUR_AREA,
        max_area_ratio: float = MAX_CONTOUR_AREA_RATIO,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_confidence = min_confidence

    def detect(self, bgr_image: "np.ndarray") -> List[WasteCandidate]:
        """Detect likely waste objects using HSV color masks and contours."""

        height, width = bgr_image.shape[:2]
        max_area = width * height * self.max_area_ratio

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = self._build_plastic_color_mask(hsv)

        # Remove noisy pixels and connect fragmented bottle/label regions.
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: List[WasteCandidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = w / float(h)
            if not 0.25 <= aspect_ratio <= 4.0:
                continue

            bbox_area = float(w * h)
            fill_ratio = area / bbox_area if bbox_area else 0.0
            if fill_ratio < 0.12:
                continue

            confidence = self._score_candidate(
                contour_area=area,
                bbox_area=bbox_area,
                image_area=float(width * height),
                aspect_ratio=aspect_ratio,
                fill_ratio=fill_ratio,
            )
            if confidence < self.min_confidence:
                continue

            candidates.append(
                WasteCandidate(
                    bbox=(x, y, w, h),
                    center=(x + w / 2.0, y + h / 2.0),
                    confidence=confidence,
                    contour_area=area,
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    @staticmethod
    def _build_plastic_color_mask(hsv: "np.ndarray") -> "np.ndarray":
        """Create masks for common simulated plastic/waste colors."""

        # White/light plastic, foam, or bottle highlights.
        white_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 90, 255]))

        # Blue/green plastic bottles and labels.
        blue_mask = cv2.inRange(hsv, np.array([85, 45, 45]), np.array([135, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 35, 45]), np.array([85, 255, 255]))

        # Bright label colors that often stand out against water.
        red_mask_1 = cv2.inRange(hsv, np.array([0, 55, 70]), np.array([12, 255, 255]))
        red_mask_2 = cv2.inRange(hsv, np.array([165, 55, 70]), np.array([180, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([18, 55, 80]), np.array([35, 255, 255]))

        mask = white_mask
        for extra_mask in (blue_mask, green_mask, red_mask_1, red_mask_2, yellow_mask):
            mask = cv2.bitwise_or(mask, extra_mask)
        return mask

    @staticmethod
    def _score_candidate(
        contour_area: float,
        bbox_area: float,
        image_area: float,
        aspect_ratio: float,
        fill_ratio: float,
    ) -> float:
        """Return a simple 0..1 confidence score for a contour."""

        area_score = min(contour_area / max(image_area * 0.015, 1.0), 1.0)
        fill_score = min(fill_ratio / 0.55, 1.0)

        # Bottles can be long or compact depending on camera angle, so this is
        # intentionally forgiving.
        if 0.45 <= aspect_ratio <= 2.5:
            shape_score = 1.0
        else:
            shape_score = 0.65

        confidence = 0.45 * area_score + 0.35 * fill_score + 0.20 * shape_score
        return max(0.0, min(confidence, 1.0))


# ----------------------------- Memory -----------------------------


class DetectionMemory:
    """Stores unique waste detections and rejects duplicates."""

    def __init__(
        self,
        duplicate_gps_distance: float = DUPLICATE_GPS_DISTANCE,
        duplicate_image_center_distance: float = DUPLICATE_IMAGE_CENTER_DISTANCE,
    ) -> None:
        self.duplicate_gps_distance = duplicate_gps_distance
        self.duplicate_image_center_distance = duplicate_image_center_distance
        self.records: Dict[str, WasteRecord] = {}
        self._next_id = 1

    def add_if_new(
        self,
        candidate: WasteCandidate,
        gps_position: Point3D,
        detection_time: float,
        zone: Optional[str] = None,
    ) -> Tuple[bool, WasteRecord]:
        """Store a candidate only if it does not match a previous detection."""

        existing = self.find_duplicate(candidate, gps_position)
        if existing is not None:
            # Keep the best available bounding box/score for reporting.
            if candidate.confidence > existing.confidence:
                existing.bbox = candidate.bbox
                existing.image_center = candidate.center
                existing.confidence = candidate.confidence
            return False, existing

        object_id = f"Waste_{self._next_id:03d}"
        self._next_id += 1

        record = WasteRecord(
            object_id=object_id,
            gps_position=gps_position,
            detection_time=detection_time,
            bbox=candidate.bbox,
            image_center=candidate.center,
            confidence=candidate.confidence,
            zone=zone,
        )
        self.records[object_id] = record
        return True, record

    def find_duplicate(
        self,
        candidate: WasteCandidate,
        gps_position: Point3D,
    ) -> Optional[WasteRecord]:
        """Find an existing record that is probably the same real object."""

        for record in self.records.values():
            gps_close = (
                distance_3d(gps_position, record.gps_position)
                <= self.duplicate_gps_distance
            )
            image_close = (
                distance_2d(candidate.center, record.image_center)
                <= self.duplicate_image_center_distance
            )

            # GPS closeness is strongest. Image closeness adds protection when
            # the drone sees the object across many consecutive frames.
            if gps_close and image_close:
                return record

        return None

    def all_records(self) -> List[WasteRecord]:
        return list(self.records.values())


# ----------------------------- Flexible Area Hook -----------------------------


def get_zone_for_location(gps_position: Point3D) -> Optional[str]:
    """Hook for a future custom zone/area system.

    Return a string such as "North Bay" or "Zone A" when you add your own
    mapping logic. It returns None now because this controller must not
    hard-code project zones.
    """

    return None


# ----------------------------- Reporting -----------------------------


class ReportGenerator:
    """Creates the final console report after scanning."""

    def __init__(self, group_distance: float = POLLUTION_GROUP_DISTANCE) -> None:
        self.group_distance = group_distance

    def print_report(self, records: Iterable[WasteRecord]) -> None:
        records = list(records)

        print("\n" + "=" * 72)
        print("AquaGuard Final Waste Detection Report")
        print("=" * 72)
        print(f"Total unique waste objects detected: {len(records)}")

        if not records:
            print("No waste objects were detected during this scan.")
            print("=" * 72)
            return

        print("\nDetected object IDs:")
        print(", ".join(record.object_id for record in records))

        print("\nDetection details:")
        for record in records:
            x, y, w, h = record.bbox
            gx, gy, gz = record.gps_position
            zone_text = record.zone if record.zone else "Unassigned"
            print(
                f"- {record.object_id}: "
                f"GPS=({gx:.2f}, {gy:.2f}, {gz:.2f}), "
                f"time={record.detection_time:.2f}s, "
                f"bbox=(x={x}, y={y}, w={w}, h={h}), "
                f"confidence={record.confidence:.2f}, "
                f"zone={zone_text}"
            )

        polluted_area = self._most_polluted_area(records)
        if polluted_area is not None:
            center, count = polluted_area
            print("\nMost polluted nearby area estimate:")
            print(
                f"- center GPS=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                f"grouped detections={count}"
            )

        print("=" * 72)

    def _most_polluted_area(
        self,
        records: List[WasteRecord],
    ) -> Optional[Tuple[Point3D, int]]:
        """Group detections by nearby GPS positions and return the largest group."""

        if not records:
            return None

        best_group: List[WasteRecord] = []
        for seed in records:
            group = [
                candidate
                for candidate in records
                if distance_3d(seed.gps_position, candidate.gps_position)
                <= self.group_distance
            ]
            if len(group) > len(best_group):
                best_group = group

        center = (
            sum(record.gps_position[0] for record in best_group) / len(best_group),
            sum(record.gps_position[1] for record in best_group) / len(best_group),
            sum(record.gps_position[2] for record in best_group) / len(best_group),
        )
        return center, len(best_group)


# ----------------------------- Scan Motion Hook -----------------------------


def perform_scan_motion(robot: Robot, timestep: int) -> None:
    """Project-specific drone movement hook.

    This controller focuses on perception and memory. Add your Webots drone
    flight logic here when you know the motor/device names and desired search
    pattern. Keeping this empty prevents the detector from fighting another
    stabilization controller.
    """

    _ = robot
    _ = timestep


# ----------------------------- Main Controller -----------------------------


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera, camera_name = get_first_available_device(robot, CAMERA_DEVICE_NAMES)
    if camera is None:
        raise RuntimeError(
            "No camera device found. Expected one of: "
            + ", ".join(CAMERA_DEVICE_NAMES)
        )
    camera.enable(timestep)

    gps, gps_name = get_first_available_device(robot, GPS_DEVICE_NAMES)
    if gps is None:
        raise RuntimeError(
            "No GPS device found. Expected one of: "
            + ", ".join(GPS_DEVICE_NAMES)
        )
    gps.enable(timestep)

    print("AquaGuard controller started.")
    print(f"Camera enabled: {camera_name}")
    print(f"GPS enabled: {gps_name}")
    print(f"Scan duration: {SCAN_DURATION_SECONDS:.1f} seconds")

    detector = WasteDetector()
    memory = DetectionMemory()
    reporter = ReportGenerator()

    start_time = robot.getTime()
    last_processed_time = -FRAME_PROCESS_INTERVAL_SECONDS

    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        elapsed_time = current_time - start_time

        perform_scan_motion(robot, timestep)

        if elapsed_time >= SCAN_DURATION_SECONDS:
            break

        if current_time - last_processed_time < FRAME_PROCESS_INTERVAL_SECONDS:
            continue
        last_processed_time = current_time

        gps_position = gps_to_tuple(gps)
        image = camera_image_to_bgr(camera)
        candidates = detector.detect(image)

        for candidate in candidates:
            zone = get_zone_for_location(gps_position)
            is_new, record = memory.add_if_new(
                candidate=candidate,
                gps_position=gps_position,
                detection_time=elapsed_time,
                zone=zone,
            )

            if is_new:
                print(
                    f"New waste detected: {record.object_id} "
                    f"at GPS=({gps_position[0]:.2f}, {gps_position[1]:.2f}, "
                    f"{gps_position[2]:.2f}) "
                    f"bbox={record.bbox} confidence={record.confidence:.2f}"
                )

    reporter.print_report(memory.all_records())


if __name__ == "__main__":
    main()
