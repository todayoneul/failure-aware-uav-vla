"""Demo 1 screen (docs/demo1_plan.md 2.4, docs/demo1.md 3): one frame made of three views.

    +----------------------------+-------------+
    |                            | chase view  |
    |  first person (960 x 540)  | drone + car |
    |  lock-on box, sentence     +-------------+
    |                            | top-down map|
    +----------------------------+-------------+

- First person: the drone camera, with the lock-on box on the followed car, its track label and distance, and the
  sentence that named it.
- Chase view: a camera behind and above the drone. CARLA has no drone model, so a four-rotor drone is drawn onto
  the image where the drone is (`draw_drone`, the one function to replace when a real 3D model comes).
- Top-down map: roads from the map's OpenDRIVE file and every car from the simulator's ground truth, drawn with
  OpenCV (no GPU).

Geometry follows CARLA: metres, x forward, y right, z up (left-handed), angles in degrees.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIRST = (960, 540)
SIDE = (480, 270)
SIZE = (FIRST[0] + SIDE[0], FIRST[1])
FONT_PATHS = (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf"))

GREEN = (60, 220, 60)       # BGR: the followed car
ORANGE = (0, 150, 255)      # look-alikes
GRAY = (170, 170, 170)      # other cars
CYAN = (230, 200, 0)        # the drone
WHITE = (255, 255, 255)
LANGS = {"en": "영어", "ko": "한국어"}


# --- geometry ---

def rotation(pitch, yaw, roll):
    """3 x 3 matrix whose columns are a CARLA (Unreal) rotation's x, y, z axes in world coordinates."""
    p, y, r = (math.radians(a) for a in (pitch, yaw, roll))
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    x_axis = (cp * cy, cp * sy, sp)
    y_axis = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    z_axis = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return np.array([x_axis, y_axis, z_axis]).T


def box_vertices(position, rot, center, extent):
    """The 8 corners (8 x 3, world) of a car's bounding box. `rot` is (pitch, yaw, roll), `center` the box centre
    in the car's frame and `extent` its half sizes."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    local = np.asarray(center, dtype=float) + signs * np.asarray(extent, dtype=float)
    return np.asarray(position, dtype=float) + local @ rotation(*rot).T


class Camera:
    """A pinhole camera at `position` looking along (yaw, pitch), no roll, as CARLA's RGB camera renders."""

    def __init__(self, position, yaw, pitch, width, height, fov):
        self.position = np.asarray(position, dtype=float)
        y, p = math.radians(yaw), math.radians(pitch)
        self.forward = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        self.right = np.array([-math.sin(y), math.cos(y), 0.0])
        self.up = np.cross(self.forward, self.right)
        self.width, self.height = width, height
        self.focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    def project(self, points):
        """Pixel coordinates (N x 2) and depth along the optical axis (N) of world points (N x 3)."""
        d = np.atleast_2d(np.asarray(points, dtype=float)) - self.position
        f, r, u = d @ self.forward, d @ self.right, d @ self.up
        safe = np.where(f > 1e-6, f, 1e-6)
        return np.c_[self.focal * r / safe + self.width / 2.0, -self.focal * u / safe + self.height / 2.0], f


def look_at(eye, point):
    d = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def chase_pose(drone_position, drone_yaw, aim_point, back=4.0, up=7.0, side=2.5):
    """Where the chase camera sits (`back` metres behind the drone along its heading, `up` metres above, `side`
    metres to its right) and the (yaw, pitch) that looks half way, by angle, between the drone and the car.
    The camera is high enough that the drone sits below the car in the image instead of in front of it."""
    y = math.radians(drone_yaw)
    drone = np.asarray(drone_position, dtype=float)
    eye = drone + np.array([-back * math.cos(y) - side * math.sin(y), -back * math.sin(y) + side * math.cos(y), up])
    to_drone, to_car = drone - eye, np.asarray(aim_point, dtype=float) - eye
    mid = to_drone / np.linalg.norm(to_drone) + to_car / np.linalg.norm(to_car)
    return eye, look_at(eye, eye + mid)


# --- text (Pillow, so the Korean sentence can be drawn) ---

class Text:
    def __init__(self):
        self.fonts = {}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            path = FONT_PATHS[1 if bold else 0]
            self.fonts[key] = ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()
        return self.fonts[key]

    def draw(self, image, lines):
        """Draw [(text, (x, y), size, bgr colour, bold)] onto a BGR image in place."""
        pil = Image.fromarray(image[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        for text, xy, size, color, bold in lines:
            d.text(xy, text, font=self.font(size, bold), fill=tuple(int(c) for c in color[::-1]))
        image[:, :, :] = np.asarray(pil)[:, :, ::-1]
        return image


def band(image, y0, y1, alpha=0.55):
    """Darken rows y0..y1 so text on them reads over any scene."""
    image[y0:y1] = (image[y0:y1].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


# --- first person ---

def lock_box(image, box, color=GREEN, corner=0.25, thickness=2):
    """Corner brackets around `box` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cx, cy = max(6, int((x1 - x0) * corner)), max(6, int((y1 - y0) * corner))
    for (ax, ay), (dx, dy) in (((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x0, y1), (1, -1)), ((x1, y1), (-1, -1))):
        cv2.line(image, (ax, ay), (ax + dx * cx, ay), color, thickness, cv2.LINE_AA)
        cv2.line(image, (ax, ay), (ax, ay + dy * cy), color, thickness, cv2.LINE_AA)


def first_person(image, text, box, label, distance, sentence_en, sentence_ko, status, given=None):
    """The drone camera view with the lock-on box (or None), its label and distance, the sentence and a status line.
    `given` ("en" or "ko"): only the sentence of that language, on the bottom line (config 3 flies one language;
    the line above is left free for its overlay, src/car_follow/pick_view.py)."""
    h, w = image.shape[:2]
    cv2.line(image, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), WHITE, 1, cv2.LINE_AA)
    cv2.line(image, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), WHITE, 1, cv2.LINE_AA)
    lines = []
    if box is not None:
        lock_box(image, box)
        x0, y0 = int(box[0]), int(box[1])
        tag = f"{label}  {distance:.1f} m"
        tw = int(text.font(15, True).getlength(tag)) + 10
        ty = max(0, y0 - 24)
        cv2.rectangle(image, (x0, ty), (x0 + tw, ty + 20), GREEN, -1)
        lines.append((tag, (x0 + 5, ty + 1), 15, (0, 0, 0), True))
    band(image, 0, 28)
    band(image, h - 58, h)
    lines.append((status, (10, 5), 15, WHITE, False))
    if given is None:
        lines += [(f'"{sentence_en}"', (12, h - 54), 18, WHITE, True),
                  (sentence_ko, (12, h - 28), 17, (220, 220, 220), False)]
    else:
        sentence = sentence_en if given == "en" else sentence_ko
        lines.append((f'지시 ({LANGS[given]})  "{sentence}"', (12, h - 28), 18, WHITE, True))
    return text.draw(image, lines)


# --- chase view: the drone drawn onto the image ---

DRONE_ARM_M = 0.45      # body centre to rotor centre
DRONE_ROTOR_M = 0.22    # rotor radius
DRONE_BODY_M = 0.16     # half size of the body


def drone_parts(position, yaw):
    """World points of a four-rotor drone (X layout) at `position` heading `yaw`: body corners, rotor centres,
    rotor rims (4 x 16) and a nose point."""
    y = math.radians(yaw)
    rot = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    p = np.asarray(position, dtype=float)
    to_world = lambda pts: p + np.asarray(pts, dtype=float) @ rot.T  # noqa: E731
    a = DRONE_ARM_M / math.sqrt(2.0)
    rotors = np.array([[a, a, 0.05], [a, -a, 0.05], [-a, -a, 0.05], [-a, a, 0.05]])
    b = DRONE_BODY_M
    body = np.array([[b * 1.3, 0, 0], [0, b, 0], [-b, 0, 0], [0, -b, 0]])
    ang = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    circle = np.c_[DRONE_ROTOR_M * np.cos(ang), DRONE_ROTOR_M * np.sin(ang), np.zeros(16)]
    return {"body": to_world(body), "rotors": to_world(rotors),
            "rims": [to_world(c + circle) for c in rotors], "nose": to_world([[b * 2.2, 0, 0]])[0]}


def draw_drone(image, camera, position, yaw):
    """Draw the drone onto a chase camera image. Returns False (and draws nothing) if it is behind the camera."""
    parts = drone_parts(position, yaw)
    centre, depth = camera.project([position])
    if depth[0] <= 0.5:
        return False
    px = lambda pts: np.round(camera.project(pts)[0]).astype(np.int32)  # noqa: E731
    c = tuple(px([position])[0])
    rotor_px = px(parts["rotors"])
    overlay = image.copy()
    for i, rim in enumerate(parts["rims"]):
        cv2.fillPoly(overlay, [px(rim)], (225, 225, 225) if i < 2 else (200, 200, 200), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, image)
    for i, r in enumerate(rotor_px):
        cv2.line(image, c, tuple(r), (40, 40, 40), 3, cv2.LINE_AA)
        cv2.polylines(image, [px(parts["rims"][i])], True, (60, 60, 230) if i < 2 else (40, 40, 40), 1,
                      cv2.LINE_AA)
    cv2.fillPoly(image, [px(parts["body"])], (30, 30, 30), cv2.LINE_AA)
    cv2.circle(image, tuple(px([parts["nose"]])[0]), 2, (60, 60, 230), -1, cv2.LINE_AA)
    return True


def chase_view(image, text, camera, drone_position, drone_yaw):
    draw_drone(image, camera, drone_position, drone_yaw)
    band(image, 0, 22)
    return text.draw(image, [("chase view", (8, 3), 13, WHITE, False)])


# --- top-down map ---

class MapRaster:
    """Roads of a CARLA map, drawn once from its OpenDRIVE file at `res` metres per pixel (north-up as CARLA's
    x right, y down)."""

    def __init__(self, image, origin, res):
        self.image, self.origin, self.res = image, np.asarray(origin, dtype=float), res

    @classmethod
    def from_opendrive(cls, xodr_path, name="map", res=0.25, spacing=1.0, margin=20.0):
        import carla
        world_map = carla.Map(name, Path(xodr_path).read_text(encoding="utf-8"))
        wps = world_map.generate_waypoints(spacing)
        xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
        widths = np.array([w.lane_width for w in wps])
        origin = xy.min(0) - margin
        size = np.ceil((xy.max(0) + margin - origin) / res).astype(int)
        image = np.full((size[1], size[0], 3), (52, 66, 48), np.uint8)
        pix = np.round((xy - origin) / res).astype(int)
        for (u, v), wdt in zip(pix, widths):
            cv2.circle(image, (int(u), int(v)), max(1, int(round(wdt / 2.0 / res))), (92, 92, 92), -1)
        return cls(image, origin, res)

    @classmethod
    def cached(cls, xodr_path, cache_dir, name="map", res=0.25):
        cache_dir = Path(cache_dir)
        png, meta = cache_dir / f"{name}_{res}.png", cache_dir / f"{name}_{res}.npy"
        if png.exists() and meta.exists():
            return cls(cv2.imread(str(png)), np.load(meta), res)
        raster = cls.from_opendrive(xodr_path, name, res)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), raster.image)
        np.save(meta, raster.origin)
        return raster

    def to_pixel(self, xy):
        return (np.asarray(xy, dtype=float)[..., :2] - self.origin) / self.res

    def to_world(self, uv):
        return np.asarray(uv, dtype=float) * self.res + self.origin


def top_map(raster, text, centre, cars, drone_position, drone_yaw, goal=None, trails=(), size=SIDE, fov=90.0):
    """The map window of `size` pixels around `centre` (world x, y).

    `cars` are (position, (pitch, yaw, roll), extent, kind) with kind "target", "lookalike", "distractor" (a stage 3
    failure trial's, drawn as a look-alike) or "background";
    `trails` are (N x 3 points, bgr colour)."""
    w, h = size
    corner = raster.to_pixel(centre) - np.array([w / 2.0, h / 2.0])
    u0, v0 = int(round(corner[0])), int(round(corner[1]))
    img = np.full((h, w, 3), (52, 66, 48), np.uint8)
    H, W = raster.image.shape[:2]
    su0, sv0, su1, sv1 = max(u0, 0), max(v0, 0), min(u0 + w, W), min(v0 + h, H)
    if su1 > su0 and sv1 > sv0:
        img[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = raster.image[sv0:sv1, su0:su1]
    off = np.array([u0, v0], dtype=float)
    px = lambda pts: np.round(raster.to_pixel(pts) - off).astype(np.int32)  # noqa: E731
    for points, color in trails:
        if len(points) > 1:
            cv2.polylines(img, [px(points)], False, color, 1, cv2.LINE_AA)
    order = {"background": 0, "lookalike": 1, "distractor": 1, "target": 2}
    for pos, rot, extent, kind in sorted(cars, key=lambda c: order[c[3]]):
        y = math.radians(rot[1])
        f, r = np.array([math.cos(y), math.sin(y)]), np.array([-math.sin(y), math.cos(y)])
        ex, ey = max(extent[0], 1.0), max(extent[1], 0.6)
        quad = [np.asarray(pos[:2]) + sx * ex * f + sy * ey * r for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        color = {"target": GREEN, "lookalike": ORANGE, "distractor": ORANGE, "background": GRAY}[kind]
        cv2.fillPoly(img, [px(quad)], color, cv2.LINE_AA)
    if goal is not None:
        g = tuple(px([goal])[0])
        cv2.drawMarker(img, g, WHITE, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    d = tuple(px([drone_position])[0])
    for side in (-1, 1):
        a = math.radians(drone_yaw + side * fov / 2.0)
        tip = np.asarray(drone_position[:2]) + 30.0 * np.array([math.cos(a), math.sin(a)])
        cv2.line(img, d, tuple(px([tip])[0]), CYAN, 1, cv2.LINE_AA)
    for k in range(4):
        a = math.radians(drone_yaw + 45.0 + 90.0 * k)
        tip = np.asarray(drone_position[:2]) + 2.2 * np.array([math.cos(a), math.sin(a)])
        cv2.line(img, d, tuple(px([tip])[0]), CYAN, 2, cv2.LINE_AA)
    cv2.circle(img, d, 3, CYAN, -1, cv2.LINE_AA)
    bar = int(round(20.0 / raster.res))
    cv2.line(img, (w - bar - 12, h - 12), (w - 12, h - 12), WHITE, 2)
    band(img, 0, 22)
    return text.draw(img, [("top-down map", (8, 3), 13, WHITE, False), ("20 m", (w - bar // 2 - 26, h - 32), 12,
                                                                      WHITE, False)])


GROUND_TRUTH = object()   # Screen.frame: draw the followed car's projected 3D box


class Screen:
    """Builds the frames of one flight: keeps the trails and the text renderer between ticks."""

    def __init__(self, raster, sentence_en, sentence_ko, trail_s=10.0, hz=20.0, fov=90.0, given=None):
        self.raster, self.text, self.fov = raster, Text(), fov
        self.sentence, self.given = (sentence_en, sentence_ko), given
        self.trail_len = int(trail_s * hz)
        self.trails = {"target": [], "drone": []}

    def first_camera(self, position, yaw, pitch):
        return Camera(position, yaw, pitch, *FIRST, self.fov)

    def frame(self, first_img, chase_img, chase_cam, drone_position, drone_yaw, drone_pitch, cars, target_index,
              label, status, goal=None, track_box=GROUND_TRUTH, distance=None):
        """One composed frame. `cars` are (position, (pitch, yaw, roll), box centre, box extent, kind) at this tick;
        `target_index` picks the followed car in it. The lock-on box is the followed car's projected 3D box unless
        `track_box` gives the tracker's box (or None: no box this tick); `distance` overrides the shown distance."""
        tpos, trot, tcen, text_, _ = cars[target_index]
        cam = self.first_camera(drone_position, drone_yaw, drone_pitch)
        uv, depth = cam.project(box_vertices(tpos, trot, tcen, text_))
        box = None
        if (depth > 0.1).all():
            x0, y0 = uv.min(0)
            x1, y1 = uv.max(0)
            if x1 > 0 and y1 > 0 and x0 < FIRST[0] and y0 < FIRST[1]:
                box = (x0, y0, x1, y1)
        if track_box is not GROUND_TRUTH:
            box = track_box
        if distance is None:
            centre_point = box_vertices(tpos, trot, tcen, (0, 0, 0))[0]
            distance = float(np.linalg.norm(centre_point - np.asarray(drone_position)))
        first = first_person(first_img, self.text, box, label, distance, *self.sentence, status, given=self.given)
        chase = chase_view(chase_img, self.text, chase_cam, drone_position, drone_yaw)
        for key, p in (("target", tpos), ("drone", drone_position)):
            self.trails[key] = (self.trails[key] + [np.asarray(p, dtype=float)])[-self.trail_len:]
        centre = (np.asarray(tpos[:2]) + np.asarray(drone_position[:2])) / 2.0
        top = top_map(self.raster, self.text, centre, [(p, r, e, k) for p, r, _, e, k in cars], drone_position,
                      drone_yaw, goal, [(np.array(self.trails["target"]), GREEN), (np.array(self.trails["drone"]), CYAN)],
                      fov=self.fov)
        return compose(first, chase, top)


def compose(first, chase, top):
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    frame[:, :FIRST[0]] = first
    frame[:SIDE[1], FIRST[0]:] = chase
    frame[SIDE[1]:, FIRST[0]:] = top
    cv2.line(frame, (FIRST[0], 0), (FIRST[0], SIZE[1]), (20, 20, 20), 2)
    cv2.line(frame, (FIRST[0], SIDE[1]), (SIZE[0], SIDE[1]), (20, 20, 20), 2)
    return frame
