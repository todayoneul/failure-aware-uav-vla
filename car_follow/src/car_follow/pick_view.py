"""Config 3 video overlay (display only, docs/demo1.md 8.7): the steps of the flight and the VLM's first pick, drawn
on the first-person view of the demo video. Nothing here feeds back into the flight or its scores.

    1 말 입력     the sentence is given
    2 VLM 선택    the frame with numbered candidate boxes, for as long as the pick really took; then the answer (the
                  chosen number, each number's probability, log(p1/p2) and the ambiguity signal) until the lock, at
                  the end of the fixed budget (first_pick.PICK_DELAY_S)
    3 락온        the picked track becomes the target (highlighted for LOCK_FLASH_S)
    4 추적        the drone follows the track; "놓침" once the tracker has dropped it
    5 고도 회피   the altitude guard raises the follow goal

The numbers are those of the VLM's first question (candidates in score order, vlm.pick); its other two questions
shuffle them. While the drone waits for the lock the cars move, so each candidate box is carried to the current
detection that overlaps it most, for display only.
"""
import math

import cv2
import numpy as np

from . import detector, screen, vlm

STEPS = ("말 입력", "VLM 선택", "락온", "추적", "고도 회피")
LANG = screen.LANGS
SIGNAL = {"clear": "명확", "ambiguous": "모호", "cannot_verify": "확인 불가 (후보 1대)"}
ON, DONE, OFF, WARN = "on", "done", "off", "warn"
INPUT, CHOOSING, ANSWERED, LOCK, TRACK = "input", "choosing", "answered", "lock", "track"
LOCK_FLASH_S = 1.0
CARRY_IOU = 0.3
YELLOW = (0, 255, 255)      # BGR, the colour of vlm.draw_marks
DIM = (200, 200, 200)
MODEL = "Qwen3-VL-4B 4-bit + QLoRA"


def phase(k, record, dt):
    """Which step tick `k` is in, from the pick's record (first_pick.choose plus the flight's updates)."""
    if record is None:
        return INPUT
    if record["lock_tick"] is None:
        return CHOOSING if (k - record["tick"]) * dt < record["seconds"] else ANSWERED
    return LOCK if (k - record["lock_tick"]) * dt < LOCK_FLASH_S else TRACK


def step_states(ph, tracking, climb_m):
    if ph == INPUT:
        return [ON, OFF, OFF, OFF, OFF]
    if ph in (CHOOSING, ANSWERED):
        return [DONE, ON, OFF, OFF, OFF]
    if ph == LOCK:
        return [DONE, DONE, ON, OFF, OFF]
    return [DONE, DONE, DONE, ON if tracking else WARN, ON if climb_m > 0 else OFF]


def fmt_p(p):
    return "<0.001" if 0 < p < 0.001 else f"{p:.3f}"


def answer_lines(record, threshold):
    """The VLM's answer as text lines."""
    n, top = record["candidates"], record["top"]
    probs = record["probs"]
    lines = [f"VLM 선택: {top + 1}번  (후보 {n}대, 확률 {fmt_p(probs[top])})",
             "확률  " + "   ".join(f"{i + 1}: {fmt_p(p)}" for i, p in enumerate(probs))]
    if record["log_ratio"] is None:
        lines.append("log(p1/p2): 후보가 1대라 확인 불가")
    else:
        lr = record["log_ratio"]
        side = "이상" if lr >= threshold else "미만"
        lines.append(f"log(p1/p2) = {lr:.2f}  (기준 {threshold:.2f} {side} → {SIGNAL[record['signal']]})")
    return lines


class PickView:
    """Per flight: the candidate boxes carried for display, and the text renderer."""

    def __init__(self, threshold, delay_s, dt):
        self.text = screen.Text()
        self.threshold, self.delay_s, self.dt = threshold, delay_s, dt
        self.boxes = None

    def carry(self, record, dets):
        """Keep the numbered boxes on their cars while the drone waits for the lock."""
        if record is None:
            return
        if self.boxes is None:
            self.boxes = [tuple(b) for b in record["boxes"]]
            return
        carried = []
        for b in self.boxes:
            best = max(dets, key=lambda d: detector.iou(d.box, b), default=None)
            ok = best is not None and detector.iou(best.box, b) >= CARRY_IOU
            carried.append(tuple(best.box) if ok else b)
        self.boxes = carried

    def draw(self, image, k, record, tracking, climb_m):
        """Draw onto the first-person view (BGR, in place)."""
        h, w = image.shape[:2]
        ph = phase(k, record, self.dt)
        lines = []
        if ph in (CHOOSING, ANSWERED):
            if ph == CHOOSING:
                body = [f"VLM 선택 중  ({MODEL})",
                        f"후보 {record['candidates']}대에 번호를 붙인 화면과 문장만 봄",
                        "번호를 섞어 3번 묻고 확률을 평균"]
            else:
                lock_in = (record["tick"] + int(round(self.delay_s / self.dt)) - k) * self.dt
                body = answer_lines(record, self.threshold) + [
                    f"계산 {record['seconds']:.2f} s, 예산 {self.delay_s:.2f} s: 락온까지 {max(0.0, lock_in):.2f} s"]
            self.panel(image, lines, body, avoid=self.boxes or ())
            self.marks(image, lines, record["top"] if ph == ANSWERED else None)
        elif ph in (LOCK, TRACK):
            body = [f"VLM 선택 {record['top'] + 1}번 (후보 {record['candidates']}대) · "
                    + (f"log(p1/p2) {record['log_ratio']:.2f}" if record["log_ratio"] is not None else "log(p1/p2) 없음")
                    + f" · {SIGNAL[record['signal']]}"]
            if ph == TRACK and not tracking:
                body += ["목표를 놓침: 마지막 추정 위치 뒤에서 대기", "(다시 찾기는 데모 1 범위 밖)"]
            self.panel(image, lines, body, size=14)
            if ph == LOCK:
                self.banner(image, lines, "락온", w)
        elif ph == INPUT:
            self.panel(image, lines, ["말 입력: 첫 화면에서 탐지된 후보를 기다림"])
        self.strip(image, lines, step_states(ph, tracking, climb_m), climb_m, (12, h - 54))
        return self.text.draw(image, lines)

    def marks(self, image, lines, chosen):
        h, w = image.shape[:2]
        for i, box in enumerate(self.boxes or []):
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            if chosen is None:
                color, thick = YELLOW, 2
            else:
                color, thick = (screen.GREEN, 3) if i == chosen else (DIM, 2)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
            tag = str(i + 1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = min(max(0, x0), w - tw - 6)
            ty = y0 - 4 if y0 - th - 8 >= 0 else min(h - 2, y1 + th + 6)
            fill = screen.GREEN if chosen == i else (255, 255, 255)
            cv2.rectangle(image, (tx, ty - th - 4), (tx + tw + 6, ty + 2), fill, -1)
            cv2.rectangle(image, (tx, ty - th - 4), (tx + tw + 6, ty + 2), (0, 0, 0), 1)
            cv2.putText(image, tag, (tx + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            if chosen == i:
                lines.append(("선택", (tx + tw + 10, ty - th - 6), 15, screen.GREEN, True))

    def panel(self, image, lines, body, y=34, size=15, avoid=()):
        """A dark panel of text lines at the top left, or the top right when it covers less of the `avoid` boxes
        (the numbered candidates and their tags)."""
        h, w = image.shape[:2]
        step = size + 7
        width = min(w - 20, max(int(self.text.font(size, True).getlength(t)) for t in body) + 16)
        height = step * len(body) + 8

        def covered(x):
            return sum(max(0.0, min(x + width, b[2]) - max(x, b[0])) * max(0.0, min(y + height, b[3]) - max(y, b[1] - 26))
                       for b in avoid)

        x = min((10, w - width - 10), key=covered)
        y1, x1 = min(h, y + height), min(w, x + width)
        image[y:y1, x:x1] = (image[y:y1, x:x1].astype(np.float32) * 0.35).astype(np.uint8)
        for i, t in enumerate(body):
            lines.append((t, (x + 8, y + 4 + i * step), size, screen.WHITE, i == 0))

    def banner(self, image, lines, label, w):
        size = 28
        tw = int(self.text.font(size, True).getlength(label)) + 28
        x0, y0 = (w - tw) // 2, 70
        cv2.rectangle(image, (x0, y0), (x0 + tw, y0 + size + 14), screen.GREEN, -1)
        lines.append((label, (x0 + 14, y0 + 4), size, (0, 0, 0), True))

    def strip(self, image, lines, states, climb_m, xy):
        x, y = xy
        size, hgt = 14, 22
        font = self.text.font(size, True)
        for i, (name, st) in enumerate(zip(STEPS, states)):
            label = f"{i + 1} {name}"
            if i == 3 and st == WARN:
                label = "4 추적 놓침"
            if i == 4 and st == ON:
                label = f"5 고도 회피 +{climb_m:.1f} m"
            tw = int(font.getlength(label)) + 16
            if st == ON:
                cv2.rectangle(image, (x, y), (x + tw, y + hgt), screen.GREEN, -1)
                color = (0, 0, 0)
            elif st == WARN:
                cv2.rectangle(image, (x, y), (x + tw, y + hgt), screen.ORANGE, -1)
                color = (0, 0, 0)
            elif st == DONE:
                cv2.rectangle(image, (x, y), (x + tw, y + hgt), (70, 70, 70), -1)
                color = (210, 210, 210)
            else:
                cv2.rectangle(image, (x, y), (x + tw, y + hgt), (110, 110, 110), 1)
                color = (140, 140, 140)
            lines.append((label, (x + 8, y + 2), size, color, True))
            x += tw + 8


def still(bgr, record, threshold, sentence, lang):
    """What the VLM saw in its first question (vlm.draw_marks, numbers 1..n) with its answer and the sentence."""
    image = vlm.draw_marks(bgr, record["boxes"], list(range(1, record["candidates"] + 1)))
    text = screen.Text()
    h = image.shape[0]
    lines = []
    view = PickView(threshold, 0.0, 1.0)
    view.text = text
    body = [f"VLM 첫 지정, 첫 질문의 화면 ({MODEL})"] + answer_lines(record, threshold) + [
        f"계산 {record['seconds']:.2f} s (번호를 섞어 3번 물은 평균)"]
    view.panel(image, lines, body, y=10, avoid=record["boxes"])
    screen.band(image, h - 32, h)
    lines.append((f'지시 ({LANG[lang]})  "{sentence}"', (12, h - 28), 18, screen.WHITE, True))
    return text.draw(image, lines)


def climb(offset, active_m):
    """The climb shown on the strip: the guard's applied offset once it counts as active (scorer.ACTIVE_M)."""
    return float(offset) if math.isfinite(offset) and offset > active_m else 0.0
