from __future__ import annotations

import os
import io
import time
import base64
import socket
import threading
import subprocess
import shutil
import re
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PIL import Image
import qrcode
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, abort

# Google Drive
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Email (SMTP)
import smtplib
from email.message import EmailMessage


# =========================
# Paths / Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_PORT = 8000
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
FALLBACK_FRAME_PATH = os.path.join(BASE_DIR, "frame.png")  # optional

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cameras
PHONE_CAMERA_INDEX = 1
DSLR_CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 1280, 720
JPEG_QUALITY_STREAM = 80
JPEG_QUALITY_SAVE = 90  # turunin dikit biar upload lebih ringan

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]

# Green key (slot detection + knock-out alpha) kalau frame slot-nya hijau
GREEN_KEY_HEX = "#01e678"

# Drive (OAuth Desktop)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_CREDENTIALS_JSON = os.path.join(BASE_DIR, "credentials.json")
DRIVE_TOKEN_JSON = os.path.join(BASE_DIR, "token.json")

# ✅ default parent folder ID kamu (bisa override via env DRIVE_PARENT_ID)
DEFAULT_PARENT_ID = "1Yg89B8oHM66Sm7uy3mLhhzik8-5oXCEA"
DRIVE_PARENT_ID = os.environ.get("DRIVE_PARENT_ID", DEFAULT_PARENT_ID).strip()

# kalau drive kamu shared drive / workspace tertentu, bisa set 1
DRIVE_SUPPORTS_ALL_DRIVES = os.environ.get("DRIVE_SUPPORTS_ALL_DRIVES", "0") == "1"

# Email SMTP settings (default Gmail)
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", "Photobooth").strip()

EVENT_TITLE = os.environ.get("EVENT_TITLE", "Hasil Photobooth").strip()


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

ACTIVE_SOURCE = "phone"
ACTIVE_FRAME: str | None = None

_drive_service = None
_drive_lock = threading.Lock()


# =========================
# In-memory sessions
# =========================
SESSION_JOBS: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()
SESSION_MAX = 300


def _prune_sessions():
    with SESSION_LOCK:
        if len(SESSION_JOBS) <= SESSION_MAX:
            return
        keys = list(SESSION_JOBS.keys())
        for k in keys[: max(0, len(keys) - SESSION_MAX)]:
            SESSION_JOBS.pop(k, None)


# =========================
# Drive helpers
# =========================
def get_drive_service():
    global _drive_service
    with _drive_lock:
        if _drive_service is not None:
            return _drive_service

        if not os.path.exists(DRIVE_CREDENTIALS_JSON):
            raise FileNotFoundError("Missing credentials.json (OAuth Desktop). Put it next to app.py")

        creds = None
        if os.path.exists(DRIVE_TOKEN_JSON):
            creds = Credentials.from_authorized_user_file(DRIVE_TOKEN_JSON, DRIVE_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(DRIVE_CREDENTIALS_JSON, DRIVE_SCOPES)
                creds = flow.run_local_server(port=0)
            with open(DRIVE_TOKEN_JSON, "w", encoding="utf-8") as f:
                f.write(creds.to_json())

        _drive_service = build("drive", "v3", credentials=creds)
        return _drive_service


def drive_set_public_reader(service, file_id: str):
    body = {"type": "anyone", "role": "reader"}
    service.permissions().create(
        fileId=file_id,
        body=body,
        fields="id",
        supportsAllDrives=DRIVE_SUPPORTS_ALL_DRIVES,
    ).execute()


def drive_folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"


def drive_file_view_link(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


def drive_file_direct_download_link(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def sanitize_participant_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r'[\\/:*?"<>|]+', "-", name)
    return name[:60] if name else ""


def build_folder_name(participant_name: str | None, fallback: str, session_id: str) -> str:
    pn = sanitize_participant_name(participant_name or "")
    if pn:
        return f"{pn}__{session_id}"
    return fallback


def drive_rename(service, file_id: str, new_name: str):
    service.files().update(
        fileId=file_id,
        body={"name": new_name},
        fields="id,name",
        supportsAllDrives=DRIVE_SUPPORTS_ALL_DRIVES,
    ).execute()


def drive_create_session_folder(service, name: str) -> tuple[str, str]:
    meta: dict[str, Any] = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if DRIVE_PARENT_ID:
        meta["parents"] = [DRIVE_PARENT_ID]

    folder = service.files().create(
        body=meta,
        fields="id",
        supportsAllDrives=DRIVE_SUPPORTS_ALL_DRIVES,
    ).execute()

    folder_id = folder["id"]
    try:
        drive_set_public_reader(service, folder_id)
    except Exception:
        # bisa gagal kalau workspace policy blok public sharing
        pass

    return folder_id, drive_folder_link(folder_id)


def drive_upload_file(service, local_path: str, drive_name: str, parent_folder_id: str, mime_type: str):
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    meta = {"name": drive_name, "parents": [parent_folder_id]}

    f = service.files().create(
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=DRIVE_SUPPORTS_ALL_DRIVES,
    ).execute()

    file_id = f["id"]
    drive_set_public_reader(service, file_id)

    return {
        "id": file_id,
        "view_url": drive_file_view_link(file_id),         # ✅ preview page
        "download_url": drive_file_direct_download_link(file_id),
    }


# =========================
# QR helper
# =========================
def make_qr_data_url(url: str) -> str:
    qr_img = qrcode.make(url).convert("RGB")
    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# =========================
# Camera Manager
# =========================
class CameraManager:
    def __init__(self, index: int, label: str):
        self.index = index
        self.label = label
        self.cap = None
        self.backend_name = None

        self.latest = None
        self.lock = threading.Lock()

        self.running = True
        self.last_frame_ts = 0.0

        self._open_camera_no_throw()
        self.t = threading.Thread(target=self._grab_loop, daemon=True)
        self.t.start()

    def _apply_settings(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

    def _try_open(self):
        for name, be in BACKENDS:
            cap = cv2.VideoCapture(self.index, be)
            if cap.isOpened():
                self._apply_settings(cap)
                self.backend_name = name
                print(f"[OK] {self.label} opened: index={self.index} backend={name}")
                return cap
            cap.release()
        return None

    def _open_camera_no_throw(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.cap = self._try_open()
        if self.cap is None:
            print(f"[WARN] {self.label} not available yet. (Reconnect later)")

    def reconnect(self):
        self._open_camera_no_throw()
        with self.lock:
            self.latest = None
            self.last_frame_ts = 0.0

    def _grab_loop(self):
        printed_first = False
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            ok, fr = self.cap.read()
            if ok:
                with self.lock:
                    self.latest = fr
                    self.last_frame_ts = time.time()
                if not printed_first:
                    print(f"[OK] {self.label} first frame received")
                    printed_first = True
            else:
                time.sleep(0.02)

    def get_latest_frame(self):
        with self.lock:
            if self.latest is None:
                return None
            return self.latest.copy()

    def camera_status(self):
        fr = self.get_latest_frame()
        if fr is None:
            return {"ok": False, "reason": "no_frame", "backend": self.backend_name, "index": self.index}

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        ok = not (mean < 10.0 and std < 8.0)

        return {
            "ok": ok,
            "mean": mean,
            "std": std,
            "backend": self.backend_name,
            "index": self.index,
            "last_frame_age_sec": max(0.0, time.time() - self.last_frame_ts) if self.last_frame_ts else None
        }


phone_cam = CameraManager(PHONE_CAMERA_INDEX, "PHONE_CAM")
dslr_cam = CameraManager(DSLR_CAMERA_INDEX, "DSLR_CAM")


def get_cam(source: str) -> CameraManager:
    return phone_cam if source == "phone" else dslr_cam


# =========================
# Utils
# =========================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def open_mobile_devices_settings():
    subprocess.Popen(["cmd", "/c", "start", "ms-settings:mobile-devices"], shell=False)


def hex_to_rgb_tuple(hex_color: str):
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Hex color must be 6 chars, e.g. #01e678")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def trim_black_borders(img_rgb: Image.Image, thresh: int = 12) -> Image.Image:
    arr = np.array(img_rgb)
    gray = arr.mean(axis=2)
    mask = gray > thresh
    if not mask.any():
        return img_rgb
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    pad = 2
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1], x2 + pad); y2 = min(arr.shape[0], y2 + pad)
    return img_rgb.crop((x1, y1, x2, y2))


def cover_resize_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    iw, ih = img.size
    scale = max(target_w / max(1, iw), target_h / max(1, ih))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - target_w) // 2
    top = (nh - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


# =========================
# Slot detection (alpha OR green)
# =========================
def detect_holes_from_alpha(frame_rgba: Image.Image, alpha_thresh: int = 32):
    W, H = frame_rgba.size
    alpha = np.array(frame_rgba.getchannel("A"))

    holes_bin = (alpha < alpha_thresh).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    holes_bin = cv2.morphologyEx(holes_bin, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(holes_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.02 * W * H:
            continue

        touches = (x <= 1) or (y <= 1) or (x + w >= W - 1) or (y + h >= H - 1)
        if touches and area > 0.65 * W * H:
            continue

        cnt_local = cnt.copy()
        cnt_local[:, 0, 0] -= x
        cnt_local[:, 0, 1] -= y
        mask_local = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_local, [cnt_local], -1, 255, thickness=-1)

        cx = (x + w / 2) / float(W)
        cy = (y + h / 2) / float(H)

        holes.append({
            "bbox_px": (x, y, x + w, y + h),
            "mask": Image.fromarray(mask_local, mode="L"),
            "area": area,
            "cx": cx,
            "cy": cy,
        })

    holes = sorted(holes, key=lambda h: (h["cy"], h["cx"]))
    return holes


def detect_holes_from_green(frame_rgba: Image.Image, key_hex: str = GREEN_KEY_HEX, tol: int = 40):
    W, H = frame_rgba.size
    key = np.array(hex_to_rgb_tuple(key_hex), dtype=np.int16)

    arr = np.array(frame_rgba.convert("RGBA")).astype(np.int16)
    rgb = arr[..., :3]
    dist = np.linalg.norm(rgb - key, axis=2)

    mask = (dist <= tol).astype(np.uint8) * 255
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.03 * W * H:
            continue

        cnt_local = cnt.copy()
        cnt_local[:, 0, 0] -= x
        cnt_local[:, 0, 1] -= y
        mask_local = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_local, [cnt_local], -1, 255, thickness=-1)

        cx = (x + w / 2) / float(W)
        cy = (y + h / 2) / float(H)

        holes.append({
            "bbox_px": (x, y, x + w, y + h),
            "mask": Image.fromarray(mask_local, mode="L"),
            "area": area,
            "cx": cx,
            "cy": cy,
        })

    holes = sorted(holes, key=lambda h: (h["cy"], h["cx"]))

    # knock out alpha in green areas
    out_arr = np.array(frame_rgba.convert("RGBA")).astype(np.uint8)
    full = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(full, contours, -1, 255, thickness=-1)
    out_arr[full == 255, 3] = 0
    out_arr[full == 255, 0:3] = 0
    knocked = Image.fromarray(out_arr, mode="RGBA")

    return holes, knocked


def detect_slots_auto(frame_rgba: Image.Image):
    holes = detect_holes_from_alpha(frame_rgba, alpha_thresh=32)
    if holes:
        return holes, frame_rgba
    holes2, knocked = detect_holes_from_green(frame_rgba, key_hex=GREEN_KEY_HEX, tol=40)
    return holes2, knocked


# =========================
# Compose photo
# =========================
def compose_strip_photo(frame_rgba: Image.Image, shots_rgb: list[Image.Image], holes):
    W, H = frame_rgba.size
    base = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    for i, hinfo in enumerate(holes):
        if i >= len(shots_rgb):
            break
        x1, y1, x2, y2 = hinfo["bbox_px"]
        sw, sh = (x2 - x1), (y2 - y1)

        shot = trim_black_borders(shots_rgb[i])
        shot = cover_resize_center_crop(shot, sw, sh).convert("RGBA")
        base.paste(shot, (x1, y1), hinfo["mask"])

    return Image.alpha_composite(base, frame_rgba).convert("RGB")


# =========================
# Video compose + transcode
# =========================
def find_ffmpeg() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    local = os.path.join(BASE_DIR, "_ffmpeg", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return None


def transcode_to_h264_mp4(in_path: str, out_path: str, fps: int = 20) -> str | None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def compose_strip_live_video(frame_rgba: Image.Image, clip_paths: list[str], holes, out_path: str, fps_out: int = 20) -> str:
    W, H = frame_rgba.size
    caps = []
    try:
        for p in clip_paths:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open clip: {p}")
            caps.append(cap)

        counts = []
        for cap in caps:
            c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            counts.append(c if c > 0 else 999999)
        nframes = max(1, min(counts))

        # write mp4v first (fast), then try transcode
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
        if not vw.isOpened():
            out_path_avi = os.path.splitext(out_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vw = cv2.VideoWriter(out_path_avi, fourcc, fps_out, (W, H))
            if not vw.isOpened():
                raise RuntimeError("VideoWriter cannot be opened (mp4/avi).")
            out_path = out_path_avi

        frame_rgba_local = frame_rgba.convert("RGBA")

        for _ in range(nframes):
            base = Image.new("RGBA", (W, H), (255, 255, 255, 255))

            for i, hinfo in enumerate(holes):
                if i >= len(caps):
                    break
                ok, fr = caps[i].read()
                if not ok or fr is None:
                    break

                fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                shot_img = Image.fromarray(fr_rgb, mode="RGB")

                x1, y1, x2, y2 = hinfo["bbox_px"]
                sw, sh = (x2 - x1), (y2 - y1)

                shot_img = trim_black_borders(shot_img)
                shot_fit = cover_resize_center_crop(shot_img, sw, sh).convert("RGBA")
                base.paste(shot_fit, (x1, y1), hinfo["mask"])

            comp = Image.alpha_composite(base, frame_rgba_local).convert("RGB")
            out_bgr = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR)
            vw.write(out_bgr)

        vw.release()
        return out_path

    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


# =========================
# Email sending
# =========================
def smtp_send(to_email: str, subject: str, text_body: str):
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP_USER/SMTP_PASS belum diset (env var).")

    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM} <{SMTP_USER}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


def _email_worker(to_email: str, participant_name: str | None, session_id: str):
    timeout_sec = 6 * 60
    start = time.time()

    while True:
        with SESSION_LOCK:
            job = SESSION_JOBS.get(session_id, {}).copy()

        if not job:
            return

        drive = job.get("drive") or {}
        folder_url = drive.get("folder_url")
        photo = drive.get("photo") or {}
        video = drive.get("video") or {}
        photo_url = photo.get("view_url")
        video_url = video.get("view_url")

        if folder_url and photo_url and video_url:
            break
        if time.time() - start > timeout_sec:
            break
        time.sleep(1.0)

    with SESSION_LOCK:
        job = SESSION_JOBS.get(session_id, {})
        drive = job.get("drive") or {}
        folder_url = drive.get("folder_url")
        photo = drive.get("photo") or {}
        video = drive.get("video") or {}
        photo_url = photo.get("view_url")
        video_url = video.get("view_url")

    pname = sanitize_participant_name(participant_name or "") or "Peserta"
    subject = f"{EVENT_TITLE} - {pname}"
    lines = [
        f"Halo {pname}!",
        "",
        "Ini link hasil photobooth kamu:",
        f"- Folder (Drive): {folder_url or '(belum tersedia)'}",
        f"- Foto (preview): {photo_url or '(belum tersedia)'}",
        f"- Video (preview): {video_url or '(belum tersedia)'}",
        "",
        "Kalau link belum tampil (internet venue lemot), buka Folder Drive lalu refresh beberapa saat.",
        "",
        "Terima kasih!",
    ]
    body = "\n".join(lines)

    try:
        smtp_send(to_email, subject, body)
        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["email"] = {"status": "sent", "to": to_email, "ts": time.time()}
    except Exception as e:
        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["email"] = {"status": "error", "to": to_email, "error": str(e), "ts": time.time()}


# =========================
# Drive session worker
# =========================
def _drive_session_worker(session_id: str, sess_name_fallback: str, photo_path: str, photo_name: str,
                         frame_rgba: Image.Image, holes, clip_paths: list[str]):
    try:
        svc = get_drive_service()

        with SESSION_LOCK:
            participant_name = (SESSION_JOBS.get(session_id, {}).get("participant") or {}).get("name")

        folder_display_name = build_folder_name(participant_name, fallback=sess_name_fallback, session_id=session_id)
        folder_id, folder_url = drive_create_session_folder(svc, folder_display_name)
        folder_qr = make_qr_data_url(folder_url)

        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["drive"]["folder_id"] = folder_id
                SESSION_JOBS[session_id]["drive"]["folder_url"] = folder_url
                SESSION_JOBS[session_id]["drive"]["folder_qr_data_url"] = folder_qr

        # upload photo
        photo_info = drive_upload_file(svc, photo_path, photo_name, folder_id, "image/jpeg")
        photo_qr = make_qr_data_url(photo_info["view_url"])

        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["drive"]["photo"] = {**photo_info, "qr_data_url": photo_qr}

        # compose + upload video (hanya kalau clip lengkap sesuai slot)
        need = len(holes)
        if len(clip_paths) == need:
            video_raw_name = f"live_{session_id}.mp4"
            video_raw_path = os.path.join(OUTPUT_DIR, video_raw_name)

            out_real = compose_strip_live_video(frame_rgba, clip_paths, holes, video_raw_path, fps_out=20)

            final_path = out_real
            try:
                h264_name = os.path.splitext(os.path.basename(out_real))[0] + "_h264.mp4"
                h264_path = os.path.join(OUTPUT_DIR, h264_name)
                h264_out = transcode_to_h264_mp4(out_real, h264_path, fps=20)
                if h264_out:
                    final_path = h264_out
            except Exception:
                pass

            ext = os.path.splitext(final_path)[1].lower()
            mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"

            video_name_drive = os.path.basename(final_path)
            video_info = drive_upload_file(svc, final_path, video_name_drive, folder_id, mime)
            video_qr = make_qr_data_url(video_info["view_url"])

            with SESSION_LOCK:
                if session_id in SESSION_JOBS:
                    SESSION_JOBS[session_id]["drive"]["video"] = {**video_info, "qr_data_url": video_qr}

        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["drive"]["status"] = "done"

        _prune_sessions()

    except Exception as e:
        with SESSION_LOCK:
            if session_id in SESSION_JOBS:
                SESSION_JOBS[session_id]["drive"]["status"] = "error"
                SESSION_JOBS[session_id]["drive"]["error"] = str(e)
        _prune_sessions()


# =========================
# Load frames
# =========================
FRAMES: dict[str, dict[str, Any]] = {}

def load_all_frames():
    global ACTIVE_FRAME
    FRAMES.clear()

    if os.path.isdir(FRAME_DIR):
        for fn in sorted(os.listdir(FRAME_DIR)):
            if not fn.lower().endswith(".png"):
                continue
            path = os.path.join(FRAME_DIR, fn)
            frame_id = os.path.splitext(fn)[0]
            img_raw = Image.open(path).convert("RGBA")

            holes, img = detect_slots_auto(img_raw)

            if len(holes) < 1:
                raise RuntimeError(
                    f"Frame '{fn}' gagal detect slot. Pastikan slot transparan atau pakai hijau key {GREEN_KEY_HEX}"
                )

            FRAMES[frame_id] = {"path": path, "img": img, "holes": holes}
            print(f"[OK] Frame '{frame_id}' slots={len(holes)} ({fn})")

    if not FRAMES and os.path.exists(FALLBACK_FRAME_PATH):
        img_raw = Image.open(FALLBACK_FRAME_PATH).convert("RGBA")
        holes, img = detect_slots_auto(img_raw)
        if len(holes) < 1:
            raise RuntimeError("Fallback frame.png gagal detect slot.")
        FRAMES["default"] = {"path": FALLBACK_FRAME_PATH, "img": img, "holes": holes}

    if not FRAMES:
        raise FileNotFoundError("No frames found. Put PNGs in frames/ or provide frame.png")

    ACTIVE_FRAME = list(FRAMES.keys())[0]
    print(f"[OK] Loaded frames: {list(FRAMES.keys())}. Active={ACTIVE_FRAME}")

load_all_frames()


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/file/<path:filename>")
def file_view(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, download_name=filename)


@app.post("/api/set_source")
def api_set_source():
    global ACTIVE_SOURCE
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or "").strip().lower()
    if source not in ("phone", "dslr"):
        return jsonify({"ok": False, "error": "source must be phone or dslr"}), 400
    ACTIVE_SOURCE = source
    return jsonify({"ok": True, "source": ACTIVE_SOURCE})


@app.get("/api/frames")
def api_frames():
    assert ACTIVE_FRAME is not None
    return jsonify({
        "ok": True,
        "active": ACTIVE_FRAME,
        "active_slots": len(FRAMES[ACTIVE_FRAME]["holes"]),
        "frames": [{"id": k, "label": k, "slots": len(v["holes"])} for k, v in FRAMES.items()]
    })


@app.post("/api/set_frame")
def api_set_frame():
    global ACTIVE_FRAME
    data = request.get_json(silent=True) or {}
    fid = (data.get("frame") or "").strip()
    if fid not in FRAMES:
        return jsonify({"ok": False, "error": "unknown frame", "frames": list(FRAMES.keys())}), 400
    ACTIVE_FRAME = fid
    return jsonify({"ok": True, "active": ACTIVE_FRAME, "slots": len(FRAMES[ACTIVE_FRAME]["holes"])})


@app.post("/api/connect_phone")
def api_connect_phone():
    open_mobile_devices_settings()
    return jsonify({"ok": True})


@app.post("/api/reconnect_camera")
def api_reconnect_camera():
    source = request.args.get("source", ACTIVE_SOURCE)
    get_cam(source).reconnect()
    return jsonify({"ok": True, "source": source})


@app.get("/api/camera_status")
def api_camera_status():
    source = request.args.get("source", ACTIVE_SOURCE)
    j = get_cam(source).camera_status()
    j["source"] = source
    return jsonify(j)


@app.get("/video_feed")
def video_feed():
    source = request.args.get("source", ACTIVE_SOURCE)
    cam = get_cam(source)

    def gen():
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY_STREAM])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-cache\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot")
def api_snapshot():
    source = request.args.get("source", ACTIVE_SOURCE)
    frame = get_cam(source).get_latest_frame()
    if frame is None:
        abort(503)
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        abort(500)
    return Response(jpg.tobytes(), mimetype="image/jpeg")


@app.post("/api/record_clip")
def api_record_clip():
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or ACTIVE_SOURCE).strip().lower()
    if source not in ("phone", "dslr"):
        return jsonify({"ok": False, "error": "source must be phone or dslr"}), 400

    seconds = float(data.get("seconds", 4.0))
    seconds = max(0.5, min(seconds, 12.0))

    fps = int(data.get("fps", 20))
    fps = max(10, min(fps, 30))

    cam = get_cam(source)
    first = cam.get_latest_frame()
    if first is None:
        abort(503)

    h, w = first.shape[:2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = f"rawclip_{source}_{ts}"

    out_path = os.path.join(OUTPUT_DIR, base + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not vw.isOpened():
        out_path = os.path.join(OUTPUT_DIR, base + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    start = time.time()
    next_t = start

    while True:
        now = time.time()
        if (now - start) >= seconds:
            break

        fr = cam.get_latest_frame()
        if fr is not None:
            if fr.shape[1] != w or fr.shape[0] != h:
                fr = cv2.resize(fr, (w, h))
            vw.write(fr)

        next_t += 1.0 / float(fps)
        sleep_s = next_t - time.time()
        if sleep_s > 0:
            time.sleep(min(sleep_s, 0.05))

    vw.release()
    filename = os.path.basename(out_path)
    return jsonify({"ok": True, "filename": filename, "seconds": seconds, "fps": fps})


@app.get("/api/session_status")
def api_session_status():
    sid = (request.args.get("id") or "").strip()
    if not sid:
        return jsonify({"ok": False, "error": "missing id"}), 400
    with SESSION_LOCK:
        job = SESSION_JOBS.get(sid)
    if not job:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "id": sid, **job})


@app.post("/api/set_participant")
def api_set_participant():
    data = request.get_json(silent=True) or {}
    sid = (data.get("id") or "").strip()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip() or None

    if not sid or len(name) < 2:
        return jsonify({"ok": False, "error": "invalid id/name"}), 400

    folder_id = None
    with SESSION_LOCK:
        job = SESSION_JOBS.get(sid)
        if not job:
            return jsonify({"ok": False, "error": "session not found"}), 404
        job["participant"]["name"] = name
        if email:
            job["participant"]["email"] = email
        folder_id = (job.get("drive") or {}).get("folder_id")

    # rename folder kalau folder sudah ada
    if folder_id:
        def _rename_bg():
            try:
                svc = get_drive_service()
                new_folder_name = build_folder_name(name, fallback=f"photobooth_{sid}", session_id=sid)
                drive_rename(svc, folder_id, new_folder_name)
            except Exception:
                pass

        threading.Thread(target=_rename_bg, daemon=True).start()

    return jsonify({"ok": True})


@app.post("/api/send_email")
def api_send_email():
    data = request.get_json(silent=True) or {}
    sid = (data.get("id") or "").strip()
    email = (data.get("email") or "").strip()
    name = (data.get("name") or "").strip()

    if not sid or "@" not in email or len(name) < 2:
        return jsonify({"ok": False, "error": "invalid id/name/email"}), 400

    folder_id = None
    with SESSION_LOCK:
        if sid not in SESSION_JOBS:
            return jsonify({"ok": False, "error": "session not found"}), 404
        SESSION_JOBS[sid]["participant"]["name"] = name
        SESSION_JOBS[sid]["participant"]["email"] = email
        SESSION_JOBS[sid]["email"] = {"status": "sending", "to": email, "ts": time.time()}
        folder_id = (SESSION_JOBS[sid].get("drive") or {}).get("folder_id")

    # rename segera kalau folder sudah ada
    if folder_id:
        try:
            svc = get_drive_service()
            new_folder_name = build_folder_name(name, fallback=f"photobooth_{sid}", session_id=sid)
            drive_rename(svc, folder_id, new_folder_name)
        except Exception:
            pass

    th = threading.Thread(target=_email_worker, args=(email, name, sid), daemon=True)
    th.start()
    return jsonify({"ok": True})


@app.post("/api/compose")
def api_compose():
    frame_id = (request.form.get("frame", "") or "").strip() or ACTIVE_FRAME
    files = request.files.getlist("shots")
    if not files:
        return jsonify({"ok": False, "error": "No shots uploaded"}), 400

    if frame_id not in FRAMES:
        frame_id = ACTIVE_FRAME
    assert frame_id is not None

    frame_rgba = FRAMES[frame_id]["img"]
    holes = FRAMES[frame_id]["holes"]
    need = len(holes)

    if len(files) < need:
        return jsonify({"ok": False, "error": f"Need {need} shots for this frame, got {len(files)}"}), 400

    clip_names = request.form.getlist("clips")[:need]
    clip_paths: list[str] = []
    for nm in clip_names:
        nm = os.path.basename(nm or "")
        p = os.path.join(OUTPUT_DIR, nm)
        if nm and os.path.exists(p):
            clip_paths.append(p)

    shots = [Image.open(f.stream).convert("RGB") for f in files[:need]]

    code = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_id = f"{frame_id}_{code}"
    sess_name_fallback = f"photobooth_{frame_id}_{code}"

    final_rgb = compose_strip_photo(frame_rgba, shots, holes)
    photo_name = f"photo_{session_id}.jpg"
    photo_path = os.path.join(OUTPUT_DIR, photo_name)
    final_rgb.save(photo_path, quality=JPEG_QUALITY_SAVE)

    ip = get_local_ip()
    local_photo_view = f"http://{ip}:{APP_PORT}/file/{photo_name}"

    with SESSION_LOCK:
        SESSION_JOBS[session_id] = {
            "created_at": time.time(),
            "frame": frame_id,
            "slots": need,
            "participant": {"name": None, "email": None},
            "local": {"photo_view_url": local_photo_view, "photo_filename": photo_name},
            "drive": {
                "status": "processing",
                "folder_id": None,
                "folder_url": None,
                "folder_qr_data_url": None,
                "photo": None,
                "video": None,
                "error": None,
            },
            "email": None
        }
    _prune_sessions()

    th = threading.Thread(
        target=_drive_session_worker,
        args=(session_id, sess_name_fallback, photo_path, photo_name, frame_rgba, holes, clip_paths),
        daemon=True
    )
    th.start()

    return jsonify({
        "ok": True,
        "id": session_id,
        "local": {"photo_view_url": local_photo_view, "photo_filename": photo_name},
        "frame": frame_id,
        "slots": need
    })


def warmup_drive_oauth():
    try:
        get_drive_service()
        print("[OK] Drive OAuth ready (token.json should exist).")
    except Exception as e:
        print("[WARN] Drive not ready yet:", e)


if __name__ == "__main__":
    warmup_drive_oauth()
    app.run(host="0.0.0.0", port=APP_PORT, debug=False, threaded=True)
