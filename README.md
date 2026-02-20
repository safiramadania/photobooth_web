# 📸 Photobooth Web System

Event-ready web-based photobooth system built with **Flask, OpenCV, and Google Drive API**.

Supports:

* DSLR / Phone camera
* Multi-slot frame auto detection (transparent or green key)
* Live-photo style video strip
* Auto upload to Google Drive
* QR code result sharing
* Automatic email delivery
* Auto folder rename by participant name

Designed for offline event deployment (local network).

---

# 🏗 Architecture Overview

Frontend:

* Vanilla HTML/CSS/JS
* MJPEG live stream
* Countdown & multi-shot capture
* Async Drive polling
* QR generation display

Backend:

* Flask API
* Threaded camera manager
* Frame slot auto-detection
* Photo & video strip composer
* Google Drive OAuth upload worker
* SMTP email worker

---

# 📂 Project Structure

```
photobooth_web/
│
├── app.py
├── make_frame.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── frames/
│   ├── frame1.png
│   ├── frame2.png
│
├── output/        (auto-generated, ignored by git)
│
├── credentials.json   (NOT committed)
└── token.json         (NOT committed)
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/USERNAME/photobooth_web.git
cd photobooth_web
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If missing, ensure these are included:

```
flask
pillow
qrcode[pil]
numpy
opencv-python
google-api-python-client
google-auth
google-auth-oauthlib
```

---

# 🔐 Google Drive Setup (OAuth Desktop)

## 1️⃣ Create OAuth Credentials

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create new project
3. Enable **Google Drive API**
4. Create OAuth Client ID

   * Application Type: **Desktop App**
5. Download `credentials.json`

Place `credentials.json` next to `app.py`.

---

## 2️⃣ First Run (Token Generation)

Run:

```bash
python app.py
```

Browser will open → Login → Allow access.

`token.json` will be generated automatically.

---

# 📁 Configure Drive Parent Folder

Inside `app.py`:

```python
DEFAULT_PARENT_ID = "YOUR_DRIVE_FOLDER_ID"
```

Or use environment variable:

```bash
set DRIVE_PARENT_ID=YOUR_FOLDER_ID
```

---

# 📧 SMTP Email Setup (Gmail Example)

Set environment variables:

```bash
set SMTP_USER=your_email@gmail.com
set SMTP_PASS=your_app_password
set SMTP_FROM=Photobooth KSE 2026
set EVENT_TITLE=Hasil Photobooth KSE 2026
```

For Gmail:

* Enable 2FA
* Generate App Password
* Use that as SMTP_PASS

---

# 🎞 FFmpeg (Optional but Recommended)

For better video compatibility (H264):

Install FFmpeg and ensure:

```bash
ffmpeg -version
```

works in terminal.

Or put `ffmpeg.exe` inside:

```
_ffmpeg/
```

---

# 🚀 How To Run

```bash
python app.py
```

Server runs at:

```
http://localhost:8000
```

Or accessible via local IP for event network:

```
http://192.168.x.x:8000
```

---

# 📷 How To Use (Event Flow)

## 1️⃣ Landing Screen

* Choose:

  * START USING PHONE
  * START USING CAMERA (DSLR)

---

## 2️⃣ Booth Screen

* Select frame
* Click START
* Countdown begins
* Auto record clip + capture photo

---

## 3️⃣ Result Screen

* Preview composed photo
* QR code appears (Drive folder)
* Can:

  * Open folder
  * Preview photo
  * Preview video

---

## 4️⃣ Participant Input

Enter:

* Name (required)
* Email (optional)

System will:

* Rename Drive folder
* Send email automatically once upload complete

---

# 🖼 Frame System

Place PNG frames inside:

```
frames/
```

Supported slot types:

### 1️⃣ Transparent Slot

Make hole area transparent in PNG.

### 2️⃣ Green Key Slot

Use exact color:

```
#01e678
```

System auto-detects slot positions.

---

# 🛠 Create Frame From Design

Use:

```bash
python make_frame.py --in newspaper.png --out frame_transparent.png
```

Advanced options available:

* Feather
* Despill
* Margin
* Green tolerance

---

# 🧠 Session Handling

* In-memory session storage
* Max 300 sessions
* Async Drive + Email worker
* Auto folder rename

---

# 🖥 Deployment Notes

Recommended for:

* Local event machine
* Windows laptop with DSLR capture
* Same WiFi network for QR access

Not optimized for:

* Public cloud deployment
* High concurrency SaaS

---

# 📦 Future Improvements

* SQLite session storage
* Docker container
* Upload progress bar
* Retry system for Drive
* Admin dashboard
* Multi-event configuration
