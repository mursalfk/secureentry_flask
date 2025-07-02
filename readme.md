## 🚪 SecureEntry – Smart Dual-Authentication System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/)
[![React Native](https://img.shields.io/badge/React%20Native-Mobile%20App-blue.svg)](https://reactnative.dev/)
[![Chaquopy](https://img.shields.io/badge/Chaquopy-Embedded%20Python-orange.svg)](https://chaquo.com/chaquopy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<details>
<summary><strong>📦 Project Overview</strong></summary>

SecureEntry is a real-time, AI-powered smart authentication system that uses both face and voice recognition to control access to secured environments. The system includes:

* 🌐 Flask-based backend
* 📱 React Native mobile frontend
* 🧠 Embedded AI (FaceNet + Keras Voice model)
* 🔐 Admin + User dashboards
* 📸 Face + 🎙️ Voice dual-auth for secure entry

</details>

---

<details>
<summary><strong>🛠️ Backend Features (Flask)</strong></summary>

* ✅ **User Authentication System**

  * Flask-Login integration
  * Sign Up / Login / Logout flows
  * Admin vs User role segregation
  * Profile management (face/voice samples)

* 👑 **Admin Role**

  * `@admin_required` route protection
  * Admin-only UI and actions

* 🧠 **AI Integration**

  * FaceNet & Voice Recognition using Keras
  * Voice model retraining via Chaquopy (in Android)
  * Face model update & retraining support

</details>

---

<details>
<summary><strong>🎨 Admin Panel UI</strong></summary>

* 📂 Structured under `templates/admin/` + `static/scss/_admin.scss`

* 🧭 Dashboard Sections:

  * 👥 Manage Users (with Promote, Delete, Export, Search, Pagination)
  * 📜 User Logs (Coming Soon)
  * 🧠 Model Status (Retrain Face/Voice)

* 🔁 Back to user dashboard button

* 🧩 Styled with modular SCSS (`@use`)

* ⚡ Fully mobile-responsive layout

</details>

---

<details>
<summary><strong>📲 User Dashboard (React Native & Flask)</strong></summary>

* 🧑 Face & 🎙 Voice authentication pages
* 👤 Profile with:

  * Add/Update face samples
  * Add/Update voice samples
  * Automatic model retraining after updates
* ⚙ Settings, ℹ About Us pages
* 👑 *Admin Dashboard* link (visible only if user is admin)

</details>

---

<details>
<summary><strong>📁 SCSS Styling System</strong></summary>

* ✅ Modular SCSS setup:

  * `_admin.scss`, `_index.scss`, `main.scss`
* 🎨 Design includes:

  * Neumorphic buttons
  * Cards with shadows and hover effects
  * Clean table UI
  * Responsive dashboard grid
* 🔁 Live-compiled via `sass --watch`

</details>

---

<details>
<summary><strong>🧠 AI & Model Training Integration</strong></summary>

* 🎙️ Voice Recognition:

  * Keras-based model (TFLite)
  * Retraining triggered from app
  * Model stored locally and updated via Chaquopy

* 👨‍🦰 Face Recognition:

  * Based on FaceNet embeddings (TFLite)
  * Face data added via profile page
  * Verification logic built in Kotlin (`FaceRecognitionHelper.kt`)

</details>

---

<details>
<summary><strong>📦 Project Folder Structure</strong></summary>

```
SecureEntry/
│
├── app/
│   ├── templates/
│   │   ├── admin/
│   │   └── base.html, login.html, etc.
│   ├── static/
│   │   ├── scss/
│   │   └── css/
│   ├── routes.py
│   ├── models.py
│   ├── forms.py
│   └── ...
│
├── SecureEntry20/          # React Native mobile app
│
├── voice_model/            # Voice samples & model
├── face_model/             # FaceNet embeddings & helper
└── run.py
```

</details>

---

## 🚀 Next Steps

* [ ] Finish **Add New User** form for admins
* [ ] Implement **User Logs** screen (logging login/logout actions)
* [ ] Improve **404 / 500 error pages**
* [ ] Add **unit tests** and backend test coverage
* [ ] Deploy backend to AWS/GCP/Render
* [ ] Convert all model paths to use S3 + AWS Lambda
* [ ] Add **notification support** (email alerts on intrusion)