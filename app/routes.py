from flask import render_template, request, redirect, url_for, flash, abort, send_file, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app import app, db
from app.models import User
from functools import wraps
from werkzeug.utils import secure_filename
from io import StringIO
import base64
import os
from app.voice_recognition_helper import predict_user

# ---------------------------
# Admin Check Decorator
# ---------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# ---------------------------
# Homepage
# ---------------------------
@app.route('/')
def home():
    return render_template('home.html')

# ---------------------------
# Auth Routes
# ---------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------------------------
# User Dashboard Routes
# ---------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/open-door')
@login_required
def open_door():
    return render_template('open_door.html')

@app.route('/face', methods=['GET', 'POST'])
@login_required
def face_recognition():
    if request.method == 'POST':
        result = "✅ Face Verified"
        return render_template('face_recognition.html', result=result)
    return render_template('face_recognition.html')

import wave
import base64
import numpy as np
from flask import request, jsonify, render_template
from flask_login import login_required, current_user
from app.voice_recognition_helper import predict_user

@app.route('/voice', methods=['GET', 'POST'])
@login_required
def voice_recognition():
    if request.method == 'POST':
        data = request.get_json()
        audio_base64 = data.get('audio')

        if not audio_base64:
            return jsonify({'message': '❌ No audio received'}), 400

        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        temp_path = f'temp_audio_{current_user.id}.wav'

        try:
            # Convert raw PCM bytes into a proper WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)         # Mono
                wf.setsampwidth(2)         # 16-bit
                wf.setframerate(16000)     # 16 kHz
                wf.writeframes(audio_bytes)

            # Run prediction
            predicted_user = predict_user(temp_path)

        except Exception as e:
            print(f"🚨 Error during voice prediction: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'message': '❌ Voice recognition failed'}), 500

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if predicted_user == current_user.username:
            return jsonify({'message': f'✅ Welcome Home, {current_user.username}'})
        else:
            return jsonify({'message': '❌ Voice not recognized'}), 200

    return render_template('voice_recognition.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.username = request.form['username']
        current_user.email = request.form['email']
        db.session.commit()
        flash("✅ Profile updated!", "success")
    return render_template('profile.html', user=current_user)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/retrain-face', methods=['POST'])
@login_required
def retrain_face():
    flash("🧑\u200d🧠 Face model retraining started!", "success")
    return redirect(url_for('settings'))

@app.route('/retrain-voice', methods=['POST'])
@login_required
def retrain_voice():
    flash("🎤 Voice model retraining started!", "success")
    return redirect(url_for('settings'))

@app.route('/about')
def about():
    return render_template('about.html')

# ---------------------------
# Admin Portal Routes
# ---------------------------
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    retrain_status = "✅ All models trained"
    return render_template("admin/admin_dashboard.html", retrain_status=retrain_status)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/promote/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def promote_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_admin = True
    db.session.commit()
    flash(f"{user.username} has been promoted to admin!", "success")
    return redirect(url_for('admin_users'))

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ You cannot delete yourself!", "danger")
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f"Deleted user {user.username}.", "warning")
    return redirect(url_for('admin_users'))

@app.route('/admin/users/export')
@login_required
@admin_required
def export_users():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Username', 'Email', 'Is Admin'])
    for user in User.query.all():
        writer.writerow([user.id, user.username, user.email, user.is_admin])
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name='users.csv', as_attachment=True)

@app.route('/admin/toggle-admin/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ You cannot change your own admin status!", "danger")
    else:
        user.is_admin = not user.is_admin
        db.session.commit()
        status = "admin" if user.is_admin else "regular user"
        flash(f"✅ {user.username} is now a {status}.", "success")
    return redirect(url_for('admin_users'))

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ You cannot delete yourself!", "danger")
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f"🗑️ Deleted user {user.username}.", "warning")
    return redirect(url_for('admin_users'))

@app.route('/admin/add_user', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_add_user():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        is_admin = 'is_admin' in request.form

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
        else:
            new_user = User(username=username, email=email, is_admin=is_admin)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('New user created successfully!', 'success')
            return redirect(url_for('admin_users'))

    return render_template('admin/add_user.html')

@app.route('/retrain-voice-model', methods=['POST'])
@login_required
def retrain_voice_model():
    import base64, os, wave
    import datetime
    from app.voice_retrain_helper import retrain_model
    from app.voice_chunker import chunk_audio_file  # you'll create this

    try:
        # Step 1: Decode base64 audio
        data = request.get_json()
        audio_base64 = data.get('audio')
        if not audio_base64:
            return jsonify({"message": "❌ No audio data received."}), 400

        user_folder = os.path.join("app/dataset/voice_data", current_user.username)
        chunk_folder = os.path.join(user_folder, "chunks")
        os.makedirs(chunk_folder, exist_ok=True)

        # Step 2: Save audio sample
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(user_folder, f"recording_{timestamp}.wav")

        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            audio_bytes = base64.b64decode(audio_base64)
            wf.writeframes(audio_bytes)

        # Step 3: Chunk the sample into 1-second .wav files
        chunk_audio_file(audio_path, chunk_folder)

        # Optional: remove original raw file
        os.remove(audio_path)

        # Step 4: Retrain the model
        message = retrain_model()

        return jsonify({"message": message or "✅ Retraining completed!"})

    except Exception as e:
        print(f"🚨 Error during retraining: {e}")
        return jsonify({"message": "❌ Voice retraining failed."}), 500
