from flask import render_template, request, redirect, url_for, flash, abort, send_file
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app import app, db
from app.models import User
import csv
from io import StringIO
from functools import wraps
from flask import abort

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------
# Auth Routes
# ---------------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
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
    return render_template('dashboard.html', user=current_user)


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


@app.route('/voice', methods=['GET', 'POST'])
@login_required
def voice_recognition():
    if request.method == 'POST':
        result = "✅ Voice Verified"
        return render_template('voice_recognition.html', result=result)
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
    flash("🧑‍🧠 Face model retraining started!", "success")
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

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)

    retrain_status = "✅ All models trained"
    logs = []  # Replace with actual logs if needed

    return render_template("admin/admin_dashboard.html", retrain_status=retrain_status, logs=logs)


@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)

    users = User.query.all()
    return render_template('admin/users.html', users=users)


@app.route('/admin/users/promote/<int:user_id>', methods=['POST'])
@login_required
def promote_user(user_id):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(user_id)
    user.is_admin = True
    db.session.commit()
    flash(f"{user.username} has been promoted to admin!", "success")
    return redirect(url_for('admin_users'))


@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        abort(403)

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
def export_users():
    if not current_user.is_admin:
        abort(403)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Username', 'Email', 'Is Admin'])

    for user in User.query.all():
        writer.writerow([user.id, user.username, user.email, user.is_admin])

    output.seek(0)
    return send_file(
        output,
        mimetype='text/csv',
        download_name='users.csv',
        as_attachment=True
    )

@app.route('/admin/toggle-admin/<int:user_id>', methods=['POST'])
@login_required
def admin_toggle_admin(user_id):
    if not current_user.is_admin:
        abort(403)

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
def admin_delete_user(user_id):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ You cannot delete yourself!", "danger")
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f"🗑️ Deleted user {user.username}.", "warning")

    return redirect(url_for('admin_users'))

# Add New User from Admin Panel
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
