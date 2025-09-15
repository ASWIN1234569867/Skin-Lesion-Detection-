# app.py
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, redirect, url_for, flash, session, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------- CONFIG --------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "skin_mobilenetv2.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

# -------- DATABASE --------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        label TEXT,
        score REAL,
        uploaded_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# -------- MODEL --------
_model = None
def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = load_model(MODEL_PATH)
        else:
            # Build small MobileNetV2 for demo (untrained)
            base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.3)(x)
            out = Dense(1, activation="sigmoid")(x)
            _model = Model(inputs=base.input, outputs=out)
            _model.save(MODEL_PATH)
    return _model

def predict_image(img_path):
    model = get_model()
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    score = float(model.predict(arr, verbose=0)[0][0])
    label = "malignant" if score >= 0.5 else "benign"
    return {"label": label, "score": score}

# -------- HELPERS --------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_confidence_chart(upload_id, score):
    labels = ["benign", "malignant"]
    scores = [1 - score, score]
    path = os.path.join(VIS_DIR, f"conf_{upload_id}.png")
    plt.figure(figsize=(4,3))
    plt.bar(labels, scores)
    plt.ylim(0,1)
    plt.title("Confidence")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f"conf_{upload_id}.png"

def save_summary_chart(user_id, rows):
    benign = sum(1 for r in rows if r["label"]=="benign")
    malignant = sum(1 for r in rows if r["label"]=="malignant")
    avg_score = np.mean([r["score"] for r in rows]) if rows else 0
    path = os.path.join(VIS_DIR, f"summary_{user_id}.png")
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.pie([benign, malignant], labels=["benign","malignant"], autopct="%1.0f%%")
    plt.title("Label dist")
    plt.subplot(1,2,2)
    plt.bar(["avg malignant"], [avg_score])
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f"summary_{user_id}.png"

# -------- ROUTES --------
@app.context_processor
def inject_user():
    return dict(logged_in=("user_id" in session), username=session.get("username"))

@app.route("/")
def index():
    return render_template_string(TPL_INDEX)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        u, p = request.form["username"], request.form["password"]
        if not u or not p:
            flash("Fill all fields","danger"); return redirect(url_for("register"))
        pw = generate_password_hash(p)
        conn=get_db()
        try:
            conn.execute("INSERT INTO users (username,password_hash) VALUES (?,?)",(u,pw))
            conn.commit()
            flash("Registered!","success"); return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username taken","danger"); return redirect(url_for("register"))
    return render_template_string(TPL_REGISTER)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u, p = request.form["username"], request.form["password"]
        conn=get_db(); row=conn.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone(); conn.close()
        if row and check_password_hash(row["password_hash"], p):
            session["user_id"]=row["id"]; session["username"]=u
            flash("Logged in","success"); return redirect(url_for("index"))
        flash("Invalid","danger"); return redirect(url_for("login"))
    return render_template_string(TPL_LOGIN)

@app.route("/logout")
def logout():
    session.clear(); flash("Logged out","info"); return redirect(url_for("index"))

@app.route("/upload", methods=["GET","POST"])
def upload():
    if "user_id" not in session:
        flash("Login required","warning"); return redirect(url_for("login"))
    if request.method=="POST":
        f=request.files.get("image")
        if not f or f.filename=="":
            flash("No file","danger"); return redirect(request.url)
        if not allowed_file(f.filename):
            flash("Bad type","danger"); return redirect(request.url)
        fname=secure_filename(f"{session['user_id']}_{int(datetime.utcnow().timestamp())}_{f.filename}")
        path=os.path.join(UPLOAD_FOLDER,fname); f.save(path)
        res=predict_image(path)
        conn=get_db()
        cur=conn.execute("INSERT INTO uploads (user_id,filename,label,score,uploaded_at) VALUES (?,?,?,?,?)",
            (session["user_id"],fname,res["label"],res["score"],datetime.utcnow().isoformat()))
        upload_id=cur.lastrowid; conn.commit(); conn.close()
        flash(f"{res['label']} ({res['score']:.2f})","success")
        return redirect(url_for("prediction",upload_id=upload_id))
    return render_template_string(TPL_UPLOAD)

@app.route("/prediction/<int:upload_id>")
def prediction(upload_id):
    conn=get_db()
    row=conn.execute("SELECT * FROM uploads WHERE id=?",(upload_id,)).fetchone(); conn.close()
    if not row: flash("Not found","danger"); return redirect(url_for("index"))
    vis=save_confidence_chart(upload_id,row["score"])
    return render_template_string(TPL_PRED,record=row,vis_img=vis)

@app.route("/history")
def history():
    if "user_id" not in session: flash("Login required","warning"); return redirect(url_for("login"))
    conn=get_db()
    rows=conn.execute("SELECT * FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",(session["user_id"],)).fetchall()
    conn.close()
    vis=save_summary_chart(session["user_id"],rows)
    return render_template_string(TPL_HISTORY,rows=rows,summary_img=vis)

@app.route("/uploads/<fname>")
def serve_upload(fname): return send_from_directory(UPLOAD_FOLDER,fname)

@app.route("/visualizations/<fname>")
def serve_vis(fname): return send_from_directory(VIS_DIR,fname)

# --------- TEMPLATES ---------
TPL_INDEX = """
<!doctype html>
<html>
<head><title>Skin Lesion Detector</title></head>
<body>
  <h1>Skin Lesion Detector</h1>
  {% with msgs = get_flashed_messages(with_categories=true) %}
    {% for cat,msg in msgs %}
      <div style="padding:5px;margin:5px;background:#eee;">{{cat}}: {{msg}}</div>
    {% endfor %}
  {% endwith %}

  {% if not logged_in %}
    <p><a href="{{ url_for('login') }}">Login</a> or 
       <a href="{{ url_for('register') }}">Register</a></p>
  {% else %}
    <p>Hello {{username}}!</p>
    <p><a href="{{ url_for('upload') }}">Upload</a> |
       <a href="{{ url_for('history') }}">History</a> |
       <a href="{{ url_for('logout') }}">Logout</a></p>
  {% endif %}
</body>
</html>
"""

TPL_REGISTER = """
<!doctype html>
<html>
<head><title>Register</title></head>
<body>
  <h2>Register</h2>
  <form method="post">
    <input name="username" placeholder="Username"><br>
    <input type="password" name="password" placeholder="Password"><br>
    <button type="submit">Register</button>
  </form>
  <p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
</body>
</html>
"""

TPL_LOGIN = """
<!doctype html>
<html>
<head><title>Login</title></head>
<body>
  <h2>Login</h2>
  <form method="post">
    <input name="username" placeholder="Username"><br>
    <input type="password" name="password" placeholder="Password"><br>
    <button type="submit">Login</button>
  </form>
  <p>No account? <a href="{{ url_for('register') }}">Register</a></p>
</body>
</html>
"""

TPL_UPLOAD = """
<!doctype html>
<html>
<head><title>Upload</title></head>
<body>
  <h2>Upload an image</h2>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image"><br>
    <button type="submit">Upload</button>
  </form>
  <p><a href="{{ url_for('index') }}">Home</a></p>
</body>
</html>
"""

TPL_PRED = """
<!doctype html>
<html>
<head><title>Prediction</title></head>
<body>
  <h2>Prediction Result</h2>
  <p><b>Label:</b> {{record.label}} | <b>Score:</b> {{'%.2f'|format(record.score)}}</p>
  <img src="{{ url_for('serve_upload',fname=record.filename) }}" width="200"><br>
  <img src="{{ url_for('serve_vis',fname=vis_img) }}" width="300"><br>
  <p><a href="{{ url_for('history') }}">Back to history</a></p>
</body>
</html>
"""

TPL_HISTORY = """
<!doctype html>
<html>
<head><title>History</title></head>
<body>
  <h2>Your Upload History</h2>
  <ul>
  {% for r in rows %}
    <li>
      {{r.uploaded_at}} - {{r.label}} ({{'%.2f'|format(r.score)}})
      <br>
      <img src="{{ url_for('serve_upload',fname=r.filename) }}" width="100">
    </li>
  {% endfor %}
  </ul>
  <h3>Summary</h3>
  <img src="{{ url_for('serve_vis',fname=summary_img) }}" width="400">
  <p><a href="{{ url_for('index') }}">Home</a></p>
</body>
</html>
"""

# --------- RUN ---------
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
