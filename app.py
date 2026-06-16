from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import base64
import numpy as np
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('best.pt')

counted_ids = set()
stats = {"helmet": 0, "no_helmet": 0, "total": 0}
violation_logs = []
last_frame = None
is_paused = False  

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

def save_automatic_snapshot(frame, label):
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f"snapshots/auto_{label}_{time_str}.jpg"
    cv2.imwrite(filename, frame)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_frame, stats, violation_logs, is_paused
    
    if is_paused:
        if last_frame is not None:
            _, buffer = cv2.imencode('.jpg', last_frame)
            return jsonify({"image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')})
        return jsonify({"status": "paused"})

    # Menerima data gambar Base64 dari frontend
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data"}), 400

    try:
        # Decode base64 menjadi array numpy, lalu menjadi format gambar OpenCV
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Proses tracking menggunakan YOLO
        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            
            for track_id, class_id in zip(track_ids, class_ids):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    stats["total"] += 1
                    
                    class_name = model.names[class_id].lower()
                    
                    if "no" in class_name or "tanpa" in class_name or "without" in class_name:
                        stats["no_helmet"] += 1
                        time_now = datetime.now().strftime("%H:%M:%S")
                        violation_logs.insert(0, {"time": time_now, "status": "Tanpa Helm"})
                        
                        if len(violation_logs) > 15:
                            violation_logs.pop()
                        
                        annotated_snapshot = results[0].plot()
                        save_automatic_snapshot(annotated_snapshot, "no_helmet")
                        
                    elif "helmet" in class_name or "helm" in class_name:
                        stats["helmet"] += 1

        annotated_frame = results[0].plot()
        last_frame = annotated_frame 
        
        # Encode kembali ke base64 untuk dikirim ke frontend
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "image": "data:image/jpeg;base64," + result_b64
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": "Failed to process frame"}), 500

@app.route('/get_stats')
def get_stats():
    return jsonify({"stats": stats, "logs": violation_logs, "is_paused": is_paused})

@app.route('/toggle_camera/<action>')
def toggle_camera(action):
    global is_paused
    if action == "pause":
        is_paused = True
        return jsonify({"status": "success"})
    elif action == "resume":
        is_paused = False
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route('/snapshot')
def snapshot():
    global last_frame
    if last_frame is not None:
        filename = f"snapshots/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, last_frame)
        return jsonify({"status": "success", "message": f"Gambar disimpan di folder: {filename}"})
    return jsonify({"status": "error", "message": "Gagal menangkap layar."})

if __name__ == "__main__":
    # Gunakan host '0.0.0.0' agar web bisa diakses dari perangkat lain dalam 1 jaringan Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)