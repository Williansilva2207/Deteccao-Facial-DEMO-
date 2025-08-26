import os
import sys
import argparse
import cv2
from mtcnn import MTCNN
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def detect_and_draw(frame, detector, save_crops=False, crop_dir="crops"):
    faces = detector.detect_faces(frame)
    h, w, _ = frame.shape
    for i, f in enumerate(faces):
        x, y, width, height = f["box"]
        
        x, y = max(0, x), max(0, y)
        x2, y2 = x + width, y + height
        
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        
        keypoints = f.get("keypoints", {})
        for k, p in keypoints.items():
            cv2.circle(frame, p, 2, (0, 0, 255), -1)
        
        if save_crops:
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            crop = frame[y:y2, x:x2]
            
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            filename = os.path.join(crop_dir, f"face_{i}.jpg")
            cv2.imwrite(filename, crop_bgr)
    return len(faces), frame

def run_on_image(path, detector, save_crops=False):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Erro: não foi possível abrir imagem: {path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    n, out = detect_and_draw(rgb, detector, save_crops=save_crops)
    
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    print(f"Faces detectadas: {n}")
    cv2.imshow("Detecao", out_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_webcam(detector, camera_id=0, save_crops=False):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_bgr = cv2.resize(frame_bgr, (0,0), fx=0.75, fy=0.75)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            n, out = detect_and_draw(rgb, detector, save_crops=save_crops, crop_dir=f"crops/frame_{frame_idx}")
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.putText(out_bgr, f"Faces: {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Webcam - Detecção (pressione q para sair)", out_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Detecção facial com TensorFlow (MTCNN) + OpenCV")
    parser.add_argument("--source", type=str, default="webcam",
                        help="webcam (default) ou caminho para imagem. Ex: --source webcam OR --source foto.jpg")
    parser.add_argument("--save-crops", action="store_true", help="salvar recortes das faces detectadas")
    parser.add_argument("--camera-id", type=int, default=0, help="id da câmera (se usar webcam)")
    args = parser.parse_args()

    print("Carregando detector (pode demorar alguns segundos)...")
    detector = MTCNN()  # usa TensorFlow internamente
    print("Detector carregado.")

    if args.source.lower() == "webcam":
        run_on_webcam(detector, camera_id=args.camera_id, save_crops=args.save_crops)
    else:
        run_on_image(args.source, detector, save_crops=args.save_crops)

if __name__ == "__main__":
    main()
