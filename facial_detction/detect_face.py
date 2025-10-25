import os
import time
import cv2
import base64
import openai
import mediapipe as mp
import re
# from google import genai

# --- API setup ---
client = openai.Client(
    api_key=os.getenv("BOSON_API_KEY"),
    base_url="https://hackathon.boson.ai/v1"
)

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=GEMINI_API_KEY)
# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

OUTPUT_INTERVAL = 3.0
last_output = 0.0

# --- MediaPipe face detector ---
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

print("Press 'q' to quit.")

label_to_display = None  # store last API emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)

    if results.detections:
        det = results.detections[0]  # first/closest face
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * iw)
        y1 = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        # enlarge box slightly
        pad = int(0.2 * max(w, h))
        x1m = max(0, x1 - pad)
        y1m = max(0, y1 - pad)
        x2m = min(iw, x1 + w + pad)
        y2m = min(ih, y1 + h + pad)

        # crop face
        face_crop = frame[y1m:y2m, x1m:x2m]

        # --- only call API every OUTPUT_INTERVAL seconds ---
        if time.time() - last_output >= OUTPUT_INTERVAL:
            last_output = time.time()
            _, buffer = cv2.imencode('.jpg', face_crop)
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

            try:
                # response = client.models.generate_content(
                #     model="gemini-2.0-flash", 
                #     contents=[img_base64, "Detect the dominant emotion in the image and respond with one to three words only."]
                # )
                # label_to_display = response.text.strip()
                resp = client.chat.completions.create(
                    model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that detects facial emotions."},
                        {"role": "user", "content": [
                            {"type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                            }},
                            {"type": "text", "text": "Output the dominant emotion with one to three words only. Do not overthink. Just provide the label."}
                        ]}
                    ],
                    max_tokens=256,
                    temperature=0
                )
                response = resp.choices[0].message.content
                label_to_display = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                print(label_to_display)
                print(f"[{time.strftime('%H:%M:%S')}] Detected emotion: {label_to_display}")
            except Exception as e:
                print("API error:", e)

        # draw bounding box and label
        cv2.rectangle(frame, (x1m, y1m), (x2m, y2m), (0, 255, 0), 2)
        if label_to_display:
            cv2.putText(frame, label_to_display, (x1m, y1m - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Face + Emotion Detection (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
