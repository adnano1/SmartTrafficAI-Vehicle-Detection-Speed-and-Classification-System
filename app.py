import cv2
from scripts.vehicle_detection import detect_vehicles
from scripts.speed_calculation import calculate_speed
from scripts.plate_recognition import recognize_plate
from config.config import VIDEO_SOURCE

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = detect_vehicles(frame)

        current_positions = {}
        for idx, (label, box) in enumerate(vehicles):
            x, y, w, h = box
            current_positions[idx] = (x + w // 2, y + h // 2)

            # Speed calculation
            speeds = calculate_speed(current_positions)

            # Display vehicle label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}, Speed: {int(speeds.get(idx, 0))} units/sec", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Number Plate recognition
            roi = frame[y:y+h, x:x+w]
            plate = recognize_plate(roi)
            cv2.putText(frame, f"Plate: {plate}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

