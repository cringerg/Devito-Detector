from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    "cascades/haarcascade_frontalface_default.xml")
devito_clf = cv2.face.LBPHFaceRecognizer_create()
devito_clf.read("devito_clf.yml")

camera = cv2.VideoCapture(0)


def draw_boundary(img, classifier, scale_factor, min_neighbours, colour, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(
        gray_img, scale_factor, min_neighbours)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)
        id_, prob = clf.predict(gray_img[y:y+h, x:x+w])
        if prob > 80 and prob < 125:
            prob = str(int(prob))
            cv2.putText(img, "Devito ", (x, y-7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour, 2, cv2.LINE_AA)
        else:
            prob = str(int(prob))
            cv2.putText(img, "Not Devito", (x, y-7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour, 2, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords, img


def recognise(img, clf, cascade):
    colour = {"blue": (255, 0, 0), "red": (0, 0, 255),
              "green": (0, 255, 0), "white": (255, 255, 255)}

    coords = draw_boundary(img, cascade, 1.1, 10,
                           colour["red"], "Face", clf)

    return img


def gen_frames():
    while True:
        success, frame = camera.read()
        frame = recognise(frame, devito_clf, face_cascade)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
