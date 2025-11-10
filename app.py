import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

N_IMGS = 10
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'static/face_recognition_model.pkl'
ATTENDANCE_DIR = 'Attendance'
FACES_DIR = 'static/faces'

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
ATTENDANCE_FILE = f'Attendance/Attendance-{datetoday}.csv'


# Initializing VideoCapture object to access WebCam
try:
    face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
except Exception as e:
    print(f"Error loading face detector: {e}")
    face_detector = None


# If these directories don't exist, create them
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w') as f:
        f.write('Name,Roll,Time')


# Helper Functions
# -----------------------------------------------------------

def totalreg():
    """Get a number of total registered users."""
    # Counts the number of non-hidden user folders in FACES_DIR
    return len([d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d)) and not d.startswith('.')])


def extract_faces(img):
    """Extract face bounding boxes from an image."""
    if face_detector is None:
        return []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except cv2.error:
        return []
    except Exception:
        return []


def identify_face(facearray):
    """Identify face using ML model."""
    if not os.path.exists(MODEL_PATH):
        return "Unknown" # Or handle error appropriately
    try:
        model = joblib.load(MODEL_PATH)
        return model.predict(facearray)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown"


def train_model():
    """Trains the model on all the faces available in faces folder."""
    faces = []
    labels = []
    userlist = os.listdir(FACES_DIR)
    for user in userlist:
        user_path = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_path): continue
        for imgname in os.listdir(user_path):
            img_path = os.path.join(user_path, imgname)
            img = cv2.imread(img_path)
            if img is None: continue

            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
            
    if not faces:
        print("No faces to train on.")
        return

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, MODEL_PATH)
    print("Model trained successfully.")


def extract_attendance():
    """Extract info from today's attendance file."""
    if not os.path.exists(ATTENDANCE_FILE):
        return [], [], [], 0

    df = pd.read_csv(ATTENDANCE_FILE)
    # Ensure columns exist before accessing
    if 'Name' in df.columns and 'Roll' in df.columns and 'Time' in df.columns:
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        return names, rolls, times, len(df)
    return [], [], [], 0


def add_attendance(name):
    """Add Attendance of a specific user."""
    try:
        username, userid = name.split('_')[0], name.split('_')[1]
    except IndexError:
        print(f"Invalid user format: {name}")
        return

    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)
    # Check if user has already marked attendance today
    if int(userid) not in df['Roll'].values:
        with open(ATTENDANCE_FILE, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    """Get names and roll numbers of all registered users."""
    userlist = os.listdir(FACES_DIR)
    names = []
    rolls = []
    valid_userlist = []
    
    for i in userlist:
        if '_' in i: # Ensure it follows the expected format
            try:
                name, roll = i.split('_')
                names.append(name)
                rolls.append(roll)
                valid_userlist.append(i)
            except ValueError:
                continue # Skip incorrectly formatted folder

    return valid_userlist, names, rolls, len(valid_userlist)


def deletefolder(duser):
    """Delete a user folder and its contents."""
    user_path = os.path.join(FACES_DIR, duser)
    if os.path.isdir(user_path):
        for i in os.listdir(user_path):
            os.remove(os.path.join(user_path, i))
        os.rmdir(user_path)


# Routing Functions
# -----------------------------------------------------------

@app.route('/')
def home():
    """Our main page."""
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/listusers')
def listusers():
    """List users page."""
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    """Delete functionality."""
    duser = request.args.get('user')
    deletefolder(duser)
    
    if not os.listdir(FACES_DIR) and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    
    try:
        if os.listdir(FACES_DIR): # Only train if there are still faces
            train_model()
    except Exception as e:
        print(f"Error during re-training after deletion: {e}")

    return redirect(url_for('listusers'))


@app.route('/start', methods=['GET'])
def start():
    """Main Face Recognition functionality - Take Attendance."""
    if not os.path.exists(MODEL_PATH):
        names, rolls, times, l = extract_attendance()
        # Redirect back to home with an error message
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Could not open webcam.')

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            # Draw a modern, clean box
            rect_color = (0, 169, 255) # Light blue
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
            
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            add_attendance(identified_person)
            
            # Display name
            text_color = (255, 255, 255)
            text_bg_color = rect_color
            cv2.rectangle(frame, (x, y - 30), (x + w, y), text_bg_color, -1)
            cv2.putText(frame, identified_person.split('_')[0], (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
        cv2.imshow('Attendance - Press ESC to Exit', frame)
        if cv2.waitKey(1) == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Reload data and redirect to home
    return redirect(url_for('home'))


@app.route('/add', methods=['POST'])
def add():
    """Function to add a new user."""
    # Using POST method ensures data security and proper form submission handling
    newusername = request.form.get('newusername')
    newuserid = request.form.get('newuserid')
    
    if not newusername or not newuserid:
        # Handle case where fields are empty (though HTML handles required)
        return redirect(url_for('home'))
        
    userimagefolder = os.path.join(FACES_DIR, f'{newusername}_{newuserid}')
    os.makedirs(userimagefolder, exist_ok=True)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        # Handle camera error elegantly
        return redirect(url_for('home'))

    while i < N_IMGS:
        _, frame = cap.read()
        if frame is None:
            break

        faces = extract_faces(frame)
        
        # Display feedback on the screen
        feedback_color = (0, 255, 0) if i > 0 else (0, 0, 255) # Green when capturing, Red before
        cv2.putText(frame, f'Capturing: {i}/{N_IMGS} images. Press ESC to stop.', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if j % 5 == 0 and i < N_IMGS: # Capture every 5th frame
                # Save the face region
                face_img = frame[y:y+h, x:x+w]
                name = f'{newusername}_{i+1}.jpg'
                cv2.imwrite(os.path.join(userimagefolder, name), face_img)
                i += 1
            j += 1
            
        cv2.imshow('Adding New User - Press ESC to Exit', frame)
        if cv2.waitKey(1) == 27: # ESC key
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Only train if at least one image was captured
    if i > 0:
        print('Training Model...')
        train_model()
    else:
        # Clean up the folder if no images were captured
        if not os.listdir(userimagefolder):
            os.rmdir(userimagefolder)
            
    return redirect(url_for('home'))


# Our main function which runs the Flask App
if __name__ == '__main__':
    # Consider running in a non-debug mode for production stability
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)