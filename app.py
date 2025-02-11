from flask import Flask, request, jsonify, render_template,session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import pickle
import spacy
import numpy as np
import json
import re


app = Flask(_name_, template_folder="templates", static_folder="static")



app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)
CORS(app, supports_credentials=True)

nlp = spacy.load("en_core_web_sm")
model = pickle.load(open("random_forest_model_download.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder_download.pkl", "rb"))


all_symptoms = [ "anxiety and nervousness","depression","shortness of breath","depressive or psychotic symptoms","sharp chest pain","dizziness","insomnia","abnormal involuntary movements","chest tightness","palpitations","irregular heartbeat","breathing fast","hoarse voice","sore throat","difficulty speaking","cough","nasal congestion","throat swelling","diminished hearing","lump in throat","throat feels tight","difficulty in swallowing","skin swelling","retention of urine","groin mass","leg pain","hip pain","suprapubic pain","blood in stool","lack of growth","emotional symptoms","elbow weakness","back weakness","pus in sputum","symptoms of the scrotum and testes","swelling of scrotum","pain in testicles","flatulence","pus draining from ear","jaundice","mass in scrotum","white discharge from eye","irritable infant","abusing alcohol","fainting","hostile behavior","drug abuse","sharp abdominal pain","feeling ill","vomiting","headache","nausea","diarrhea","vaginal itching","vaginal dryness","painful urination","involuntary urination","pain during intercourse","frequent urination","lower abdominal pain","vaginal discharge","blood in urine","hot flashes","intermenstrual bleeding","hand or finger pain","wrist pain","hand or finger swelling","arm pain","wrist swelling","arm stiffness or tightness","arm swelling","hand or finger stiffness or tightness","wrist stiffness or tightness","lip swelling","toothache","abnormal appearing skin","skin lesion","acne or pimples","dry lips","facial pain","mouth ulcer","skin growth","eye deviation","diminished vision","double vision","cross-eyed","symptoms of eye","pain in eye","eye moves abnormally","abnormal movement of eyelid","foreign body sensation in eye","irregular appearing scalp","swollen lymph nodes","back pain","neck pain","low back pain","pain of the anus","pain during pregnancy","pelvic pain","impotence","infant spitting up","vomiting blood","regurgitation","burning abdominal pain","restlessness","symptoms of infants","wheezing","peripheral edema","neck mass","ear pain","jaw swelling","mouth dryness","neck swelling","knee pain","foot or toe pain","bowlegged or knock-kneed","ankle pain","bones are painful","knee weakness","elbow pain","knee swelling","skin moles","knee lump or mass","weight gain","problems with movement","knee stiffness or tightness","leg swelling","foot or toe swelling","heartburn","smoking problems","muscle pain","infant feeding problem","recent weight loss","problems with shape or size of breast","underweight","difficulty eating","scanty menstrual flow","vaginal pain","vaginal redness","vulvar irritation","weakness","decreased heart rate","increased heart rate","bleeding or discharge from nipple","ringing in ear","plugged feeling in ear","itchy ear(s)","frontal headache","fluid in ear","neck stiffness or tightness","spots or clouds in vision","eye redness","lacrimation","itchiness of eye","blindness","eye burns or stings","itchy eyelid","feeling cold","decreased appetite","excessive appetite","excessive anger","loss of sensation","focal weakness","slurring words","symptoms of the face","disturbance of memory","paresthesia","side pain","fever","shoulder pain","shoulder stiffness or tightness","shoulder weakness","arm cramps or spasms","shoulder swelling","tongue lesions","leg cramps or spasms","abnormal appearing tongue","ache all over","lower body pain","problems during pregnancy","spotting or bleeding during pregnancy","cramps and spasms","upper abdominal pain","stomach bloating","changes in stool appearance","unusual color or odor to urine","kidney mass","swollen abdomen","symptoms of prostate","leg stiffness or tightness","difficulty breathing","rib pain","joint pain","muscle stiffness or tightness","pallor","hand or finger lump or mass","chills","groin pain","fatigue","abdominal distention","regurgitation.1","symptoms of the kidneys","melena","flushing","coughing up sputum","seizures","delusions or hallucinations","shoulder cramps or spasms","joint stiffness or tightness","pain or soreness of breast","excessive urination at night","bleeding from eye","rectal bleeding","constipation","temper problems","coryza","wrist weakness","eye strain","hemoptysis","lymphedema","skin on leg or foot looks infected","allergic reaction","congestion in chest","muscle swelling","pus in urine","abnormal size or shape of ear","low back weakness","sleepiness","apnea","abnormal breathing sounds","excessive growth","elbow cramps or spasms","feeling hot and cold","blood clots during menstrual periods","absence of menstruation","pulling at ears","gum pain","redness in ear","fluid retention","flu-like syndrome","sinus congestion","painful sinuses","fears and phobias","recent pregnancy","uterine contractions","burning chest pain","back cramps or spasms","stiffness all over","muscle cramps, contractures, or spasms","low back cramps or spasms","back mass or lump","nosebleed","long menstrual periods","heavy menstrual flow","unpredictable menstruation","painful menstruation","infertility","frequent menstruation","sweating","mass on eyelid","swollen eye","eyelid swelling","eyelid lesion or rash","unwanted hair","symptoms of bladder","irregular appearing nails","itching of skin","hurts to breath","nailbiting","skin dryness, peeling, scaliness, or roughness","skin on arm or hand looks infected","skin irritation","itchy scalp","hip swelling","incontinence of stool","foot or toe cramps or spasms","warts","bumps on penis","too little hair","foot or toe lump or mass","skin rash","mass or swelling around the anus","low back swelling","ankle swelling","hip lump or mass","drainage in throat","dry or flaky scalp","premenstrual tension or irritability","feeling hot","feet turned in","foot or toe stiffness or tightness","pelvic pressure","elbow swelling","elbow stiffness or tightness","early or late onset of menopause","mass on ear","bleeding from ear","hand or finger weakness","low self-esteem","throat irritation","itching of the anus","swollen or red tonsils","irregular belly button","swollen tongue","lip sore","vulvar sore","hip stiffness or tightness","mouth pain","arm weakness","leg lump or mass","disturbance of smell or taste","discharge in stools","penis pain","loss of sex drive","obsessions and compulsions","antisocial behavior","neck cramps or spasms","pupils unequal","poor circulation","thirst","sleepwalking","skin oiliness","sneezing","bladder mass","knee cramps or spasms","premature ejaculation","leg weakness","posture problems","bleeding in mouth","tongue bleeding","change in skin mole size or color","penis redness","penile discharge","shoulder lump or mass","polyuria","cloudy eye","hysterical behavior","arm lump or mass","nightmares","bleeding gums","pain in gums","bedwetting","diaper rash","lump or mass of breast","vaginal bleeding after menopause","infrequent menstruation","mass on vulva","jaw pain","itching of scrotum","postpartum problems of the breast","eyelid retracted","hesitancy","elbow lump or mass","muscle weakness","throat redness","joint swelling","tongue pain","redness in or around nose","wrinkles on skin","foot or toe weakness","hand or finger cramps or spasms","back stiffness or tightness","wrist lump or mass","skin pain","low back stiffness or tightness","low urine output","skin on head or neck looks infected","stuttering or stammering","problems with orgasm","nose deformity","lump over jaw","sore in nose","hip weakness","back swelling","ankle stiffness or tightness","ankle weakness","neck weakness"]

@app.route("/")
def home():
    return render_template("finalwebpage.html")

def extract_symptoms(user_text):
    doc = nlp(user_text.lower())  

    detected_symptoms = []
    user_symptom_string = " ".join([token.text for token in doc]) 

    for symptom in all_symptoms:
        if symptom in user_symptom_string:  
            detected_symptoms.append(symptom)

    return list(set(detected_symptoms))

def load_risk_levels():
    with open("risk_levels_full.json", "r") as file:
        return json.load(file)

def assess_urgency(disease, severity, duration):
    risk_levels = load_risk_levels()
    
    if disease not in risk_levels:
        return "Unknown Disease"
    
    base_urgency = risk_levels[disease][severity]
    
    if base_urgency == "Emergency":
        return "Emergency"
    
    if duration > 14:
        if base_urgency == "Moderate":
            return "Emergency"
        elif base_urgency == "Mild":
            return "Severe"
    elif duration > 7:
        if base_urgency == "Severe":
            return "Emergency"
        elif base_urgency == "Moderate":
            return "Severe"
        elif base_urgency == "Mild":
            return "Moderate"
    
    return base_urgency

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Email Validation Regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Signup Route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name, email, password = data.get("name"), data.get("email"), data.get("password")

    # Validate email format
    if not EMAIL_REGEX.match(email):
        return jsonify({"message": "Invalid email format!"}), 400

    # Check if email is already registered
    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email already registered!"}), 400

    # Hash password before storing
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Store user in database
    new_user = User(name=name, email=email, username=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully!"}), 201

# Login Route
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email, password = data.get("username"), data.get("password")

    user = User.query.filter_by(email=email).first()

    # Check user exists and verify password
    if user and bcrypt.check_password_hash(user.password, password):
        session['user'] = user.username  # Store in session
        return jsonify({"message": "Login successful!", "username": user.username, "email": user.email}), 200
    return jsonify({"message": "Invalid credentials!"}), 401

# Get Profile Data Route
@app.route('/profile', methods=['GET'])
def profile():
    if 'user' in session:
        user = User.query.filter_by(username=session['user']).first()
        return jsonify({"username": user.username, "email": user.email})
    return jsonify({"message": "Unauthorized"}), 401

# Logout Route
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)  # Remove user from session
    return jsonify({"message": "Logged out successfully!"})

@app.route("/book-appointment.html")
def home1():
    return render_template("book-appointment.html")

@app.route("/book-lab-test.html")
def home2():
    return render_template("book-lab-test.html")

@app.route("/organdonor.html")
def home3():
    return render_template("organdonor.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("message", "")
    severity = data.get("severity", "mild").lower()
    duration = int(data.get("duration", 1))
    print(duration)
    print(severity)
    extracted_symptoms = extract_symptoms(user_text)
    input_vector = [1 if symptom in extracted_symptoms else 0 for symptom in all_symptoms]

    predicted_label = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    urgency=assess_urgency(predicted_disease,severity,duration)
    return jsonify({
        "user_symptoms": extracted_symptoms,
        "prediction": predicted_disease,
        "urgency": urgency
    })

if _name_ == "_main_":
    app.run(debug=True)