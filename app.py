import os
import secrets
from functools import wraps
from datetime import timedelta
from flask import Flask, flash, redirect, url_for, session, request, render_template, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Regexp
from dotenv import load_dotenv
import requests
import json
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import WebApplicationClient
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re
import csv
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///sit.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True

db = SQLAlchemy(app)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

client = WebApplicationClient(GOOGLE_CLIENT_ID)

# User model for Flask-Login
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=True)
    name = db.Column(db.String(100))
    security_question = db.Column(db.String(100))
    security_answer = db.Column(db.String(255))
    google_id = db.Column(db.String(255), unique=True, nullable=True)

    def get_id(self):
        return str(self.id)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in to access this page.", 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Forms
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    new_password = PasswordField('Password', validators=[
        DataRequired(), 
        Length(min=8), 
        Regexp(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('new_password')])
    security_question = SelectField('Security Question', 
                                    choices=[('pet', 'What was the name of your first pet?'), 
                                             ('mother_maiden', 'What is your mother\'s maiden name?'), 
                                             ('birth_city', 'In which city were you born?')],
                                    validators=[DataRequired()])
    security_answer = StringField('Security Answer', validators=[DataRequired()])
    submit = SubmitField('Register')

class ResetPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    security_question = SelectField('Security Question', 
                                    choices=[('pet', 'What was the name of your first pet?'), 
                                             ('school', 'What was the name of your elementary school?'), 
                                             ('city', 'In which city were you born?'), 
                                             ('friend', 'What is your best friend\'s first name?'), 
                                             ('mother', 'What is your mother\'s maiden name?')],
                                    validators=[DataRequired()])
    security_answer = StringField('Security Answer', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[
        DataRequired(), 
        Length(min=8), 
        Regexp(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$')
    ])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Reset Password')

# Google OAuth Helper
def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

# Google Login Routes
@app.route("/google_login")
def google_login():
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    state = secrets.token_urlsafe(16)
    session["state"] = state
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.url_root + url_for("google_callback"),
        scope=["openid", "email", "profile"],
        state=state
    )
    return redirect(request_uri)

@app.route("/google_login/callback")
def google_callback():
    state = session["state"]
    if request.args.get("state") != state:
        flash("Invalid state parameter.", 'error')
        return redirect(url_for("login"))

    token_endpoint = get_google_provider_cfg()["token_endpoint"]
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.url,
        state=state
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )
    client.parse_request_body_response(json.dumps(token_response.json()))
    userinfo_endpoint = get_google_provider_cfg()["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        users_name = userinfo_response.json()["given_name"]
    else:
        flash("User email not available or not verified by Google.", 'error')
        return redirect(url_for("login"))

    user = User.query.filter_by(google_id=unique_id).first()
    if not user:
        user = User(email=users_email, name=users_name, google_id=unique_id)
        try:
            db.session.add(user)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error adding new user: {e}")

    login_user(user, remember=True)
    flash('Logged in successfully with Google.', 'success')
    return redirect(url_for('index'))

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.password and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        elif user and user.google_id:
            flash('This account was registered with Google. Please sign in with Google.', 'error')
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('login'))
        new_user = User(
            email=form.email.data,
            password=generate_password_hash(form.new_password.data),
            security_question=form.security_question.data,
            security_answer=generate_password_hash(form.security_answer.data),
            name=""
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"Error during registration: {e}")
            flash('An error occurred while registering. Please try again.', 'error')
    return render_template('register.html', form=form)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.security_question == form.security_question.data:
            if check_password_hash(user.security_answer, form.security_answer.data):
                try:
                    user.password = generate_password_hash(form.new_password.data)
                    db.session.commit()
                    flash('Password reset successful! Please login.', 'success')
                    return redirect(url_for('login'))
                except Exception as e:
                    db.session.rollback()
                    print(f"Error updating password: {e}")
                    flash('An error occurred while resetting the password.', 'error')
            else:
                flash('Incorrect security answer.', 'error')
        else:
            flash('Email or security question not found.', 'error')
    return render_template('reset_password.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Load Decision Tree Model and Related Objects
with open('decision_tree_model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    cols = pickle.load(f)

# Load CSV Files into Dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: index for index, symptom in enumerate(cols)}

def getSeverityDict():
    with open('Symptom_severity.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:
                severityDictionary[row[0]] = int(row[1])

def getDescription():
    with open('symptom_Description.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:
                description_list[row[0]] = row[1]

def getPrecautionDict():
    with open('symptom_precaution.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 4:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getSeverityDict()
getDescription()
getPrecautionDict()

# Decision Tree Helper Functions
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if len(pred_list) > 0 else (0, [])

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])[0]

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary[item] for item in exp if item in severityDictionary)
    return "You should take the consultation from a doctor." if ((sum_severity * days) / (len(exp) + 1)) > 13 else "It might not be that bad but you should take precautions."

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return [d.strip() for d in disease]

# Chatbot State Management
def get_user_chat_state():
    if 'chat_state' not in session:
        session['chat_state'] = {
            'mode': 'general',
            'step': 0,
            'name': '',
            'disease_input': '',
            'num_days': 0,
            'symptoms_present': [],
            'symptoms_exp': [],
            'current_symptom': 0,
            'disease_predicted': False,
            'present_disease': '',
            'symptoms_given': [],
            'cnf_dis': []
        }
    print(f"Current chat_state: {session['chat_state']}")  # Debugging
    return session['chat_state']

try:
    model_path = r'C:\frontend\quickcare_final\quickcare - Copy\my_t5_model'  # Use absolute path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 100
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load T5 model: {e}")
    model = None
    tokenizer = None

def remove_question_prefix(response, question):
    question_no_punct = re.sub(r'[^\w\s]', '', question).strip()
    response_no_punct = re.sub(r'[^\w\s]', '', response).strip()
    if response_no_punct.lower().startswith(question_no_punct.lower()):
        prefix_length = len(question)
        separators = [' ', ' - ', ': ']
        for sep in separators:
            if response.startswith(question + sep):
                return response[len(question) + len(sep):].strip()
        return response[len(question):].strip()
    return response

def chatbot_response(question):
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please check the server configuration."
    
    model.eval()
    source = f'answer: {question}'
    encoding = tokenizer.encode_plus(
        source,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    raw_response = preds[0]

    cleaned_response = remove_question_prefix(raw_response, question)
    if len(cleaned_response.split()) < 10:
        return "I’m not sure about that. Please consult a healthcare professional for accurate advice."
    return cleaned_response


# Routes
@app.route('/')
@login_required
def index():
    chat_state = get_user_chat_state()
    chat_state['mode'] = 'general'
    chat_state['step'] = 0
    session.modified = True
    return render_template('index.html', user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user_input = request.form.get('question', '').strip()
    chat_state = get_user_chat_state()
    print(f"User input: '{user_input}', Mode: {chat_state['mode']}, Step: {chat_state['step']}")  # Debugging

    if user_input.lower() in ['exit', 'quit']:
        chat_state['mode'] = 'general'
        chat_state['step'] = 0
        session.modified = True
        return jsonify({'answer': 'Goodbye!'})

    
    if chat_state['mode'] == 'general':
        if user_input.lower() == 'switch to symptom mode':
            chat_state['mode'] = 'symptom'
            chat_state['step'] = 0
            chat_state['name'] = ''
            chat_state['disease_input'] = ''
            chat_state['num_days'] = 0
            chat_state['symptoms_present'] = []
            chat_state['symptoms_exp'] = []
            chat_state['current_symptom'] = 0
            chat_state['disease_predicted'] = False
            chat_state['present_disease'] = ''
            chat_state['symptoms_given'] = []
            chat_state['cnf_dis'] = []
            session.modified = True
            print(f"Switched to symptom mode: {chat_state}")  # Debugging
            return jsonify({'answer': f"Hello {current_user.name or current_user.email.split('@')[0]}\n\nEnter the symptom you are experiencing:"})
        response = chatbot_response(user_input)
        return jsonify({'answer': response})

    elif chat_state['mode'] == 'symptom':
        if chat_state['step'] == 0:  # Get name
            chat_state['name'] = user_input or current_user.name or current_user.email.split('@')[0]
            chat_state['step'] = 1
            session.modified = True
            print(f"Step 0 completed: {chat_state}")  # Debugging
            return jsonify({'answer': f"Hello {chat_state['name']}\n\nEnter the symptom you are experiencing:"})

        elif chat_state['step'] == 1:  # Get initial symptom
            conf, cnf_dis = check_pattern(cols, user_input)
            if conf == 1:
                if len(cnf_dis) > 1:
                    options = "\n".join([f"{i} ) {sym}" for i, sym in enumerate(cnf_dis)])
                    chat_state['step'] = 2
                    chat_state['cnf_dis'] = cnf_dis
                    session.modified = True
                    print(f"Step 1 (multiple symptoms): {chat_state}")  # Debugging
                    return jsonify({'answer': f"searches related to input:\n{options}\nSelect the one you meant (0 - {len(cnf_dis)-1}):"})
                else:
                    chat_state['disease_input'] = cnf_dis[0]
                    chat_state['step'] = 3
                    session.modified = True
                    print(f"Step 1 (single symptom): {chat_state}")  # Debugging
                    return jsonify({'answer': "Okay. From how many days?"})
            return jsonify({'answer': "Enter a valid symptom."})

        elif chat_state['step'] == 2:  # Select symptom from options
            try:
                conf_inp = int(user_input)
                if 0 <= conf_inp < len(chat_state['cnf_dis']):
                    chat_state['disease_input'] = chat_state['cnf_dis'][conf_inp]
                    chat_state['step'] = 3
                    session.modified = True
                    print(f"Step 2 completed: {chat_state}")  # Debugging
                    return jsonify({'answer': "Okay. From how many days?"})
                return jsonify({'answer': "Invalid selection. Try again."})
            except ValueError:
                return jsonify({'answer': "Enter a valid number."})

        elif chat_state['step'] == 3:  # Get number of days
            try:
                chat_state['num_days'] = int(user_input)
                chat_state['step'] = 4
                tree_ = clf.tree_
                node = 0
                while tree_.feature[node] != -2:
                    name = cols[tree_.feature[node]]
                    threshold = tree_.threshold[node]
                    val = 1 if name == chat_state['disease_input'] else 0
                    if val <= threshold:
                        node = tree_.children_left[node]
                    else:
                        chat_state['symptoms_present'].append(name)
                        node = tree_.children_right[node]
                present_disease = print_disease(tree_.value[node])
                chat_state['present_disease'] = present_disease[0]
                df = pd.read_csv('Training.csv')
                reduced_data = df.groupby(df['prognosis']).max()
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                chat_state['symptoms_given'] = list(symptoms_given)
                chat_state['current_symptom'] = 0
                session.modified = True
                print(f"Step 3 completed: {chat_state}")  # Debugging
                return jsonify({'answer': f"Are you experiencing any\n{chat_state['symptoms_given'][0]}?"})
            except ValueError:
                return jsonify({'answer': "Enter a valid number of days."})

        elif chat_state['step'] == 4:  # Collect symptom responses
            if user_input.lower() in ['yes', 'no']:
                if user_input.lower() == 'yes':
                    chat_state['symptoms_exp'].append(chat_state['symptoms_given'][chat_state['current_symptom']])
                chat_state['current_symptom'] += 1
                if chat_state['current_symptom'] < len(chat_state['symptoms_given']):
                    session.modified = True
                    print(f"Step 4 (next symptom): {chat_state}")  # Debugging
                    return jsonify({'answer': f"Are you experiencing any\n{chat_state['symptoms_given'][chat_state['current_symptom']]}?"})
                else:
                    second_prediction = sec_predict(chat_state['symptoms_exp'])
                    severity_msg = calc_condition(chat_state['symptoms_exp'], chat_state['num_days'])
                    output = f"{severity_msg}\nYou may have  {chat_state['present_disease']} or  {second_prediction}\n{description_list.get(chat_state['present_disease'], 'No description available.')}\n{description_list.get(second_prediction, 'No description available.')}"
                    precautions = precautionDictionary.get(chat_state['present_disease'], ['No precautions available.'])
                    output += "\nTake following measures:\n" + "\n".join([f"{i+1} ) {p}" for i, p in enumerate(precautions)])
                    chat_state['mode'] = 'general'
                    chat_state['step'] = 0
                    chat_state['name'] = ''
                    chat_state['disease_input'] = ''
                    chat_state['num_days'] = 0
                    chat_state['symptoms_present'] = []
                    chat_state['symptoms_exp'] = []
                    chat_state['current_symptom'] = 0
                    chat_state['disease_predicted'] = False
                    chat_state['present_disease'] = ''
                    chat_state['symptoms_given'] = []
                    chat_state['cnf_dis'] = []
                    session.modified = True
                    print(f"Step 4 completed, back to general: {chat_state}")  # Debugging
                    return jsonify({'answer': output})
            return jsonify({'answer': "Please answer with 'yes' or 'no'."})

@app.before_request
def before_request():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully.")
        except Exception as e:
            print(f"Failed to create database tables: {e}")
    app.run(debug=True)