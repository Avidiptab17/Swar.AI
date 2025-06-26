from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, decode_token
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime, timedelta
import uuid
import threading
import time
from typing import Dict, List, Optional
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ZU0WR4R7KhXhDeLYulHxICG4EEpf7hQFF9LNNyVKqfo'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///swar_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = '0vO4VL5e0-dIvECHvK7j8N4ODJ8Np1Uua5SrPsMBg2XOZQtAc0ESqDxwD4qza8CGZ9EWgME9EilA2ipcAZdjXg'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for audio files
os.makedirs('generated_audio', exist_ok=True)
os.makedirs('user_uploads', exist_ok=True)

# Database initialization flag
_database_initialized = False

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    level = db.Column(db.Integer, default=1)
    experience = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Composition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    raag = db.Column(db.String(100), nullable=False)
    tempo = db.Column(db.Integer, nullable=False)
    intensity = db.Column(db.Integer, nullable=False)
    mood = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    lyrics_theme = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class JamSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.String(100), unique=True, nullable=False)
    host_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    participants = db.Column(db.Text, nullable=True)  # JSON string
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Indian Classical Music Theory
RAAGS = {
    'yaman': {
        'notes': ['Sa', 'Re', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni'],
        'mood': 'peaceful',
        'time': 'evening',
        'color': '#ffd700'
    },
    'bhairav': {
        'notes': ['Sa', 'Reb', 'Ga', 'Ma', 'Pa', 'Dhb', 'Ni'],
        'mood': 'devotional',
        'time': 'morning',
        'color': '#ff8c00'
    },
    'kafi': {
        'notes': ['Sa', 'Re', 'Gab', 'Ma', 'Pa', 'Dha', 'Nib'],
        'mood': 'romantic',
        'time': 'night',
        'color': '#dc143c'
    },
    'darbari': {
        'notes': ['Sa', 'Re', 'Gab', 'Ma', 'Pa', 'Dhb', 'Nib'],
        'mood': 'serious',
        'time': 'night',
        'color': '#8b4513'
    },
    'malkauns': {
        'notes': ['Sa', 'Gab', 'Ma', 'Dhb', 'Nib'],
        'mood': 'mystical',
        'time': 'night',
        'color': '#4b0082'
    }
}

NOTE_FREQUENCIES = {
    'Sa': 261.63,   # C4
    'Reb': 277.18,  # C#4/Db4
    'Re': 293.66,   # D4
    'Gab': 311.13,  # D#4/Eb4
    'Ga': 329.63,   # E4
    'Ma': 349.23,   # F4
    'Ma#': 369.99,  # F#4/Gb4
    'Pa': 392.00,   # G4
    'Dhb': 415.30,  # G#4/Ab4
    'Dha': 440.00,  # A4
    'Nib': 466.16,  # A#4/Bb4
    'Ni': 493.88    # B4
}

class MusicGenerator:
    def __init__(self):
        self.sample_rate = 44100

    def generate_tone(self, frequency: float, duration: float, intensity: float = 0.5) -> np.ndarray:
        """Generate a pure tone with given frequency and duration"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Add harmonics for more realistic sound
        wave = np.sin(2 * np.pi * frequency * t) * intensity
        wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t) * intensity
        wave += 0.1 * np.sin(2 * np.pi * frequency * 3 * t) * intensity

        # Apply envelope for natural sound
        envelope = np.exp(-t * 2)  # Exponential decay
        wave *= envelope

        return wave

    def generate_composition(self, raag: str, tempo: int, intensity: int,
                             mood: str, notes: List[str] = None,
                             lyrics_theme: str = None) -> str:
        """Generate a complete musical composition"""
        try:
            if raag not in RAAGS:
                raag = 'yaman'  # Default raag

            raag_info = RAAGS[raag]
            composition_notes = notes if notes else raag_info['notes']

            # Calculate note duration based on tempo
            beat_duration = 60.0 / tempo  # Duration of one beat in seconds
            note_duration = beat_duration / 2  # Two notes per beat

            # Adjust intensity (1-10 scale to 0.1-1.0)
            audio_intensity = intensity / 10.0

            # Generate composition
            audio_segments = []
            composition_length = 32  # Number of notes in composition

            for i in range(composition_length):
                # Select note based on raag and musical progression
                note_index = i % len(composition_notes)
                note = composition_notes[note_index]

                # Add some musical variation
                if i % 4 == 0:  # Emphasize every 4th note
                    current_intensity = audio_intensity * 1.2
                    current_duration = note_duration * 1.5
                else:
                    current_intensity = audio_intensity
                    current_duration = note_duration

                # Generate note
                frequency = NOTE_FREQUENCIES.get(note, 261.63)
                tone = self.generate_tone(frequency, current_duration, current_intensity)
                audio_segments.append(tone)

                # Add small pause between notes
                pause = np.zeros(int(self.sample_rate * 0.05))
                audio_segments.append(pause)

            # Combine all segments
            full_audio = np.concatenate(audio_segments)

            # Normalize audio
            full_audio = full_audio / np.max(np.abs(full_audio))

            # Save to file
            filename = f"composition_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join('generated_audio', filename)
            sf.write(filepath, full_audio, self.sample_rate)

            logger.info(f"Generated composition: {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error generating composition: {str(e)}")
            return None

# Initialize music generator
music_generator = MusicGenerator()

# Database initialization function
def init_database():
    global _database_initialized
    if not _database_initialized:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created")
            _database_initialized = True

# Initialize database before first request
@app.before_request
def before_request():
    init_database()

# Helper function to validate JWT token for Socket.IO
def validate_socket_token(token):
    try:
        if not token:
            return None
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        # Decode token manually
        decoded_token = decode_token(token)
        return decoded_token['sub']
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return None

# Test route for debugging
@app.route('/')
def index():
    return jsonify({
        'message': 'Swar.AI Backend is running!',
        'status': 'success',
        'endpoints': [
            '/api/register (POST)',
            '/api/login (POST)',
            '/api/generate-music (POST)',
            '/api/compositions (GET)',
            '/api/raags (GET)',
            '/api/user/profile (GET)',
            '/api/chat (POST)'
        ]
    })

@app.route('/api/test')
def test_endpoint():
    return jsonify({'message': 'Test endpoint working!', 'status': 'success'})

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        # Validate input data
        if not data or not all(k in data for k in ('username', 'email', 'password')):
            return jsonify({'error': 'Missing required fields'}), 400

        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400

        user = User(
            username=data['username'],
            email=data['email']
        )
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        access_token = create_access_token(identity=str(user.id))

        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'level': user.level,
                'experience': user.experience
            }
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        # Validate input data
        if not data or not all(k in data for k in ('username', 'password')):
            return jsonify({'error': 'Missing username or password'}), 400

        user = User.query.filter_by(username=data['username']).first()

        if user and user.check_password(data['password']):
            access_token = create_access_token(identity=str(user.id))
            return jsonify({
                'access_token': access_token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'level': user.level,
                    'experience': user.experience
                }
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/generate-music', methods=['POST'])
@jwt_required()
def generate_music():
    try:
        data = request.get_json()
        user_id = int(get_jwt_identity())

        # Extract parameters with defaults
        raag = data.get('raag', 'yaman')
        tempo = int(data.get('tempo', 120))
        intensity = int(data.get('intensity', 5))
        mood = data.get('mood', 'calm')
        notes = data.get('notes', [])
        lyrics_theme = data.get('lyrics_theme', '')
        title = data.get('title', f'Composition {datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # Generate music
        filename = music_generator.generate_composition(
            raag=raag,
            tempo=tempo,
            intensity=intensity,
            mood=mood,
            notes=notes,
            lyrics_theme=lyrics_theme
        )

        if not filename:
            return jsonify({'error': 'Failed to generate music'}), 500

        # Save composition to database
        composition = Composition(
            user_id=user_id,
            title=title,
            raag=raag,
            tempo=tempo,
            intensity=intensity,
            mood=mood,
            notes=json.dumps(notes),
            lyrics_theme=lyrics_theme,
            file_path=filename
        )

        db.session.add(composition)

        # Award experience points
        user = User.query.get(user_id)
        user.experience += 50

        # Level up check
        if user.experience >= user.level * 100:
            user.level += 1
            user.experience = 0

        db.session.commit()

        return jsonify({
            'message': 'Music generated successfully',
            'composition': {
                'id': composition.id,
                'title': composition.title,
                'filename': filename,
                'raag': raag,
                'tempo': tempo,
                'intensity': intensity,
                'mood': mood
            },
            'user': {
                'level': user.level,
                'experience': user.experience
            }
        })

    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate music'}), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    try:
        filepath = os.path.join('generated_audio', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Audio serving error: {str(e)}")
        return jsonify({'error': 'Failed to serve audio'}), 500

@app.route('/api/compositions')
@jwt_required()
def get_compositions():
    try:
        user_id = int(get_jwt_identity())
        compositions = Composition.query.filter_by(user_id=user_id).order_by(Composition.created_at.desc()).all()

        return jsonify({
            'compositions': [{
                'id': comp.id,
                'title': comp.title,
                'raag': comp.raag,
                'tempo': comp.tempo,
                'intensity': comp.intensity,
                'mood': comp.mood,
                'filename': comp.file_path,
                'created_at': comp.created_at.isoformat()
            } for comp in compositions]
        })

    except Exception as e:
        logger.error(f"Get compositions error: {str(e)}")
        return jsonify({'error': 'Failed to fetch compositions'}), 500

@app.route('/api/raags')
def get_raags():
    return jsonify({'raags': RAAGS})

@app.route('/api/user/profile')
@jwt_required()
def get_user_profile():
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)

        if not user:
            return jsonify({'error': 'User not found'}), 404

        composition_count = Composition.query.filter_by(user_id=user_id).count()

        return jsonify({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'level': user.level,
                'experience': user.experience,
                'composition_count': composition_count,
                'joined': user.created_at.isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return jsonify({'error': 'Failed to fetch profile'}), 500

# Socket.IO Events for Real-time Features (Fixed)
@socketio.on('connect')
def handle_connect(auth):
    try:
        # Get token from auth data
        token = auth.get('token') if auth else None
        user_id = validate_socket_token(token)

        if not user_id:
            logger.warning('Unauthorized socket connection attempt')
            return False  # Reject connection

        logger.info(f'User {user_id} connected via socket')
        emit('connected', {'status': 'Connected to Swar.AI'})
        return True
    except Exception as e:
        logger.error(f"Socket connection error: {str(e)}")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('User disconnected from socket')

@socketio.on('join_jam_session')
def handle_join_jam_session(data):
    try:
        # Get token from data
        token = data.get('token')
        user_id = validate_socket_token(token)

        if not user_id:
            emit('error', {'message': 'Authentication required'})
            return

        room_id = data.get('room_id')
        if not room_id:
            emit('error', {'message': 'Room ID required'})
            return

        join_room(room_id)

        # Update or create jam session
        session = JamSession.query.filter_by(room_id=room_id).first()
        if not session:
            session = JamSession(room_id=room_id, host_id=int(user_id))
            db.session.add(session)

        # Add user to participants
        participants = json.loads(session.participants) if session.participants else []
        if int(user_id) not in participants:
            participants.append(int(user_id))
            session.participants = json.dumps(participants)

        db.session.commit()

        emit('joined_jam_session', {'room_id': room_id, 'participants': participants}, room=room_id)
        logger.info(f'User {user_id} joined jam session {room_id}')

    except Exception as e:
        logger.error(f"Join jam session error: {str(e)}")
        emit('error', {'message': 'Failed to join jam session'})

@socketio.on('leave_jam_session')
def handle_leave_jam_session(data):
    try:
        # Get token from data
        token = data.get('token')
        user_id = validate_socket_token(token)

        if not user_id:
            emit('error', {'message': 'Authentication required'})
            return

        room_id = data.get('room_id')
        leave_room(room_id)

        # Remove user from participants
        session = JamSession.query.filter_by(room_id=room_id).first()
        if session:
            participants = json.loads(session.participants) if session.participants else []
            if int(user_id) in participants:
                participants.remove(int(user_id))
                session.participants = json.dumps(participants)
                db.session.commit()

        emit('left_jam_session', {'room_id': room_id}, room=room_id)
        logger.info(f'User {user_id} left jam session {room_id}')

    except Exception as e:
        logger.error(f"Leave jam session error: {str(e)}")
        emit('error', {'message': 'Failed to leave jam session'})

@socketio.on('jam_session_note')
def handle_jam_session_note(data):
    try:
        # Get token from data
        token = data.get('token')
        user_id = validate_socket_token(token)

        if not user_id:
            emit('error', {'message': 'Authentication required'})
            return

        room_id = data.get('room_id')
        note = data.get('note')

        if not room_id or not note:
            emit('error', {'message': 'Room ID and note required'})
            return

        # Broadcast note to all participants in the room
        emit('jam_session_note_played', {
            'user_id': user_id,
            'note': note,
            'timestamp': datetime.utcnow().isoformat()
        }, room=room_id)

    except Exception as e:
        logger.error(f"Jam session note error: {str(e)}")
        emit('error', {'message': 'Failed to play note'})

# Chatbot endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'Please send a message!'})

        message = data.get('message', '').lower()

        # Simple rule-based chatbot responses
        responses = {
            'hello': 'Hello! Welcome to Swar.AI. How can I help you create beautiful music today?',
            'help': 'I can help you understand raags, music theory, or guide you through the music generation process. What would you like to know?',
            'raag': 'Raags are melodic frameworks in Indian classical music. Each raag has specific notes and evokes particular emotions. Would you like to know about a specific raag?',
            'tempo': 'Tempo is the speed of your music measured in beats per minute (BPM). Slower tempos (60-90 BPM) create calm music, while faster tempos (120-180 BPM) create energetic music.',
            'notes': 'In Indian classical music, we use seven basic notes: Sa, Re, Ga, Ma, Pa, Dha, Ni. These correspond to Do, Re, Mi, Fa, Sol, La, Ti in Western music.',
            'generate': 'To generate music, select a raag, adjust tempo and intensity, choose your mood, and optionally add specific notes. Then click "Generate My Raag!"',
            'default': 'That\'s interesting! I\'m here to help with music creation, raags, and Indian classical music. Feel free to ask me anything about music theory or how to use Swar.AI!'
        }

        # Find appropriate response
        response = responses.get('default')
        for key in responses:
            if key in message:
                response = responses[key]
                break

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.unauthorized_loader
def unauthorized_callback(error):
    return jsonify({'error': 'Authorization token required'}), 401

if __name__ == '__main__':
    # Initialize database
    init_database()
    logger.info("Starting Swar.AI Backend Server...")
    logger.info("Available endpoints:")
    logger.info("- GET  /")
    logger.info("- GET  /api/test")
    logger.info("- POST /api/register")
    logger.info("- POST /api/login")
    logger.info("- GET  /api/raags")
    logger.info("- POST /api/generate-music")
    logger.info("- GET  /api/compositions")
    logger.info("- GET  /api/user/profile")
    logger.info("- POST /api/chat")
    # Run the application
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)