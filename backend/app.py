from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Model path
model_path = r"D:\Dropbox\29. Ampelos\28_dep\test_09_06_2025\ML_Agent_Code\ab_agent_model_2.pth"

# Define the PyTorch model architecture (must match training)
class ABClassifier(nn.Module):
    """Neural network classifier for AB decisions"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ABDecisionPredictor:
    """Interactive AB Decision Predictor using trained PyTorch model - 56 Features"""
    
    def __init__(self, model_path, threshold=0.5):
        self.model = None
        self.threshold = threshold
        self.case_history = {}
        self.input_dim = 55  # Corrected: 55 features, not 56!
        
        # Define all expected features (must match training order) - 55 features
        self.feature_names = [
            # Core features (2) - NO progress_depth!
            'depth_reached', 'event_duration',
            
            # Obstruction classes (7)
            'obstr_CASING_ISSUES', 'obstr_CEMENT_RELATED', 'obstr_DEBRIS_AND_FOREIGN_OBJECTS',
            'obstr_EQUIPMENT_PROBLEMS', 'obstr_FLUID_AND_FLOW_ISSUES',
            'obstr_FORMATION_AND_BOTTOM_ISSUES', 'obstr_HOLE_INTEGRITY_AND_COLLAPSE',
            
            # Deployment methods (9)
            'deploy_Cable Tool', 'deploy_Double Pole', 'deploy_Dozer Rig',
            'deploy_Drilling Rig', 'deploy_Power Swivel', 'deploy_Pulling Rig',
            'deploy_Service Rig', 'deploy_Single Pole', 'deploy_Wireline',
            
            # Tools (32)
            'tool_CCL', 'tool_Hydraulic Jars', 'tool_Stem', 'tool_Cable Tool Jars',
            'tool_GR', 'tool_Mud pump', 'tool_Camera', 'tool_Friction Socket',
            'tool_Circulation', 'tool_Drilling Rig', 'tool_Shooting', 'tool_Power Swivel',
            'tool_Mill', 'tool_Bit', 'tool_Magnet', 'tool_Chisel Bit', 'tool_Drill Pipe',
            'tool_CBL', 'tool_Spear', 'tool_Center Spear', 'tool_Mouse Trap Socket',
            'tool_Tri-cone', 'tool_Overshot', 'tool_Grapple', 'tool_Jet Cutter',
            'tool_Logging tool', 'tool_Bailer', 'tool_Washover', 'tool_Sand pump',
            'tool_Cable Tool', 'tool_Chemical Cutter', 'tool_Perforator',
            
            # Engineered features (5)
            'confidence_num', 'depth_ratio', 'num_previous_events',
            'cumulative_duration', 'tools_cum_unique'
        ]
        
        # Obstruction mapping (frontend -> model)
        self.obstruction_mapping = {
            'CASING_ISSUES': 'obstr_CASING_ISSUES',
            'CEMENT_RELATED': 'obstr_CEMENT_RELATED',
            'DEBRIS_AND_FOREIGN_OBJECTS': 'obstr_DEBRIS_AND_FOREIGN_OBJECTS',
            'EQUIPMENT_PROBLEMS': 'obstr_EQUIPMENT_PROBLEMS',
            'FLUID_AND_FLOW_ISSUES': 'obstr_FLUID_AND_FLOW_ISSUES',
            'FORMATION_AND_BOTTOM_ISSUES': 'obstr_FORMATION_AND_BOTTOM_ISSUES',
            'HOLE_INTEGRITY_AND_COLLAPSE': 'obstr_HOLE_INTEGRITY_AND_COLLAPSE'
        }
        
        # Deployment mapping
        self.deployment_mapping = {
            'Cable Tool': 'deploy_Cable Tool',
            'Double Pole': 'deploy_Double Pole',
            'Dozer Rig': 'deploy_Dozer Rig',
            'Drilling Rig': 'deploy_Drilling Rig',
            'Power Swivel': 'deploy_Power Swivel',
            'Pulling Rig': 'deploy_Pulling Rig',
            'Service Rig': 'deploy_Service Rig',
            'Single Pole': 'deploy_Single Pole',
            'Wireline': 'deploy_Wireline'
        }
        
        # Confidence mapping
        self.confidence_mapping = {
            'low': 0.0,
            'medium': 0.5,
            'high': 1.0
        }
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            self.model = ABClassifier(input_dim=self.input_dim)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print(f"✓ Model loaded successfully from: {model_path}")
            print(f"  Model architecture: {self.input_dim} → 128 → 64 → 1")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def start_new_case(self, case_id, attainable_bottom=None):
        """Start tracking a new case"""
        self.case_history[case_id] = {
            'events': [],
            'attainable_bottom': attainable_bottom,
            'step_count': 0
        }
        print(f"Started new case: {case_id}")
        if attainable_bottom:
            print(f"  Target depth: {attainable_bottom} ft")
    
    def add_event(self, case_id, depth_reached, event_duration,
                  obstruction_class, deployment_method, tools_used, confidence):
        """Add a new event to ongoing case"""
        
        if case_id not in self.case_history:
            print(f"Case {case_id} not found. Starting new case...")
            self.start_new_case(case_id)
        
        case_data = self.case_history[case_id]
        case_data['step_count'] += 1
        
        event = {
            'step': case_data['step_count'],
            'depth_reached': depth_reached,
            'event_duration': event_duration,
            'obstruction_class': obstruction_class,
            'deployment_method': deployment_method,
            'tools_used': tools_used or [],
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        case_data['events'].append(event)
        print(f"  Added event #{case_data['step_count']}: depth={depth_reached}ft, duration={event_duration}hrs")
    
    def _compute_features(self, case_id):
        """Compute all 56 features for current case state"""
        case_data = self.case_history[case_id]
        events = case_data['events']
        
        if not events:
            return None
        
        print(f"\n{'='*60}")
        print(f"FEATURE COMPUTATION DEBUG - Case: {case_id}")
        print(f"{'='*60}")
        print(f"Total events: {len(events)}")
        print(f"Target depth: {case_data['attainable_bottom']} ft")
        
        features = {}
        current_event = events[-1]
        
        print(f"\nCurrent Event #{current_event['step']}:")
        print(f"  depth_reached: {current_event['depth_reached']}")
        print(f"  event_duration: {current_event['event_duration']}")
        print(f"  obstruction_class: {current_event['obstruction_class']}")
        print(f"  deployment_method: {current_event['deployment_method']}")
        print(f"  tools_used: {current_event['tools_used']}")
        print(f"  confidence: {current_event['confidence']}")
        
        # 1. Core features (2 features) - NO progress_depth
        features['depth_reached'] = current_event['depth_reached'] or 0
        features['event_duration'] = current_event['event_duration'] or 0
        
        # NOTE: progress_depth is NOT in the training features, so we skip it
        
        # 2. Obstruction class one-hot (7 features)
        obstruction_classes = [
            'CASING_ISSUES', 'CEMENT_RELATED', 'DEBRIS_AND_FOREIGN_OBJECTS',
            'EQUIPMENT_PROBLEMS', 'FLUID_AND_FLOW_ISSUES',
            'FORMATION_AND_BOTTOM_ISSUES', 'HOLE_INTEGRITY_AND_COLLAPSE'
        ]
        
        print(f"\nObstruction Mapping:")
        print(f"  Input: {current_event['obstruction_class']}")
        current_obs = self.obstruction_mapping.get(current_event['obstruction_class'])
        print(f"  Mapped to: {current_obs}")
        
        for obs_class in obstruction_classes:
            features[f'obstr_{obs_class}'] = 1 if current_obs == f'obstr_{obs_class}' else 0
            if features[f'obstr_{obs_class}'] == 1:
                print(f"  ✓ obstr_{obs_class} = 1")
        
        # 3. Deployment method one-hot (9 features)
        deployment_methods = [
            'Cable Tool', 'Double Pole', 'Dozer Rig', 'Drilling Rig',
            'Power Swivel', 'Pulling Rig', 'Service Rig', 'Single Pole', 'Wireline'
        ]
        
        print(f"\nDeployment Mapping:")
        print(f"  Input: {current_event['deployment_method']}")
        current_deploy = self.deployment_mapping.get(current_event['deployment_method'])
        print(f"  Mapped to: {current_deploy}")
        
        for method in deployment_methods:
            features[f'deploy_{method}'] = 1 if current_deploy == f'deploy_{method}' else 0
            if features[f'deploy_{method}'] == 1:
                print(f"  ✓ deploy_{method} = 1")
        
        # 4. Tools one-hot (32 features)
        all_tool_features = [
            'CCL', 'Hydraulic Jars', 'Stem', 'Cable Tool Jars', 'GR', 'Mud pump',
            'Camera', 'Friction Socket', 'Circulation', 'Drilling Rig', 'Shooting',
            'Power Swivel', 'Mill', 'Bit', 'Magnet', 'Chisel Bit', 'Drill Pipe',
            'CBL', 'Spear', 'Center Spear', 'Mouse Trap Socket', 'Tri-cone',
            'Overshot', 'Grapple', 'Jet Cutter', 'Logging tool', 'Bailer',
            'Washover', 'Sand pump', 'Cable Tool', 'Chemical Cutter', 'Perforator'
        ]
        
        print(f"\nTools Mapping:")
        print(f"  Input tools: {current_event['tools_used']}")
        current_tools = set(current_event['tools_used'])
        print(f"  Matched tools:")
        
        tool_count = 0
        for tool in all_tool_features:
            features[f'tool_{tool}'] = 1 if tool in current_tools else 0
            if features[f'tool_{tool}'] == 1:
                print(f"    ✓ tool_{tool} = 1")
                tool_count += 1
        print(f"  Total tools activated: {tool_count}")
        
        # 5. Engineered features (5 features)
        print(f"\nEngineered Features:")
        
        # confidence_num
        features['confidence_num'] = self.confidence_mapping.get(
            current_event['confidence'], 0.5
        )
        print(f"  confidence_num = {features['confidence_num']} (from '{current_event['confidence']}')")
        
        # depth_ratio
        if case_data['attainable_bottom'] and case_data['attainable_bottom'] > 0:
            features['depth_ratio'] = features['depth_reached'] / case_data['attainable_bottom']
            print(f"  depth_ratio = {features['depth_reached']} / {case_data['attainable_bottom']} = {features['depth_ratio']:.4f}")
        else:
            features['depth_ratio'] = 0
            print(f"  depth_ratio = 0 (no target depth)")
        
        # num_previous_events
        features['num_previous_events'] = len(events) - 1
        print(f"  num_previous_events = {features['num_previous_events']}")
        
        # cumulative_duration
        features['cumulative_duration'] = sum(e['event_duration'] for e in events)
        print(f"  cumulative_duration = {features['cumulative_duration']} hrs")
        
        # tools_cum_unique
        all_tools_used = set()
        for e in events:
            all_tools_used.update(e['tools_used'])
        features['tools_cum_unique'] = len(all_tools_used)
        print(f"  tools_cum_unique = {features['tools_cum_unique']} (unique tools: {all_tools_used})")
        
        print(f"\n{'='*60}\n")
        
        return features
    
    def predict_ab_decision(self, case_id):
        """Predict AB decision for current case state"""
        
        if case_id not in self.case_history:
            return {"error": f"Case {case_id} not found"}
        
        if not self.model:
            return {"error": "Model not loaded"}
        
        try:
            features = self._compute_features(case_id)
            if features is None:
                return {"error": "No events found for case"}
        except Exception as e:
            return {"error": f"Feature computation error: {e}"}
        
        # Create feature vector in correct order
        print(f"{'='*60}")
        print(f"FEATURE VECTOR CONSTRUCTION")
        print(f"{'='*60}")
        
        feature_vector = []
        missing_features = []
        non_zero_features = []
        
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in features:
                val = float(features[feat_name])
                feature_vector.append(val)
                if val != 0:
                    non_zero_features.append((i, feat_name, val))
            else:
                feature_vector.append(0.0)
                missing_features.append(feat_name)
        
        print(f"Total features: {len(feature_vector)}")
        print(f"Non-zero features ({len(non_zero_features)}):")
        for idx, name, val in non_zero_features:
            print(f"  [{idx:2d}] {name:30s} = {val}")
        
        if missing_features:
            print(f"\nWarning: Missing features ({len(missing_features)}): {missing_features[:5]}...")
        
        # Convert to PyTorch tensor
        X = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        
        print(f"\nInput tensor shape: {X.shape}")
        print(f"Input tensor stats:")
        print(f"  Min: {X.min().item():.4f}")
        print(f"  Max: {X.max().item():.4f}")
        print(f"  Mean: {X.mean().item():.4f}")
        print(f"  Non-zero count: {(X != 0).sum().item()}")
        
        try:
            with torch.no_grad():
                logits = self.model(X)
                probability = torch.sigmoid(logits).item()
            
            print(f"\nModel Output:")
            print(f"  Raw logit: {logits.item():.6f}")
            print(f"  Probability (sigmoid): {probability:.6f}")
            print(f"  Threshold: {self.threshold}")
            
            ab_decision = probability >= self.threshold
            
            result = {
                'case_id': str(case_id),
                'step': int(self.case_history[case_id]['step_count']),
                'hazard_rate': round(probability, 3),
                'threshold': float(self.threshold),
                'recommendation': 'ABANDON' if ab_decision else 'CONTINUE',
                'confidence': 'HIGH' if abs(probability - self.threshold) > 0.2 else 'MEDIUM'
            }
            
            print(f"\nFinal Prediction: {result['recommendation']} (P={probability:.3f})")
            print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction error: {e}"}

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor with the model"""
    global predictor
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print("="*60)
        print("Initializing AB Decision Predictor (55 Features)")
        print("="*60)
        predictor = ABDecisionPredictor(model_path, threshold=0.5)
        
        if predictor.model is not None:
            print("✓ Predictor initialized successfully!")
            
            # Test prediction
            try:
                test_tensor = torch.zeros(1, 55)
                with torch.no_grad():
                    test_output = predictor.model(test_tensor)
                    test_prob = torch.sigmoid(test_output).item()
                print(f"✓ Model test successful (test output: {test_prob:.3f})")
            except Exception as e:
                print(f"⚠ Model test failed: {e}")
            
            print("="*60)
            return True
        else:
            print("✗ Model loading failed")
            return False
        
    except Exception as e:
        print(f"✗ Error initializing predictor: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None and predictor.model is not None,
        'model_path_exists': os.path.exists(model_path),
        'model_type': 'PyTorch ABClassifier (55 features)',
        'num_features': 55
    })

@app.route('/api/start-case', methods=['POST'])
def start_case():
    """Start a new case"""
    global predictor
    
    if predictor is None or predictor.model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    try:
        data = request.get_json()
        case_id = data.get('case_id')
        attainable_bottom = data.get('attainable_bottom')
        
        if not case_id or not attainable_bottom:
            return jsonify({
                'success': False,
                'error': 'case_id and attainable_bottom are required'
            }), 400
        
        predictor.start_new_case(case_id, attainable_bottom)
        
        return jsonify({
            'success': True,
            'message': f'Case {case_id} started successfully',
            'case_id': case_id,
            'attainable_bottom': attainable_bottom
        })
        
    except Exception as e:
        print(f"Error starting case: {e}")
        return jsonify({
            'success': False,
            'error': f'Error starting case: {str(e)}'
        }), 500

@app.route('/api/add-event-predict', methods=['POST'])
def add_event_and_predict():
    """Add an event and get prediction"""
    global predictor
    
    if predictor is None or predictor.model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    try:
        data = request.get_json()
        case_id = data.get('case_id')
        depth_reached = data.get('depth_reached')
        event_duration = data.get('event_duration')
        obstruction_class = data.get('obstruction_class')
        deployment_method = data.get('deployment_method')
        tools_used = data.get('tools_used', [])
        confidence = data.get('confidence', 'medium')
        
        # Validate required fields
        if not all([case_id, depth_reached is not None, event_duration is not None, 
                    obstruction_class, deployment_method]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: case_id, depth_reached, event_duration, obstruction_class, deployment_method'
            }), 400
        
        # Add event
        predictor.add_event(
            case_id=case_id,
            depth_reached=float(depth_reached),
            event_duration=float(event_duration),
            obstruction_class=obstruction_class,
            deployment_method=deployment_method,
            tools_used=tools_used,
            confidence=confidence
        )
        
        # Get prediction
        prediction_result = predictor.predict_ab_decision(case_id)
        
        if 'error' in prediction_result:
            return jsonify({
                'success': False,
                'error': prediction_result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'result': prediction_result
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AB Decision Predictor Flask Backend")
    print("Model: PyTorch ABClassifier with 55 features")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print()
    
    if initialize_predictor():
        print("Starting Flask server on http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize predictor. Server not started.")
        print("Please check:")
        print("1. Model file exists at the specified path")
        print("2. Model was trained with 55 features")
        print("3. PyTorch is installed correctly")