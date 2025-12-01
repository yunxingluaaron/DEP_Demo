import pickle
import sys
import types
import os

# =============================================================================
# Custom Unpickler to Handle Missing Classes
# =============================================================================

class ABSurvivalModel:
    """
    Dummy class to replace the missing ABSurvivalModel during unpickling
    """
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def predict_hazard(self, X):
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            elif hasattr(self.model, 'predict'):
                return self.model.predict(X)
        return [0.5]  # Default fallback
    
    def predict_proba(self, X):
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            elif hasattr(self.model, 'predict'):
                pred = self.model.predict(X)
                # Convert single predictions to probability format
                return [[1-p, p] for p in pred]
        return [[0.5, 0.5]]  # Default fallback

class ModelLoader:
    """
    Custom model loader that handles missing class definitions
    """
    
    @staticmethod
    def load_model_with_fallback(model_path):
        """
        Try to load the model with various fallback strategies
        """
        
        # Strategy 1: Try direct loading
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully (direct)")
            return model
        except Exception as e:
            print(f"Direct loading failed: {e}")
        
        # Strategy 2: Add missing class to global namespace and try again
        try:
            # Add the missing class to the main module
            sys.modules['__main__'].ABSurvivalModel = ABSurvivalModel
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully (with class injection)")
            return model
        except Exception as e:
            print(f"Class injection loading failed: {e}")
        
        # Strategy 3: Custom unpickler
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'ABSurvivalModel':
                        return ABSurvivalModel
                    return super().find_class(module, name)
            
            with open(model_path, 'rb') as f:
                model = CustomUnpickler(f).load()
            print("✓ Model loaded successfully (custom unpickler)")
            return model
        except Exception as e:
            print(f"Custom unpickler failed: {e}")
        
        # Strategy 4: Try to extract the underlying sklearn model
        try:
            import dill
            with open(model_path, 'rb') as f:
                model = dill.load(f)
            print("✓ Model loaded successfully (dill)")
            return model
        except Exception as e:
            print(f"Dill loading failed: {e}")
        
        print("✗ All loading strategies failed")
        return None

# =============================================================================
# Test the model loading
# =============================================================================

def test_model_loading(model_path):
    """Test loading the model and check its capabilities"""
    
    print(f"Testing model loading from: {model_path}")
    
    # Try to load the model
    model = ModelLoader.load_model_with_fallback(model_path)
    
    if model is None:
        print("✗ Failed to load model")
        return None
    
    print(f"✓ Model loaded: {type(model)}")
    
    # Test model capabilities
    try:
        import numpy as np
        test_features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3 + [1, 1]]).reshape(1, -1)  # 32 features
        
        if hasattr(model, 'predict_hazard'):
            result = model.predict_hazard(test_features)
            print(f"✓ predict_hazard works: {result}")
        elif hasattr(model, 'predict_proba'):
            result = model.predict_proba(test_features)
            print(f"✓ predict_proba works: {result}")
        elif hasattr(model, 'predict'):
            result = model.predict(test_features)
            print(f"✓ predict works: {result}")
        else:
            print("✗ No prediction methods found")
            
    except Exception as e:
        print(f"✗ Model testing failed: {e}")
    
    return model

if __name__ == "__main__":
    # Test with your model path
    model_path = r"D:\Dropbox\29. Ampelos\28_dep\test_09_06_2025\Demo_Code\model\ab_survival_model.pkl"
    
    if os.path.exists(model_path):
        model = test_model_loading(model_path)
    else:
        print(f"Model file not found: {model_path}")