"""
Prediction script for weapon detection model.
This script loads the trained model and performs predictions on input images.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path

class WeaponDetectionModel:
    def __init__(self, model_path='models/weapon_detection_model_complete.pth'):
        """
        Initialize the weapon detection model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load class names from checkpoint or JSON file
        if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
            self.num_classes = checkpoint.get('num_classes', len(self.class_names))
        else:
            # Try to load from JSON file as fallback
            try:
                with open('weapon_classes.json', 'r') as f:
                    self.class_names = json.load(f)
                self.num_classes = len(self.class_names)
            except FileNotFoundError:
                raise ValueError(f"Could not find class names in model checkpoint or weapon_classes.json")
        
        # Load model architecture
        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, self.num_classes)
        )
        
        # Load model weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=1):
        """
        Predict weapon type from an image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            dict: Dictionary containing prediction results
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {
                'error': f'Error loading image: {str(e)}',
                'weapon_detected': 'no',
                'weapon_type': 'Unable to find',
                'confidence': 0.01
            }
        
        # Perform prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # Get top predictions
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'weapon_type': self.class_names[idx.item()],
                'confidence': prob.item()
            })
        
        # Format results
        top_prediction = predictions[0]
        confidence_threshold = 0.5  # Threshold for weapon detection
        
        weapon_detected = 'yes' if top_prediction['confidence'] >= confidence_threshold else 'no'
        
        # If no weapon detected, set weapon_type to "Unable to find" and confidence to 0
        if weapon_detected == 'no':
            weapon_type = 'Unable to find'
            confidence = 0.0
        else:
            weapon_type = top_prediction['weapon_type']
            confidence = round(top_prediction['confidence'] * 100, 2)
        
        result = {
            'weapon_detected': weapon_detected,
            'weapon_type': weapon_type,
            'confidence': confidence,
            'all_predictions': predictions
        }
        
        return result

# Global model instance
_model_instance = None

def get_model():
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = WeaponDetectionModel()
    return _model_instance

if __name__ == "__main__":
    # Test the model
    model = WeaponDetectionModel()
    print("Model loaded successfully!")
    print(f"Classes: {model.class_names}")

