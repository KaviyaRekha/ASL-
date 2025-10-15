# models/create_placeholder_models.py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

def create_placeholder_models():
    """Create placeholder models for testing"""
    print("🤖 Creating placeholder ASL models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create dummy training data (21 landmarks * 3 coordinates = 63 features)
    n_samples = 100
    n_features = 63  # 21 hand landmarks * 3 coordinates (x,y,z)
    
    X_dummy = np.random.randn(n_samples, n_features)
    y_dummy = np.array(['A', 'B', 'C', 'D', 'E'] * 20)  # 5 gestures repeated
    
    # 1. Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    # 2. Create and fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_dummy)
    
    # 3. Create and fit KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    y_encoded = label_encoder.transform(y_dummy)
    knn.fit(X_dummy, y_encoded)
    
    # Save models
    with open('models/asl_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/asl_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('models/asl_knn.pkl', 'wb') as f:
        pickle.dump(knn, f)
    
    with open('models/asl_update_knn.pkl', 'wb') as f:
        pickle.dump(knn, f)  # Same model for now
    
    print("✅ Placeholder models created successfully!")
    print("📁 Model files generated in 'models/' folder:")
    print("   - asl_knn.pkl")
    print("   - asl_update_knn.pkl") 
    print("   - asl_scaler.pkl")
    print("   - asl_label_encoder.pkl")

if __name__ == "__main__":
    create_placeholder_models()
