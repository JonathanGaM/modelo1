from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# =====================================================================
# FUNCIONES DEL MODELO (las mismas que generaste)
# =====================================================================

def preprocess_input(data):
    """
    Procesa los datos de entrada para el modelo
    """
    df = pd.DataFrame([data])
    
    # 1. Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 2. Asignar título basado en Sexo y Edad
    def assign_title(row):
        if row['Sex'] == 'male':
            if row['Age'] < 16:
                return 'Master'
            else:
                return 'Mr'
        else:
            if row['Age'] < 16:
                return 'Miss'
            elif row['Age'] < 35:
                return 'Miss'
            else:
                return 'Mrs'
    
    df['Title'] = df.apply(assign_title, axis=1)
    
    # 3. Crear grupos de edad y fare
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    df['FareBin'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31.0, 512.33], labels=['Low', 'Medium', 'High', 'Luxury'])
    
    # 4. Variables de cabina
    df['HasCabin'] = df['HasCabin'].astype(int)
    
    def assign_cabin_letter(row):
        if row['HasCabin'] == 0:
            return 'Unknown'
        else:
            if row['Pclass'] == 1:
                return 'B'
            elif row['Pclass'] == 2:
                return 'D'
            else:
                return 'F'
    
    df['CabinLetter'] = df.apply(assign_cabin_letter, axis=1)
    
    # 5. Variables de interacción
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    
    # 6. Codificar variables categóricas
    categorical_encodings = {
        'Sex': {'male': 1, 'female': 0},
        'Embarked': {'C': 0, 'Q': 1, 'S': 2},
        'Title': {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4},
        'FareBin': {'Low': 0, 'Medium': 1, 'High': 2, 'Luxury': 3},
        'CabinLetter': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'Unknown': 8}
    }
    
    for col, encoding in categorical_encodings.items():
        df[col] = df[col].map(encoding).fillna(0)
    
    # 7. Seleccionar características finales
    final_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 
                     'IsAlone', 'Title', 'FareBin', 'HasCabin', 'CabinLetter', 'Age_Pclass']
    
    return df[final_features].values

def predict_survival(passenger_data, model, scaler, pca):
    """
    Predice supervivencia para un pasajero
    """
    X_processed = preprocess_input(passenger_data)
    X_scaled = scaler.transform(X_processed)
    X_pca = pca.transform(X_scaled)
    
    prediction = model.predict(X_pca)[0]
    probability = model.predict_proba(X_pca)[0]
    
    return prediction, probability

def validate_input(data):
    """
    Valida los datos de entrada
    """
    errors = []
    
    # Validaciones básicas
    try:
        pclass = int(data.get('pclass', 0))
        if pclass not in [1, 2, 3]:
            errors.append("La clase debe ser 1, 2 o 3")
    except (ValueError, TypeError):
        errors.append("La clase debe ser un número válido")
    
    if data.get('sex') not in ['male', 'female']:
        errors.append("El sexo debe ser 'male' o 'female'")
    
    try:
        age = float(data.get('age', 0))
        if age < 0 or age > 120:
            errors.append("La edad debe estar entre 0 y 120 años")
    except (ValueError, TypeError):
        errors.append("La edad debe ser un número válido")
    
    try:
        fare = float(data.get('fare', 0))
        if fare < 0:
            errors.append("El precio del ticket debe ser positivo")
    except (ValueError, TypeError):
        errors.append("El precio del ticket debe ser un número válido")
    
    if data.get('embarked') not in ['C', 'Q', 'S']:
        errors.append("El puerto de embarque debe ser 'C', 'Q' o 'S'")
    
    return errors

# =====================================================================
# CARGAR MODELO AL INICIAR LA APLICACIÓN
# =====================================================================

model_package = None

def load_model():
    """
    Carga el modelo PKL al iniciar la aplicación
    """
    global model_package
    try:
        model_path = 'titanic_pca_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            print("✅ Modelo cargado exitosamente")
            return True
        else:
            print("❌ Archivo titanic_pca_model.pkl no encontrado")
            return False
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

# Cargar modelo al iniciar
load_model()

# =====================================================================
# RUTAS DE LA APLICACIÓN WEB
# =====================================================================

@app.route('/')
def index():
    """
    Página principal con el formulario
    """
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones
    """
    try:
        # Verificar que el modelo esté cargado
        if model_package is None:
            return jsonify({
                "status": "error",
                "mensaje": "Modelo no disponible. Contacta al administrador.",
                "error": "Model not loaded"
            }), 500
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "mensaje": "No se recibieron datos",
                "error": "No JSON data received"
            }), 400
        
        # Validar datos de entrada
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({
                "status": "error",
                "mensaje": "Datos inválidos",
                "error": "; ".join(validation_errors)
            }), 400
        
        # Preparar datos para el modelo
        passenger_data = {
            'Pclass': int(data.get('pclass')),
            'Sex': str(data.get('sex')).lower(),
            'Age': float(data.get('age')),
            'SibSp': int(data.get('sibsp', 0)),
            'Parch': int(data.get('parch', 0)),
            'Fare': float(data.get('fare')),
            'Embarked': str(data.get('embarked', 'S')).upper(),
            'HasCabin': 1 if data.get('has_cabin') else 0
        }
        
        # Hacer predicción
        prediction, probability = predict_survival(
            passenger_data,
            model_package['model'],
            model_package['scaler'],
            model_package['pca']
        )
        
        # Convertir numpy types a Python types para JSON
        prediction = bool(prediction)
        prob_survival = float(probability[1] * 100)
        prob_death = float(probability[0] * 100)
        
        # Preparar respuesta
        result = {
            "status": "success",
            "sobrevive": prediction,
            "probabilidad_supervivencia": round(prob_survival, 1),
            "probabilidad_muerte": round(prob_death, 1),
            "mensaje": "✅ Sobrevivio" if prediction else "❌ No sobrevivio",
            "datos_procesados": {
                "clase": passenger_data['Pclass'],
                "sexo": passenger_data['Sex'],
                "edad": passenger_data['Age'],
                "precio": passenger_data['Fare'],
                "puerto": passenger_data['Embarked'],
                "familia_total": passenger_data['SibSp'] + passenger_data['Parch'] + 1,
                "tiene_cabina": bool(passenger_data['HasCabin'])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({
            "status": "error",
            "mensaje": "Error interno del servidor",
            "error": str(e)
        }), 500

@app.route('/health')
def health():
    """
    Endpoint para verificar el estado de la aplicación
    """
    model_status = "OK" if model_package is not None else "ERROR"
    return jsonify({
        "status": "healthy",
        "model_loaded": model_status,
        "version": "1.0"
    })

@app.route('/model-info')
def model_info():
    """
    Información del modelo cargado
    """
    if model_package is None:
        return jsonify({
            "status": "error",
            "mensaje": "Modelo no cargado"
        }), 500
    
    return jsonify({
        "status": "success",
        "modelo_info": {
            "componentes_pca": model_package['pca_components'],
            "caracteristicas": model_package['feature_names'],
            "version": model_package.get('version', 'unknown'),
            "reduccion_dimensional": f"{len(model_package['feature_names'])} → {model_package['pca_components']}"
        }
    })

# =====================================================================
# CONFIGURACIÓN Y EJECUCIÓN
# =====================================================================

if __name__ == '__main__':
    print("🚢 Iniciando Servidor del Predictor Titanic...")
    print("="*50)
    
    # Verificar que el modelo esté disponible
    if model_package is None:
        print("⚠️  ADVERTENCIA: Modelo no cargado. Verifica que 'titanic_pca_model.pkl' exista.")
        print("   Puedes generar el modelo ejecutando el script de entrenamiento.")
    else:
        print("✅ Modelo cargado correctamente")
        print(f"   📊 Componentes PCA: {model_package['pca_components']}")
        print(f"   🔧 Características: {len(model_package['feature_names'])}")
    
    print("\n🌐 Servidor disponible en:")
    print("   Local: http://127.0.0.1:5000")
    print("   Red:   http://0.0.0.0:5000")
    print("\n🎯 Endpoints disponibles:")
    print("   GET  /          - Formulario principal")
    print("   POST /predict   - API de predicción")
    print("   GET  /health    - Estado del servidor")
    print("   GET  /model-info - Información del modelo")
    print("="*50)
    
    # Configuración del servidor
    app.run(
        host='0.0.0.0',  # Permite conexiones externas
        port=5000,       # Puerto estándar
        debug=True,      # Modo desarrollo (cambiar a False en producción)
        threaded=True    # Permite múltiples requests simultáneos
    )