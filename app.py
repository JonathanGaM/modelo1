# app.py - VERSIÓN SIMPLIFICADA PARA PREDICCIÓN DE ASISTENCIA
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import mysql.connector
import os
from dotenv import load_dotenv
import pickle
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

# DB Config
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),           
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),           
    'password': os.getenv('DB_PASSWORD', ''),  
    'database': os.getenv('DB_NAME'),       
    'charset': 'utf8mb4'
}

# ML Variables
modelo_ml = None
scaler_ml = None
encoders_ml = None

def cargar_modelo():
    global modelo_ml, scaler_ml, encoders_ml
    try:
        # CARGAR TU MODELO REAL: sututeh_model.pkl
        with open('sututeh_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Extraer componentes del modelo
        modelo_ml = model_data.get('prediction_model') or model_data.get('model')
        scaler_ml = model_data.get('scaler')
        
        # Crear encoders desde los que tienes
        encoders_ml = {
            'genero': model_data.get('le_genero_pred') or model_data.get('le_genero'),
            'puesto_universidad': model_data.get('le_puesto_pred') or model_data.get('le_puesto')
        }
        
        print("✅ Modelo sututeh_model.pkl cargado")
        print(f"✅ Componentes: modelo={modelo_ml is not None}, scaler={scaler_ml is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Error cargando sututeh_model.pkl: {e}")
        return False

def predecir_asistencia(usuario_data):
    """Predicción simplificada: Asistirá/No Asistirá"""
    if not modelo_ml or not encoders_ml:
        return None
    
    try:
        # Usar encoders del modelo cargado
        le_genero = encoders_ml['genero']
        le_puesto = encoders_ml['puesto_universidad']
        
        # Datos del usuario
        puntaje_promedio = usuario_data.get('puntaje_promedio', 2.0)
        antiguedad = usuario_data.get('antiguedad', 2)
        puesto_sindicato = usuario_data.get('puesto_sindicato', 0)
        genero = usuario_data.get('genero', 'Femenino')
        puesto_universidad = usuario_data.get('puesto_universidad', 'Docente')
        
        # Encode con manejo de errores
        try:
            genero_encoded = int(le_genero.transform([genero])[0]) if le_genero else (0 if genero == 'Femenino' else 1)
        except:
            genero_encoded = 0 if genero == 'Femenino' else 1
        
        try:
            puesto_encoded = int(le_puesto.transform([puesto_universidad])[0]) if le_puesto else 0
        except:
            puesto_encoded = 0
        
        # Features para el modelo
        features = [
            antiguedad,          
            puesto_sindicato,    
            genero_encoded,      
            puesto_encoded,      
            puntaje_promedio     
        ]
        
        # Predicción directa (sin scaler por desajuste)
        features_array = np.array([features])
        probabilidad = modelo_ml.predict_proba(features_array)[0, 1]
        
        # SIMPLIFICADO: Solo 2 categorías
        # Umbral más equilibrado para mejor distribución
        umbral = 0.5  # 50% de probabilidad como corte
        
        if probabilidad >= umbral:
            prediccion = 'ASISTIRA'
            cluster_simple = 'Activistas Comprometidos'  # Mantenemos para compatibilidad
        else:
            prediccion = 'NO_ASISTIRA'
            cluster_simple = 'Inactivos Críticos'
        
        return {
            'prediccion': prediccion,
            'probabilidad': float(probabilidad),
            'probabilidad_texto': f"{probabilidad:.1%}",
            'cluster': cluster_simple,  # Para compatibilidad con frontend
            'asistira': bool(probabilidad >= umbral),
            'en_riesgo': bool(probabilidad < umbral)  # Para compatibilidad
        }
        
    except Exception as e:
        print(f"Error predicción: {e}")
        return None

def conectar_bd():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error BD: {e}")
        return None

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/api/datos')
def get_datos():
    try:
        if not modelo_ml:
            return jsonify({
                'success': False,
                'error': 'Modelo ML no disponible'
            }), 500
        
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Error conexión BD'
            }), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Query usando la vista de usuarios
        query = """
        SELECT 
            usuario_id,
            nombre_completo,
            genero,
            antiguedad,
            puesto_sindicato,
            puesto_universidad,
            puntaje_promedio,
            total_reuniones_recientes,
            asistencias_efectivas,
            tasa_asistencia
        FROM vw_usuarios_ml_agregado
        WHERE puntaje_promedio IS NOT NULL
        ORDER BY puntaje_promedio DESC
        """
        
        cursor.execute(query)
        usuarios_bd = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"🔍 Usuarios obtenidos de BD: {len(usuarios_bd)}")
        
        # Procesar con ML SIMPLIFICADO
        usuarios = []
        asistiran = 0
        no_asistiran = 0
        
        for usuario in usuarios_bd:
            prediccion = predecir_asistencia({
                'antiguedad': usuario['antiguedad'],
                'puesto_sindicato': usuario['puesto_sindicato'],
                'genero': usuario['genero'],
                'puesto_universidad': usuario['puesto_universidad'],
                'puntaje_promedio': float(usuario['puntaje_promedio']) if usuario['puntaje_promedio'] else 0
            })
            
            if prediccion:
                usuario_final = {
                    'id': int(usuario['usuario_id']),
                    'nombre': usuario['nombre_completo'] or f"Usuario {usuario['usuario_id']}",
                    'cluster': prediccion['cluster'],  # Para compatibilidad
                    'prediccion': prediccion['prediccion'],
                    'asistira': prediccion['asistira'],
                    'probabilidad': float(prediccion['probabilidad']),
                    'probabilidad_texto': prediccion['probabilidad_texto'],
                    'en_riesgo': bool(prediccion['en_riesgo']),  # Para compatibilidad
                    'puntaje': round(float(usuario['puntaje_promedio']) if usuario['puntaje_promedio'] else 0, 2),
                    'reuniones': int(usuario['total_reuniones_recientes']) if usuario['total_reuniones_recientes'] else 0,
                    'tasa_asistencia': round(float(usuario['tasa_asistencia']) if usuario['tasa_asistencia'] else 0, 3)
                }
                
                usuarios.append(usuario_final)
                
                # Contadores SIMPLIFICADOS
                if prediccion['asistira']:
                    asistiran += 1
                else:
                    no_asistiran += 1
        
        print(f"✅ Usuarios procesados: {len(usuarios)} | Asistirán: {asistiran} | No Asistirán: {no_asistiran}")
        
        # Clusters simplificados para compatibilidad
        clusters_compatibilidad = {
            'Activistas Comprometidos': asistiran,
            'Inactivos Críticos': no_asistiran
        }
        
        return jsonify({
            'success': True,
            'usuarios': usuarios,
            'total': len(usuarios),
            'clusters': clusters_compatibilidad,  # Para compatibilidad
            'asistiran': asistiran,
            'no_asistiran': no_asistiran,
            'riesgo': no_asistiran,  # Para compatibilidad
            'modelo_activo': True
        })
        
    except Exception as e:
        print(f"❌ Error en get_datos: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering')
def get_clustering():
    """Endpoint para datos de clasificación y predicción"""
    try:
        if not modelo_ml:
            return jsonify({
                'success': False,
                'error': 'Modelo ML no disponible'
            }), 500
        
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Error conexión BD'
            }), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Misma query que en /api/datos
        query = """
        SELECT 
            usuario_id,
            nombre_completo,
            genero,
            antiguedad,
            puesto_sindicato,
            puesto_universidad,
            puntaje_promedio,
            total_reuniones_recientes,
            asistencias_efectivas,
            tasa_asistencia
        FROM vw_usuarios_ml_agregado
        WHERE puntaje_promedio IS NOT NULL
        ORDER BY puntaje_promedio DESC
        """
        
        cursor.execute(query)
        usuarios_bd = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Procesar datos
        usuarios_data = []
        asistiran = 0
        no_asistiran = 0
        
        for usuario in usuarios_bd:
            prediccion = predecir_asistencia({
                'antiguedad': usuario['antiguedad'],
                'puesto_sindicato': usuario['puesto_sindicato'],
                'genero': usuario['genero'],
                'puesto_universidad': usuario['puesto_universidad'],
                'puntaje_promedio': float(usuario['puntaje_promedio']) if usuario['puntaje_promedio'] else 0
            })
            
            if prediccion:
                # Clasificación por puntaje (tu sistema)
                puntaje = float(usuario['puntaje_promedio']) if usuario['puntaje_promedio'] else 0
                if puntaje >= 2.5:
                    categoria = 'EXCELENTE'
                elif puntaje >= 2.0:
                    categoria = 'BUENO'
                elif puntaje >= 1.0:
                    categoria = 'REGULAR'
                else:
                    categoria = 'DEFICIENTE'
                
                usuario_clustering = {
                    'usuario_id': int(usuario['usuario_id']),
                    'nombre': usuario['nombre_completo'] or f"Usuario {usuario['usuario_id']}",
                    'prediccion': prediccion['prediccion'],
                    'asistira': prediccion['asistira'],
                    'categoria': categoria,
                    'puntaje': round(puntaje, 2),
                    'probabilidad': float(prediccion['probabilidad']),
                    'antiguedad': usuario['antiguedad'],
                    'puesto_sindicato': bool(usuario['puesto_sindicato']),
                    'genero': usuario['genero'],
                    'en_riesgo': bool(prediccion['en_riesgo']),
                    'tasa_asistencia': round(float(usuario['tasa_asistencia']) if usuario['tasa_asistencia'] else 0, 3)
                }
                
                usuarios_data.append(usuario_clustering)
                
                # Contadores
                if prediccion['asistira']:
                    asistiran += 1
                else:
                    no_asistiran += 1
        
        # Estadísticas por categoría (tu sistema)
        distribucion_categorias = {
            'EXCELENTE': len([u for u in usuarios_data if u['categoria'] == 'EXCELENTE']),
            'BUENO': len([u for u in usuarios_data if u['categoria'] == 'BUENO']),
            'REGULAR': len([u for u in usuarios_data if u['categoria'] == 'REGULAR']),
            'DEFICIENTE': len([u for u in usuarios_data if u['categoria'] == 'DEFICIENTE'])
        }
        
        # Estadísticas simplificadas por predicción
        clusters_stats = {
            'Asistirán': {
                'nombre': 'Asistirán',
                'total': asistiran,
                'excelente': len([u for u in usuarios_data if u['asistira'] and u['categoria'] == 'EXCELENTE']),
                'bueno': len([u for u in usuarios_data if u['asistira'] and u['categoria'] == 'BUENO']),
                'regular': len([u for u in usuarios_data if u['asistira'] and u['categoria'] == 'REGULAR']),
                'deficiente': len([u for u in usuarios_data if u['asistira'] and u['categoria'] == 'DEFICIENTE']),
                'puntaje_promedio': round(np.mean([u['puntaje'] for u in usuarios_data if u['asistira']]), 2) if asistiran > 0 else 0,
                'riesgo': 0  # Por definición, los que asistirán no están en riesgo
            },
            'No Asistirán': {
                'nombre': 'No Asistirán',
                'total': no_asistiran,
                'excelente': len([u for u in usuarios_data if not u['asistira'] and u['categoria'] == 'EXCELENTE']),
                'bueno': len([u for u in usuarios_data if not u['asistira'] and u['categoria'] == 'BUENO']),
                'regular': len([u for u in usuarios_data if not u['asistira'] and u['categoria'] == 'REGULAR']),
                'deficiente': len([u for u in usuarios_data if not u['asistira'] and u['categoria'] == 'DEFICIENTE']),
                'puntaje_promedio': round(np.mean([u['puntaje'] for u in usuarios_data if not u['asistira']]), 2) if no_asistiran > 0 else 0,
                'riesgo': no_asistiran  # Todos los que no asistirán están en riesgo
            }
        }
        
        # Distribución simplificada por predicción
        distribucion_clusters = {
            'Asistirán': asistiran,
            'No Asistirán': no_asistiran
        }
        
        total_usuarios = len(usuarios_data)
        
        return jsonify({
            'success': True,
            'total_usuarios': total_usuarios,
            'usuarios_riesgo': no_asistiran,
            'asistiran': asistiran,
            'no_asistiran': no_asistiran,
            'usuarios': usuarios_data,
            'clusters_stats': clusters_stats,
            'distribucion_categorias': distribucion_categorias,
            'distribucion_clusters': distribucion_clusters,
            'resumen': {
                'total_analizado': total_usuarios,
                'prediccion_asistiran': asistiran,
                'prediccion_no_asistiran': no_asistiran,
                'porcentaje_asistiran': round((asistiran / total_usuarios) * 100, 1) if total_usuarios > 0 else 0,
                'modelo_activo': 'ML Predicción Activa',
                'sistema_clasificacion': 'DEFICIENTE/REGULAR/BUENO/EXCELENTE'
            }
        })
        
    except Exception as e:
        print(f"❌ Error en clustering: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'modelo_ml': modelo_ml is not None,
        'bd_config': bool(os.getenv('DB_HOST'))
    })

print("🚀 Iniciando Sistema de Predicción SUTUTEH...")
cargar_modelo()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)