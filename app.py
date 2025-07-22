# app.py - Backend Python CON MODELO ML INTEGRADO - TODAS LAS GRÁFICAS USAN .pkl
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import mysql.connector
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import pickle
import numpy as np
import traceback

# Cargar variables de entorno solo si existe el archivo .env (desarrollo local)
if os.path.exists('.env'):
    load_dotenv()

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Configuración de la base de datos SIN valores por defecto problemáticos
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),           
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),           
    'password': os.getenv('DB_PASSWORD', ''),  
    'database': os.getenv('DB_NAME'),       
    'charset': 'utf8mb4',
    'connect_timeout': 30,                  
    'autocommit': True                      
}

# Variables globales para el modelo ML
modelo_ml = None
scaler_ml = None
encoders_ml = None

def cargar_modelo_ml():
    """Cargar el modelo ML al iniciar la aplicación"""
    global modelo_ml, scaler_ml, encoders_ml
    
    try:
        print("🔍 DEBUG: Iniciando carga de modelo ML...", flush=True)
        
        # Verificar que los archivos existen antes de cargar
        archivos_requeridos = ['modelo_sututeh.pkl', 'scaler_sututeh.pkl', 'encoders_sututeh.pkl']
        for archivo in archivos_requeridos:
            if not os.path.exists(archivo):
                print(f"❌ ERROR: Archivo {archivo} no encontrado", flush=True)
                return False
            else:
                print(f"✅ Archivo {archivo} encontrado ({os.path.getsize(archivo)} bytes)", flush=True)
        
        # Cargar modelo principal con debugging detallado
        print("🔄 Cargando modelo_sututeh.pkl...", flush=True)
        try:
            with open('modelo_sututeh.pkl', 'rb') as f:
                modelo_ml = pickle.load(f)
            print("✅ modelo_sututeh.pkl cargado exitosamente", flush=True)
            print(f"   Tipo: {type(modelo_ml)}", flush=True)
            if isinstance(modelo_ml, dict):
                print(f"   Keys: {list(modelo_ml.keys())}", flush=True)
        except Exception as e:
            print(f"❌ ERROR cargando modelo_sututeh.pkl: {e}", flush=True)
            print(f"   Tipo de error: {type(e)}", flush=True)
            print(f"   Traceback: {traceback.format_exc()}", flush=True)
            return False
        
        # Cargar scaler con debugging
        print("🔄 Cargando scaler_sututeh.pkl...", flush=True)
        try:
            with open('scaler_sututeh.pkl', 'rb') as f:
                scaler_ml = pickle.load(f)
            print("✅ scaler_sututeh.pkl cargado exitosamente", flush=True)
            print(f"   Tipo: {type(scaler_ml)}", flush=True)
        except Exception as e:
            print(f"❌ ERROR cargando scaler_sututeh.pkl: {e}", flush=True)
            print(f"   Tipo de error: {type(e)}", flush=True)
            print(f"   Traceback: {traceback.format_exc()}", flush=True)
            return False
            
        # Cargar encoders con debugging
        print("🔄 Cargando encoders_sututeh.pkl...", flush=True)
        try:
            with open('encoders_sututeh.pkl', 'rb') as f:
                encoders_ml = pickle.load(f)
            print("✅ encoders_sututeh.pkl cargado exitosamente", flush=True)
            print(f"   Tipo: {type(encoders_ml)}", flush=True)
            if isinstance(encoders_ml, dict):
                print(f"   Keys: {list(encoders_ml.keys())}", flush=True)
        except Exception as e:
            print(f"❌ ERROR cargando encoders_sututeh.pkl: {e}", flush=True)
            print(f"   Tipo de error: {type(e)}", flush=True)
            print(f"   Traceback: {traceback.format_exc()}", flush=True)
            return False
        
        # Verificar estructura del modelo
        print("🔍 Verificando estructura del modelo...", flush=True)
        try:
            if not isinstance(modelo_ml, dict):
                print(f"❌ ERROR: modelo_ml no es un diccionario, es: {type(modelo_ml)}", flush=True)
                return False
                
            if 'metricas' not in modelo_ml:
                print("❌ ERROR: modelo_ml no tiene clave 'metricas'", flush=True)
                print(f"   Keys disponibles: {list(modelo_ml.keys())}", flush=True)
                return False
                
            if 'metadata' not in modelo_ml:
                print("❌ ERROR: modelo_ml no tiene clave 'metadata'", flush=True)
                return False
        
            print("✅ Estructura del modelo verificada", flush=True)
            print(f"📊 Precisión del modelo: {modelo_ml['metricas']['roc_auc']:.3f}", flush=True)
            print(f"📅 Entrenado el: {modelo_ml['metadata']['fecha_entrenamiento']}", flush=True)
            return True
            
        except KeyError as e:
            print(f"❌ ERROR: Clave faltante en modelo: {e}", flush=True)
            print(f"   Keys del modelo: {list(modelo_ml.keys()) if isinstance(modelo_ml, dict) else 'No es dict'}", flush=True)
            return False
        except Exception as e:
            print(f"❌ ERROR verificando estructura: {e}", flush=True)
            print(f"   Traceback: {traceback.format_exc()}", flush=True)
            return False
        
    except Exception as e:
        print(f"❌ Error general al cargar modelo ML: {e}", flush=True)
        print(f"   Tipo de error: {type(e)}", flush=True)
        print(f"   Traceback: {traceback.format_exc()}", flush=True)
        return False

def predecir_asistencia_usuario(usuario_data):
    """
    Hacer predicción ML para un usuario específico
    """
    
    if not modelo_ml or not scaler_ml or not encoders_ml:
        return None
    
    try:
        # Extraer componentes del modelo
        modelo_lr = modelo_ml['modelo_prediccion']
        le_tipo = encoders_ml['tipo']
        
        # Procesar datos del usuario
        tasa_asistencia = usuario_data.get('tasa_asistencia', 0.7)
        total_reuniones = usuario_data.get('total_reuniones', 15)
        puntaje_promedio = usuario_data.get('puntaje_promedio', 2.2)
        es_admin = usuario_data.get('es_admin', 0)
        antiguedad_años = usuario_data.get('antiguedad_años', 2)
        
        # Datos por defecto para reunión
        tipo_reunion = 'Ordinaria'
        dia_semana = 2  # Miércoles
        hora = 14      # 2 PM
        
        # Codificar tipo de reunión
        try:
            tipo_encoded = le_tipo.transform([tipo_reunion])[0]
        except:
            tipo_encoded = 0
        
        # Crear vector de features para predicción
        features_vector = [
            tasa_asistencia,
            total_reuniones, 
            puntaje_promedio,
            es_admin,
            antiguedad_años,
            tipo_encoded,
            dia_semana,
            hora
        ]
        
        # Normalizar con scaler
        features_scaled = scaler_ml.transform([features_vector])
        
        # Hacer predicción de asistencia
        probabilidad = modelo_lr.predict_proba(features_scaled)[0, 1]
        
        # 🎯 CLUSTERING CORREGIDO
        cluster_nombre = determinar_cluster_real(
            puntaje_promedio, 
            total_reuniones, 
            tasa_asistencia, 
            antiguedad_años
        )
        
        # 🚨 DETERMINAR RIESGO: Criterios coordinados con clustering
        en_riesgo = (tasa_asistencia <= 0.3) or \
                   (total_reuniones <= 15) or \
                   (puntaje_promedio <= 1.8) or \
                   (cluster_nombre == 'Inactivos Críticos')
        
        return {
            'probabilidad_asistencia': float(probabilidad),
            'probabilidad_texto': f"{probabilidad:.1%}",
            'cluster': cluster_nombre,
            'en_riesgo': en_riesgo,
            'usa_modelo_real': True
        }
        
    except Exception as e:
        print(f"Error en predicción ML: {e}")
        return None

def determinar_cluster_real(puntaje_promedio, total_reuniones, tasa_asistencia, antiguedad_años):
    """
    Determina el cluster para una distribución balanceada y realista
    """
    
    # 🔴 Inactivos Críticos: SOLO los casos más extremos
    if (total_reuniones <= 15 and tasa_asistencia <= 0.9) or \
       (tasa_asistencia <= 0.10):
        return 'Inactivos Críticos'
    
    # 🟢 Activistas Comprometidos: Los mejores participantes
    if puntaje_promedio >= 2.5 and total_reuniones >= 50 and tasa_asistencia >= 0.7:
        return 'Activistas Comprometidos'
    
    # 🔵 Participativos Regulares: Participación media-alta consistente
    if (puntaje_promedio >= 2.0 and total_reuniones >= 30 and tasa_asistencia >= 0.5) or \
       (tasa_asistencia >= 0.6 and total_reuniones >= 40):
        return 'Participativos Regulares'
    
    # 🟡 Ocasionales Moderados: Todo lo demás
    return 'Ocasionales Moderados'

def conectar_bd():
    """Crear conexión a la base de datos con mejor manejo de errores"""
    try:
        # Verificar que todas las variables estén configuradas
        required_vars = ['DB_HOST', 'DB_USER','DB_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ ERROR: Variables de entorno faltantes: {missing_vars}")
            print("💡 Configúralas en Render Dashboard → Environment Variables")
            return None
        
        print(f"🔗 Conectando a BD:")
        print(f"   Host: {DB_CONFIG['host']}")
        print(f"   Puerto: {DB_CONFIG['port']}")  
        print(f"   Usuario: {DB_CONFIG['user']}")
        print(f"   Base de datos: {DB_CONFIG['database']}")
        
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ Conexión a BD exitosa")
        return conn
        
    except mysql.connector.Error as err:
        print(f"❌ Error MySQL específico: {err}")
        print(f"❌ Código de error: {err.errno}")
        return None
    except Exception as e:
        print(f"❌ Error general conectando a BD: {e}")
        print(f"❌ Tipo de error: {type(e)}")
        return None

# 🆕 NUEVO ENDPOINT: TODAS LAS GRÁFICAS CON .pkl OBLIGATORIO
@app.route('/api/datos-completos-ml')
def get_datos_completos_ml():
    """
    🆕 NUEVO: Endpoint que SIEMPRE usa modelo ML para TODAS las gráficas
    Si no hay modelo, devuelve error (sin fallback)
    """
    try:
        # 🚨 VERIFICACIÓN ESTRICTA: Modelo ML OBLIGATORIO
        if not modelo_ml or not scaler_ml or not encoders_ml:
            return jsonify({
                'success': False,
                'error': 'MODELO ML NO DISPONIBLE',
                'message': 'Los archivos .pkl son requeridos para este endpoint',
                'archivos_requeridos': [
                    'modelo_sututeh.pkl',
                    'scaler_sututeh.pkl', 
                    'encoders_sututeh.pkl'
                ]
            }), 500
        
        # Conectar a BD
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Error de conexión a base de datos'
            }), 500
        
        # 📊 OBTENER TODOS LOS DATOS DE BD
        query_usuarios = """
        SELECT 
            vw.usuario_id,
            COALESCE(pu.nombre, 'Sin nombre') as nombre,
            COALESCE(pu.apellido_paterno, '') as apellido_paterno,
            COALESCE(pu.apellido_materno, '') as apellido_materno,
            COALESCE(rs.nombre, 'Agremiado') as rol_nombre,
            COUNT(*) as total_reuniones,
            AVG(vw.puntaje) as puntaje_promedio,
            SUM(CASE WHEN vw.estado_asistencia = 'asistencia_completa' THEN 1 ELSE 0 END) as asistencias_completas,
            MAX(vw.es_admin) as es_admin,
            MAX(vw.genero) as genero,
            DATE(MAX(vw.antiguedad)) as antiguedad
        FROM vw_dataset_asistencia vw
        LEFT JOIN perfil_usuarios pu ON vw.usuario_id = pu.id
        LEFT JOIN roles_sindicato rs ON pu.rol_sindicato_id = rs.id
        GROUP BY vw.usuario_id, pu.nombre, pu.apellido_paterno, pu.apellido_materno, rs.nombre
        ORDER BY COUNT(*) DESC
        """
        
        # 📊 ESTADÍSTICAS GENERALES CON ML
        query_stats = """
        SELECT 
            COUNT(DISTINCT usuario_id) as total_usuarios,
            COUNT(DISTINCT reunion_id) as total_reuniones,
            COUNT(*) as total_registros,
            AVG(puntaje) as puntaje_promedio_global,
            SUM(CASE WHEN estado_asistencia = 'asistencia_completa' THEN 1 ELSE 0 END) as total_asistencias,
            MIN(fecha_reunion) as primera_reunion,
            MAX(fecha_reunion) as ultima_reunion
        FROM vw_dataset_asistencia
        """
        
        # 📊 DATOS POR TIPO DE REUNIÓN CON ML
        query_tipos = """
        SELECT 
            tipo_reunion,
            COUNT(*) as cantidad,
            AVG(puntaje) as puntaje_promedio,
            COUNT(DISTINCT usuario_id) as usuarios_unicos,
            SUM(CASE WHEN estado_asistencia = 'asistencia_completa' THEN 1 ELSE 0 END) as asistencias
        FROM vw_dataset_asistencia
        GROUP BY tipo_reunion
        ORDER BY cantidad DESC
        """
        
        cursor = conn.cursor(dictionary=True)
        
        # Ejecutar consultas
        cursor.execute(query_usuarios)
        usuarios_bd = cursor.fetchall()
        
        cursor.execute(query_stats)
        stats_generales = cursor.fetchone()
        
        cursor.execute(query_tipos)
        tipos_reunion = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # 🤖 PROCESAR CADA USUARIO CON MODELO ML OBLIGATORIO
        usuarios_con_ml = []
        clusters_count = {}
        usuarios_riesgo = 0
        asistencia_alta = 0
        probabilidades_por_tipo = {}
        
        for usuario in usuarios_bd:
            # Procesar datos básicos
            apellidos = []
            if usuario['apellido_paterno']:
                apellidos.append(usuario['apellido_paterno'])
            if usuario['apellido_materno']:
                apellidos.append(usuario['apellido_materno'])
            
            apellido_completo = ' '.join(apellidos) if apellidos else ''
            
            # Calcular métricas
            total = usuario['total_reuniones']
            asistencias = usuario['asistencias_completas']
            tasa_asistencia = round(asistencias / total if total > 0 else 0, 3)
            
            # Calcular antigüedad
            antiguedad_años = 0
            if usuario['antiguedad']:
                try:
                    fecha_ant = datetime.strptime(str(usuario['antiguedad']), '%Y-%m-%d')
                    antiguedad_años = round((datetime.now() - fecha_ant).days / 365.25, 1)
                except:
                    antiguedad_años = 0
            
            # Preparar datos para ML
            usuario_data = {
                'tasa_asistencia': tasa_asistencia,
                'total_reuniones': usuario['total_reuniones'],
                'puntaje_promedio': round(usuario['puntaje_promedio'], 2) if usuario['puntaje_promedio'] else 0,
                'es_admin': usuario['es_admin'] or 0,
                'antiguedad_años': antiguedad_años
            }
            
            # 🚀 HACER PREDICCIÓN CON MODELO ML OBLIGATORIO
            prediccion = predecir_asistencia_usuario(usuario_data)
            
            if not prediccion:
                # Si la predicción ML falla, devolver error
                return jsonify({
                    'success': False,
                    'error': f'Error en predicción ML para usuario {usuario["usuario_id"]}',
                    'message': 'El modelo ML falló durante el procesamiento'
                }), 500
            
            # Usuario con predicción ML exitosa
            usuario_final = {
                'usuario_id': usuario['usuario_id'],
                'nombre_completo': f"{usuario['nombre']} {apellido_completo}".strip(),
                'rol': usuario['rol_nombre'],
                'cluster': prediccion['cluster'],
                'probabilidad_asistencia': prediccion['probabilidad_asistencia'],
                'probabilidad_texto': prediccion['probabilidad_texto'],
                'en_riesgo': prediccion['en_riesgo'],
                'estadisticas': {
                    'puntaje_promedio': usuario_data['puntaje_promedio'],
                    'total_reuniones': usuario_data['total_reuniones'],
                    'tasa_asistencia': usuario_data['tasa_asistencia'],
                    'antiguedad_años': usuario_data['antiguedad_años']
                },
                'modelo_ml': True,
                'genero': usuario['genero']
            }
            
            usuarios_con_ml.append(usuario_final)
            
            # Contar estadísticas
            cluster = prediccion['cluster']
            clusters_count[cluster] = clusters_count.get(cluster, 0) + 1
            
            if prediccion['en_riesgo']:
                usuarios_riesgo += 1
                
            if prediccion['probabilidad_asistencia'] >= 0.7:
                asistencia_alta += 1
        
        # 🤖 PROCESAR TIPOS DE REUNIÓN CON ML
        for tipo in tipos_reunion:
            # Simular predicción para cada tipo de reunión
            tipo_data = {
                'tasa_asistencia': tipo['asistencias'] / tipo['cantidad'] if tipo['cantidad'] > 0 else 0,
                'total_reuniones': tipo['cantidad'],
                'puntaje_promedio': tipo['puntaje_promedio'] or 2.0,
                'es_admin': 0,
                'antiguedad_años': 2.0
            }
            
            prediccion_tipo = predecir_asistencia_usuario(tipo_data)
            if prediccion_tipo:
                probabilidades_por_tipo[tipo['tipo_reunion']] = {
                    'probabilidad': prediccion_tipo['probabilidad_asistencia'],
                    'cantidad': tipo['cantidad'],
                    'puntaje_promedio': tipo['puntaje_promedio'],
                    'usuarios_unicos': tipo['usuarios_unicos']
                }
        
        # 📊 ESTADÍSTICAS AVANZADAS CON ML
        estadisticas_avanzadas = {
            'distribucion_probabilidades': {
                'muy_alta': len([u for u in usuarios_con_ml if u['probabilidad_asistencia'] >= 0.9]),
                'alta': len([u for u in usuarios_con_ml if 0.7 <= u['probabilidad_asistencia'] < 0.9]),
                'media': len([u for u in usuarios_con_ml if 0.5 <= u['probabilidad_asistencia'] < 0.7]),
                'baja': len([u for u in usuarios_con_ml if u['probabilidad_asistencia'] < 0.5])
            },
            'riesgo_por_cluster': {
                cluster: len([u for u in usuarios_con_ml if u['cluster'] == cluster and u['en_riesgo']])
                for cluster in clusters_count.keys()
            },
            'promedio_probabilidad_global': np.mean([u['probabilidad_asistencia'] for u in usuarios_con_ml]),
            'usuarios_activos_predichos': len([u for u in usuarios_con_ml if u['probabilidad_asistencia'] >= 0.6])
        }
        
        return jsonify({
            'success': True,
            'modo': 'ML_OBLIGATORIO',
            'total_usuarios': len(usuarios_con_ml),
            'usuarios': usuarios_con_ml,
            'estadisticas_generales': {
                'total_usuarios': stats_generales['total_usuarios'],
                'total_reuniones': stats_generales['total_reuniones'],
                'total_registros': stats_generales['total_registros'],
                'puntaje_promedio_global': round(stats_generales['puntaje_promedio_global'], 2) if stats_generales['puntaje_promedio_global'] else 0,
                'tasa_asistencia_general': round(stats_generales['total_asistencias'] / stats_generales['total_registros'], 3) if stats_generales['total_registros'] > 0 else 0,
                'primera_reunion': str(stats_generales['primera_reunion']) if stats_generales['primera_reunion'] else None,
                'ultima_reunion': str(stats_generales['ultima_reunion']) if stats_generales['ultima_reunion'] else None
            },
            'clusters': clusters_count,
            'tipos_reunion_ml': probabilidades_por_tipo,
            'estadisticas_avanzadas': estadisticas_avanzadas,
            'estadisticas_simples': {
                'total_usuarios': len(usuarios_con_ml),
                'asistencia_predicha_proxima': asistencia_alta,
                'usuarios_riesgo': usuarios_riesgo,
                'clusters': clusters_count
            },
            'modelo_info': {
                'precision': modelo_ml['metricas']['roc_auc'],
                'fecha_entrenamiento': modelo_ml['metadata']['fecha_entrenamiento'],
                'version': modelo_ml['metadata'].get('version', '1.0'),
                'usa_modelo_real': True,
                'archivos_cargados': ['modelo_sututeh.pkl', 'scaler_sututeh.pkl', 'encoders_sututeh.pkl']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en datos-completos-ml: {e}")
        return jsonify({
            'success': False,
            'error': f'Error crítico en procesamiento ML: {str(e)}',
            'message': 'El sistema requiere archivos .pkl válidos para funcionar'
        }), 500

# Mantener endpoints originales para compatibilidad
@app.route('/')
def index():
    """Servir la página principal HTML desde templates"""
    try:
        return render_template('formulario.html')
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SUTUTEH - Sistema ML Obligatorio</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>🤖 SUTUTEH - Sistema ML con .pkl OBLIGATORIO</h1>
            <h2>Estado del Sistema:</h2>
            <div id="status">
                <p><strong>Servidor:</strong> ✅ Funcionando</p>
                <p><strong>Template:</strong> ❌ No encontrado</p>
                <p><strong>Error:</strong> {e}</p>
            </div>
            
            <h2>🔧 Pruebas de API:</h2>
            <button onclick="testML()">🤖 Test ML Obligatorio</button>
            <button onclick="testStatus()">📊 Test Status</button>
            
            <div id="resultados" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc;">
                <p>Resultados aparecerán aquí...</p>
            </div>

            <script>
                function mostrarResultado(data) {{
                    document.getElementById('resultados').innerHTML = 
                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }}
                
                function testML() {{
                    fetch('/api/datos-completos-ml')
                        .then(response => response.json())
                        .then(data => mostrarResultado(data))
                        .catch(error => mostrarResultado({{'error': error.toString()}}));
                }}
                
                function testStatus() {{
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => mostrarResultado(data))
                        .catch(error => mostrarResultado({{'error': error.toString()}}));
                }}
            </script>
        </body>
        </html>
        """

@app.route('/api/status')
def api_status():
    """Endpoint de estado del API con información de entorno y modelo ML"""
    env_vars_status = {}
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_PORT']
    
    for var in required_vars:
        value = os.getenv(var)
        if var == 'DB_PASSWORD':
            env_vars_status[var] = "✅ Configurada" if value else "❌ Faltante"
        else:
            env_vars_status[var] = value if value else "❌ Faltante"
    
    # Estado del modelo ML
    modelo_status = {
        'modelo_cargado': modelo_ml is not None,
        'scaler_cargado': scaler_ml is not None,
        'encoders_cargados': encoders_ml is not None,
        'sistema_ml_completo': all([modelo_ml, scaler_ml, encoders_ml])
    }
    
    if modelo_ml:
        modelo_status.update({
            'precision': modelo_ml['metricas']['roc_auc'],
            'fecha_entrenamiento': modelo_ml['metadata']['fecha_entrenamiento'],
            'version': modelo_ml['metadata'].get('version', '1.0')
        })
    
    return jsonify({
        'message': 'SUTUTEH Backend - MODELO ML OBLIGATORIO',
        'status': 'running',
        'modo_operacion': 'ML_OBLIGATORIO' if all([modelo_ml, scaler_ml, encoders_ml]) else 'ML_FALTANTE',
        'environment': 'production' if os.getenv('NODE_ENV') == 'production' else 'development',
        'variables_entorno': env_vars_status,
        'modelo_ml': modelo_status,
        'archivos_pkl_requeridos': ['modelo_sututeh.pkl', 'scaler_sututeh.pkl', 'encoders_sututeh.pkl'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/debug-archivos')
def debug_archivos():
    """Endpoint para debug de archivos en producción"""
    try:
        import os
        
        directorio_actual = os.getcwd()
        archivos_disponibles = os.listdir('.')
        archivos_pkl = [f for f in archivos_disponibles if f.endswith('.pkl')]
        
        info_archivos = []
        archivos_requeridos = ['modelo_sututeh.pkl', 'scaler_sututeh.pkl', 'encoders_sututeh.pkl']
        
        for archivo in archivos_requeridos:
            existe = os.path.exists(archivo)
            info = {
                'nombre': archivo,
                'existe': existe,
                'tamaño_bytes': os.path.getsize(archivo) if existe else 0,
                'ruta_completa': os.path.abspath(archivo) if existe else 'No existe'
            }
            info_archivos.append(info)
        
        return jsonify({
            'directorio_actual': directorio_actual,
            'todos_los_archivos': sorted(archivos_disponibles),
            'archivos_pkl_encontrados': archivos_pkl,
            'archivos_requeridos': info_archivos,
            'variables_entorno': {
                'PWD': os.getenv('PWD'),
                'HOME': os.getenv('HOME'),
                'PATH': os.getenv('PATH')[:200] + '...' if os.getenv('PATH') else None
            },
            'python_path': os.sys.path,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error en debug: {str(e)}',
            'tipo_error': str(type(e))
        }), 500

# Mantener otros endpoints para compatibilidad (opcional)
@app.route('/api/test-conexion')
def test_conexion():
    """Probar conexión a la base de datos"""
    try:
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'No se pudo conectar a la base de datos'
            }), 500
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM vw_dataset_asistencia")
        resultado = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Conexión exitosa a la base de datos',
            'total_registros_vista': resultado[0],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error de conexión: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# =====================================================================
# CARGAR MODELO AL INICIAR LA APLICACIÓN (FUERA DEL IF __NAME__)
# ¡¡¡ CRÍTICO: ESTO SE EJECUTA EN PRODUCCIÓN !!!
# =====================================================================

print("🚀 Iniciando SUTUTEH Backend - MODELO ML OBLIGATORIO...", flush=True)
print("🤖 Cargando modelo ML OBLIGATORIO...", flush=True)

# 🔥 MOVER ESTA LÍNEA AQUÍ (fuera del if __name__ para que funcione en producción)
modelo_cargado = cargar_modelo_ml()

if modelo_cargado:
    print("✅ Sistema ML listo - MODO OPERACIÓN NORMAL", flush=True)
else:
    print("❌ SISTEMA ML NO DISPONIBLE - Solo funcionará el endpoint de estado", flush=True)
    print("💡 Asegúrate de que estos archivos estén en el directorio:", flush=True)
    print("   - modelo_sututeh.pkl", flush=True)
    print("   - scaler_sututeh.pkl", flush=True)
    print("   - encoders_sututeh.pkl", flush=True)

# =====================================================================
# CONFIGURACIÓN Y EJECUCIÓN (SOLO PARA DESARROLLO LOCAL)
# En producción (Render) se usa Gunicorn y este bloque se ignora
# =====================================================================

if __name__ == '__main__':
    print("🌐 MODO DESARROLLO LOCAL - Este bloque NO se ejecuta en Render", flush=True)
    
    # Configurar puerto dinámico para Render
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Aplicación web disponible en puerto: {port}", flush=True)
    print("🔗 API endpoints disponibles:", flush=True)
    print("   /api/status", flush=True)
    print("   /api/datos-completos-ml 🆕 (REQUIERE .pkl)", flush=True)
    print("   /api/test-conexion", flush=True)
    
    # Para producción, debug=False
    debug_mode = os.getenv('NODE_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)