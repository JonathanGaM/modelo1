# app.py - Backend Python SIMPLE para cargar datos SUTUTEH
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import mysql.connector
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Configuración de la base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'dbsututeh'),
    'charset': 'utf8mb4'
}

def conectar_bd():
    """Crear conexión a la base de datos"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error conectando a BD: {e}")
        return None

# Leer el archivo HTML desde templates
@app.route('/')
def index():
    """Servir la página principal HTML desde templates"""
    return render_template('formulario.html')

@app.route('/api/status')
def api_status():
    """Endpoint de estado del API"""
    return jsonify({
        'message': 'Backend Python SIMPLE SUTUTEH',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dataset-completo')
def get_dataset_completo():
    """Obtener dataset completo de la vista - SIN MODELO"""
    try:
        conn = conectar_bd()
        if not conn:
            return jsonify({'success': False, 'error': 'Error de conexión a BD'}), 500
        
        limit = request.args.get('limit', 1000, type=int)
        
        query = """
        SELECT 
            asistencia_id,
            reunion_id,
            usuario_id,
            estado_asistencia,
            puntaje,
            DATE(antiguedad) as antiguedad,
            rol_sindicato_id,
            es_admin,
            puesto_id,
            nivel_id,
            tipo_reunion,
            genero,
            DATE(fecha_reunion) as fecha_reunion,
            TIME(hora_reunion) as hora_reunion
        FROM vw_dataset_asistencia 
        ORDER BY fecha_reunion DESC
        LIMIT %s
        """
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (limit,))
        dataset = cursor.fetchall()
        
        # Convertir fechas a string para JSON
        for row in dataset:
            if row['antiguedad']:
                row['antiguedad'] = str(row['antiguedad'])
            if row['fecha_reunion']:
                row['fecha_reunion'] = str(row['fecha_reunion'])
            if row['hora_reunion']:
                row['hora_reunion'] = str(row['hora_reunion'])
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'total_registros': len(dataset),
            'dataset': dataset,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en dataset-completo: {e}")
        return jsonify({
            'success': False,
            'error': f'Error al obtener dataset: {str(e)}'
        }), 500

@app.route('/api/usuarios-basico')
def get_usuarios_basico():
    """Obtener usuarios con estadísticas básicas - SIN MODELO"""
    try:
        conn = conectar_bd()
        if not conn:
            return jsonify({'success': False, 'error': 'Error de conexión a BD'}), 500
        
        query = """
        SELECT 
            vw.usuario_id,
            COALESCE(pu.nombre, 'Sin nombre') as nombre,
            COALESCE(pu.apellido_paterno, '') as apellido_paterno,
            COALESCE(pu.apellido_materno, '') as apellido_materno,
            COALESCE(rs.nombre, 'Agremiado') as rol_nombre,
            COUNT(*) as total_reuniones,
            AVG(vw.puntaje) as puntaje_promedio,
            SUM(CASE WHEN vw.estado_asistencia = 'asistencia_completa' THEN 1 ELSE 0 END) as asistencias_completas,
            SUM(CASE WHEN vw.estado_asistencia = 'retardo' THEN 1 ELSE 0 END) as retardos,
            SUM(CASE WHEN vw.estado_asistencia = 'falta' THEN 1 ELSE 0 END) as faltas,
            MAX(vw.es_admin) as es_admin,
            MAX(vw.genero) as genero,
            DATE(MAX(vw.antiguedad)) as antiguedad
        FROM vw_dataset_asistencia vw
        LEFT JOIN perfil_usuarios pu ON vw.usuario_id = pu.id
        LEFT JOIN roles_sindicato rs ON pu.rol_sindicato_id = rs.id
        GROUP BY vw.usuario_id, pu.nombre, pu.apellido_paterno, pu.apellido_materno, rs.nombre
        ORDER BY COUNT(*) DESC
        """
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        usuarios = cursor.fetchall()
        
        # Procesar datos y calcular métricas simples
        for usuario in usuarios:
            # Convertir fecha a string
            if usuario['antiguedad']:
                usuario['antiguedad'] = str(usuario['antiguedad'])
            
            # Crear nombre completo concatenando apellidos
            apellidos = []
            if usuario['apellido_paterno']:
                apellidos.append(usuario['apellido_paterno'])
            if usuario['apellido_materno']:
                apellidos.append(usuario['apellido_materno'])
            
            usuario['apellido'] = ' '.join(apellidos) if apellidos else ''
            
            # Calcular tasa de asistencia
            total = usuario['total_reuniones']
            asistencias = usuario['asistencias_completas']
            usuario['tasa_asistencia'] = round(asistencias / total if total > 0 else 0, 3)
            
            # Calcular años de antigüedad (aproximado)
            if usuario['antiguedad']:
                try:
                    from datetime import datetime
                    fecha_ant = datetime.strptime(usuario['antiguedad'], '%Y-%m-%d')
                    años = (datetime.now() - fecha_ant).days / 365.25
                    usuario['antiguedad_años'] = round(años, 1)
                except:
                    usuario['antiguedad_años'] = 0
            else:
                usuario['antiguedad_años'] = 0
            
            # Redondear puntaje
            if usuario['puntaje_promedio']:
                usuario['puntaje_promedio'] = round(usuario['puntaje_promedio'], 2)
            else:
                usuario['puntaje_promedio'] = 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'total_usuarios': len(usuarios),
            'usuarios': usuarios,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en usuarios-basico: {e}")
        return jsonify({
            'success': False,
            'error': f'Error al obtener usuarios: {str(e)}'
        }), 500

@app.route('/api/estadisticas-generales')
def get_estadisticas_generales():
    """Obtener estadísticas generales - SIN MODELO"""
    try:
        conn = conectar_bd()
        if not conn:
            return jsonify({'success': False, 'error': 'Error de conexión a BD'}), 500
        
        # Estadísticas básicas
        query_stats = """
        SELECT 
            COUNT(DISTINCT usuario_id) as total_usuarios,
            COUNT(DISTINCT reunion_id) as total_reuniones,
            COUNT(*) as total_registros,
            AVG(puntaje) as puntaje_promedio_global,
            SUM(CASE WHEN estado_asistencia = 'asistencia_completa' THEN 1 ELSE 0 END) as total_asistencias,
            SUM(CASE WHEN estado_asistencia = 'retardo' THEN 1 ELSE 0 END) as total_retardos,
            SUM(CASE WHEN estado_asistencia = 'falta' THEN 1 ELSE 0 END) as total_faltas
        FROM vw_dataset_asistencia
        """
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query_stats)
        stats = cursor.fetchone()
        
        # Calcular tasa de asistencia general
        total_registros = stats['total_registros']
        total_asistencias = stats['total_asistencias']
        tasa_asistencia_general = round(total_asistencias / total_registros if total_registros > 0 else 0, 3)
        
        # Estadísticas por tipo de reunión
        query_tipos = """
        SELECT 
            tipo_reunion,
            COUNT(*) as cantidad,
            AVG(puntaje) as puntaje_promedio
        FROM vw_dataset_asistencia
        GROUP BY tipo_reunion
        """
        
        cursor.execute(query_tipos)
        tipos_reunion = cursor.fetchall()
        
        # Estadísticas por género
        query_genero = """
        SELECT 
            genero,
            COUNT(DISTINCT usuario_id) as cantidad_usuarios
        FROM vw_dataset_asistencia
        GROUP BY genero
        """
        
        cursor.execute(query_genero)
        por_genero = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'estadisticas_generales': {
                'total_usuarios': stats['total_usuarios'],
                'total_reuniones': stats['total_reuniones'],
                'total_registros': stats['total_registros'],
                'puntaje_promedio_global': round(stats['puntaje_promedio_global'], 2) if stats['puntaje_promedio_global'] else 0,
                'tasa_asistencia_general': tasa_asistencia_general,
                'total_asistencias': stats['total_asistencias'],
                'total_retardos': stats['total_retardos'],
                'total_faltas': stats['total_faltas']
            },
            'por_tipo_reunion': tipos_reunion,
            'por_genero': por_genero,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en estadisticas-generales: {e}")
        return jsonify({
            'success': False,
            'error': f'Error al obtener estadísticas: {str(e)}'
        }), 500

@app.route('/api/test-conexion')
def test_conexion():
    """Probar conexión a la base de datos"""
    try:
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'No se pudo conectar a la base de datos',
                'config_usada': {
                    'host': DB_CONFIG['host'],
                    'port': DB_CONFIG['port'],
                    'user': DB_CONFIG['user'],
                    'database': DB_CONFIG['database']
                }
            }), 500
        
        # Probar consulta simple
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM vw_dataset_asistencia")
        resultado = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Conexión exitosa',
            'total_registros_vista': resultado[0],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error de conexión: {str(e)}',
            'config_usada': {
                'host': DB_CONFIG['host'],
                'port': DB_CONFIG['port'],
                'user': DB_CONFIG['user'],
                'database': DB_CONFIG['database']
            }
        }), 500

if __name__ == '__main__':
    print("🚀 Iniciando Backend Python SIMPLE SUTUTEH...")
    print(f"📊 Configuración BD:")
    print(f"   Host: {DB_CONFIG['host']}")
    print(f"   Puerto: {DB_CONFIG['port']}")
    print(f"   Usuario: {DB_CONFIG['user']}")
    print(f"   Base de datos: {DB_CONFIG['database']}")
    
    # Probar conexión al iniciar
    conn_test = conectar_bd()
    if conn_test:
        print("✅ Conexión a base de datos exitosa")
        try:
            cursor = conn_test.cursor()
            cursor.execute("SELECT COUNT(*) FROM vw_dataset_asistencia")
            total = cursor.fetchone()[0]
            print(f"📊 Registros en vista: {total}")
            cursor.close()
        except Exception as e:
            print(f"⚠️ Error probando vista: {e}")
        conn_test.close()
    else:
        print("❌ Error de conexión a base de datos")
    
    print("🌐 Aplicación web disponible en http://localhost:5000")
    print("🔗 API endpoints en http://localhost:5000/api/")
    
    app.run(debug=True, host='0.0.0.0', port=5000)