# app.py - Backend Python CORREGIDO PARA RENDER
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import mysql.connector
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar variables de entorno solo si existe el archivo .env (desarrollo local)
if os.path.exists('.env'):
    load_dotenv()

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Configuración de la base de datos SIN valores por defecto problemáticos
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),           # Sin default 'localhost'
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),           # Sin default 'root'
    'password': os.getenv('DB_PASSWORD'),   # Sin default vacío
    'database': os.getenv('DB_NAME'),       # Sin default
    'charset': 'utf8mb4',
    'connect_timeout': 30,                  # Timeout mayor para conexiones remotas
    'autocommit': True                      # Para evitar problemas de transacciones
}

def conectar_bd():
    """Crear conexión a la base de datos con mejor manejo de errores"""
    try:
        # Verificar que todas las variables estén configuradas
        required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
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

# Leer el archivo HTML desde templates
@app.route('/')
def index():
    """Servir la página principal HTML desde templates"""
    try:
        return render_template('formulario.html')
    except Exception as e:
        # Si no encuentra el template, devolver HTML básico con diagnóstico
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SUTUTEH - Backend Diagnóstico</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>🚀 Backend Python SUTUTEH</h1>
            <h2>Estado del Sistema:</h2>
            <div id="status">
                <p><strong>Servidor:</strong> ✅ Funcionando</p>
                <p><strong>Template:</strong> ❌ No encontrado (formulario.html)</p>
                <p><strong>Error:</strong> {e}</p>
            </div>
            
            <h2>🔧 Pruebas de API:</h2>
            <button onclick="testStatus()">Test Status</button>
            <button onclick="testConexion()">Test BD Conexión</button>
            <button onclick="testUsuarios()">Test Usuarios</button>
            
            <div id="resultados" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc;">
                <p>Resultados aparecerán aquí...</p>
            </div>

            <script>
                function mostrarResultado(data) {{
                    document.getElementById('resultados').innerHTML = 
                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }}
                
                function testStatus() {{
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => mostrarResultado(data))
                        .catch(error => mostrarResultado({{'error': error.toString()}}));
                }}
                
                function testConexion() {{
                    fetch('/api/test-conexion')
                        .then(response => response.json())
                        .then(data => mostrarResultado(data))
                        .catch(error => mostrarResultado({{'error': error.toString()}}));
                }}
                
                function testUsuarios() {{
                    fetch('/api/usuarios-basico')
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
    """Endpoint de estado del API con información de entorno"""
    env_vars_status = {}
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_PORT']
    
    for var in required_vars:
        value = os.getenv(var)
        if var == 'DB_PASSWORD':
            env_vars_status[var] = "✅ Configurada" if value else "❌ Faltante"
        else:
            env_vars_status[var] = value if value else "❌ Faltante"
    
    return jsonify({
        'message': 'Backend Python SUTUTEH para Render',
        'status': 'running',
        'environment': 'production' if os.getenv('NODE_ENV') == 'production' else 'development',
        'variables_entorno': env_vars_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test-conexion')
def test_conexion():
    """Probar conexión a la base de datos con diagnóstico detallado"""
    try:
        # Mostrar configuración (sin contraseña)
        config_display = {
            'host': DB_CONFIG.get('host', 'NO_CONFIGURADO'),
            'port': DB_CONFIG.get('port', 'NO_CONFIGURADO'),
            'user': DB_CONFIG.get('user', 'NO_CONFIGURADO'),
            'database': DB_CONFIG.get('database', 'NO_CONFIGURADO'),
            'password_configured': bool(DB_CONFIG.get('password'))
        }
        
        conn = conectar_bd()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'No se pudo conectar a la base de datos',
                'configuracion': config_display,
                'sugerencia': 'Verifica las variables de entorno en Render Dashboard'
            }), 500
        
        # Probar consulta simple
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM vw_dataset_asistencia")
        resultado = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Conexión exitosa a la base de datos',
            'total_registros_vista': resultado[0],
            'configuracion': config_display,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error de conexión: {str(e)}',
            'configuracion': config_display,
            'timestamp': datetime.now().isoformat()
        }), 500

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

if __name__ == '__main__':
    print("🚀 Iniciando Backend Python SUTUTEH...")
    
    # Mostrar configuración de entorno
    print(f"📊 Variables de entorno:")
    env_vars = ['DB_HOST', 'DB_USER', 'DB_NAME', 'DB_PORT']
    for var in env_vars:
        value = os.getenv(var)
        print(f"   {var}: {value if value else '❌ NO CONFIGURADO'}")
    
    print(f"   DB_PASSWORD: {'✅ Configurado' if os.getenv('DB_PASSWORD') else '❌ NO CONFIGURADO'}")
    
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
        print("❌ Error de conexión a base de datos - Revisa las variables de entorno")
    
    # Configurar puerto dinámico para Render
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Aplicación web disponible en puerto: {port}")
    print("🔗 API endpoints disponibles:")
    print("   /api/status")
    print("   /api/test-conexion")
    print("   /api/usuarios-basico")
    print("   /api/dataset-completo")
    print("   /api/estadisticas-generales")
    
    # Para producción, debug=False
    debug_mode = os.getenv('NODE_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)