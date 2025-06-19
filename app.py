from flask import Flask

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>¡Bienvenido a miproyecto_py!</h1>
    <p>Tu aplicación Flask está funcionando correctamente.</p>
    <p>Entorno virtual configurado exitosamente.</p>
    """

@app.route('/info')
def info():
    return {
        "proyecto": "miproyecto_py",
        "framework": "Flask",
        "entorno": "Entorno virtual Python",
        "estado": "Funcionando"
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)