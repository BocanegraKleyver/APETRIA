# Instalación de APETRIA (Windows)

Guía probada de forma independiente para el prototipo **APETRIA E100** en Windows con **Python 3.11.1** y PowerShell.

## 1. Obtener el código

```powershell
git clone https://github.com/BocanegraKleyver/APETRIA.git
cd APETRIA
```

El repositorio público contiene la aplicación y la documentación. No incluye los pesos entrenados ni los conjuntos de datos procesados.

## 2. Instalar las dependencias

```powershell
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py --version
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip check
```

La instalación con `requirements.txt` fue validada en un entorno virtual vacío el 23/11/2026. Usar el ejecutable de `.venv` directamente evita depender de la política de ejecución de PowerShell.

## 3. Incorporar los artefactos reales

Obtener el paquete de demo indicado en [`artefactos.md`](artefactos.md). Su URL de distribución pública está **pendiente de publicación**. Si ya se dispone del ZIP en Descargas, desde la raíz del repositorio:

```powershell
$zip = Join-Path $env:USERPROFILE 'Downloads\APETRIA_Demo_5_fuentes_20261123.zip'
if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) { throw 'No se encuentra el ZIP demo' }
$sha = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
$expected = 'C0FF73F5960509596AD7985C8EA43F30322BC9BE38BC7D41F32BE04921B37148'
if ($sha -ne $expected) { throw 'SHA-256 del ZIP diferente del documentado' }
Expand-Archive -LiteralPath $zip -DestinationPath . -ErrorAction Stop
```

El ZIP crea la estructura `data/`, `models/` y `results/` dentro de la raíz de APETRIA. No cambiar los nombres ni mover los archivos. Verificar que no se esté extrayendo contenido sobre un directorio con resultados previos que deban preservarse.

## 4. Crear un administrador

```powershell
& $py -m backend.app.cli.create_admin
```

Elegir usuario y contraseña de **al menos 12 caracteres**. No hay usuario ni contraseña predeterminados. La base SQLite de cuentas se crea en `instance/auth.db`. Es local, privada y está excluida de Git.

## 5. Iniciar la aplicación

```powershell
& $py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Abrir **http://127.0.0.1:8000/**, ingresar con el administrador y elegir una fuente. El backend y el frontend se sirven **desde el mismo puerto**; no iniciar `http.server` ni usar el puerto 5173.

Para detener el servidor, presionar `Ctrl+C` en su terminal.

## 6. Comprobar el estado

- `http://127.0.0.1:8000/api/health`: salud del proceso (público).
- `http://127.0.0.1:8000/api/sources`: fuentes y disponibilidad (requiere sesión).
- `http://127.0.0.1:8000/docs`: documentación API de FastAPI.

Comprobar las fuentes SMD (`machine-1-1`), NASA (`A-1`), NAB (`ec2_cpu_utilization_24ae8d.csv`), HDFS y BGL. Las fuentes deben estar habilitadas desde la vista **Configuración**.

## Problemas comunes

**Puerto 8000 ocupado (`WinError 10048`)**: otro proceso está escuchando. En PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
```

Identificar el proceso antes de detenerlo; preferir `Ctrl+C` en la terminal del Uvicorn anterior.

**Contraseña incorrecta (`HTTP 401`)**: el rechazo es esperado. Reintentar con credenciales válidas, sin compartirlas en registros o capturas.

**Falta un dataset/modelo**: comprobar la presencia y la ruta exacta de los 14 archivos descritos en [`artefactos.md`](artefactos.md). No reemplazar archivos faltantes con datos sintéticos o resultados inventados.

**Visualizaciones sin gráficos**: verificar conectividad con los CDN utilizados por Chart.js y Lucide.

## Seguridad

Solo para ejecución local con `--host 127.0.0.1`. No publicar el servidor en Internet ni usarlo para sistemas de producción sin adecuaciones de seguridad, TLS, operación y gestión de usuarios.
