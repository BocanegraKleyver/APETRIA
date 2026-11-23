# APETRIA

**Arquitectura de Procesamiento de Eventos en Tiempo Real con Inteligencia Artificial para la Detección de Anomalías en Sistemas Distribuidos**

Prototipo académico del Proyecto Final de Ingeniería en Informática, Universidad Argentina de la Empresa (UADE), 2026.

[Repositorio público](https://github.com/BocanegraKleyver/APETRIA)

## Descripción

APETRIA permite reproducir telemetría **real** de sistemas distribuidos, analizar detecciones de anomalías y examinar sus resultados en una consola web. El backend usa Python y FastAPI; la interfaz utiliza HTML, CSS y JavaScript y recibe observaciones mediante WebSocket.

**No genera telemetría sintética.** Es un prototipo experimental para uso académico, no un producto de observabilidad preparado para producción.

### Cinco fuentes de datos

| Fuente | Caso de demostración | Procesamiento en la aplicación |
|---|---|---|
| SMD (Server Machine Dataset) | `machine-1-1` | Reproducción real con inferencia Isolation Forest y LSTM Autoencoder |
| NASA SMAP/MSL | Canal `A-1` | Reproducción real con inferencia Isolation Forest y LSTM Autoencoder |
| NAB | `ec2_cpu_utilization_24ae8d.csv` | Reproducción de predicciones históricas |
| HDFS | Sesiones de logs | Reproducción de predicciones históricas |
| BGL | Eventos de logs | Reproducción de predicciones históricas |

Las fuentes NAB, HDFS y BGL **no ejecutan una nueva inferencia por cada mensaje** durante la demostración. Las modalidades son diferentes y sus métricas no se deben interpretar como equivalentes.

La consola tiene siete vistas: **Resumen, Métricas, Anomalías, Eventos, Fuentes, Alertas y Configuración**. Permite seleccionar fuentes y políticas compatibles, iniciar reproducciones y analizar métricas del segmento reproducido.

## Requisitos

- Windows y PowerShell (entorno donde se verificó la instalación).
- Python **3.11.1** y Git.
- Conexión a Internet para instalar dependencias y cargar Chart.js/Lucide desde CDN.
- El **paquete de demostración de cinco fuentes** (datos, modelos entrenados y resultados históricos), que se distribuye **por separado** del código.

Las 13 dependencias directas se encuentran fijadas en [`requirements.txt`](requirements.txt). La instalación limpia fue comprobada con `pip check` e importación del backend el 23 de noviembre de 2026.

## Instalación rápida en Windows

```powershell
git clone https://github.com/BocanegraKleyver/APETRIA.git
cd APETRIA
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip check
```

**Antes de reproducir fuentes:** obtener el archivo `APETRIA_Demo_5_fuentes_20261123.zip` y extraerlo en la **raíz** del repositorio, preservando `data/`, `models/` y `results/`. El paquete no está incluido en Git; su enlace público de descarga se documentará cuando se autorice y publique. Consultar [`docs/artefactos.md`](docs/artefactos.md) para nombres, rutas y SHA-256.

### Crear administrador local

```powershell
& $py -m backend.app.cli.create_admin
```

Elegir nombre de usuario y contraseña de **al menos 12 caracteres**. No existen credenciales de administrador predeterminadas. La base local se crea en `instance/auth.db` y **no debe compartirse**.

### Ejecutar

```powershell
& $py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Abrir <http://127.0.0.1:8000/> e iniciar sesión. El frontend es servido **por FastAPI desde el mismo puerto**: no hay que iniciar un servidor adicional en el puerto 5173.

El endpoint público de salud es <http://127.0.0.1:8000/api/health>. Los endpoints de fuentes y configuración requieren autenticación.

Más detalles en [`docs/instalacion.md`](docs/instalacion.md) y [`docs/autenticacion.md`](docs/autenticacion.md).

## Organización

```text
APETRIA/
├── backend/                 API, autenticación, adaptadores e inferencia
├── frontend/                Interfaz web
├── docs/                    Guías de instalación y artefactos
├── requirements.txt        Dependencias Python fijadas
└── .gitignore               Exclusiones de archivos locales
```

Después de extraer el paquete demo aparecen, entre otras, las carpetas locales `data/`, `models/` y `results/`. Esas carpetas están excluidas del repositorio Git público. `instance/` y `.venv/` también son locales y están excluidas.

## Validación funcional

El **23 de noviembre de 2026** se realizó una instalación limpia en Windows con Python 3.11.1, creación de administrador, inicio de sesión y una ejecución independiente de la prueba **PF-13** sobre el paquete demo: SMD, NASA, NAB, HDFS y BGL transmitieron **10 de 10 observaciones cada uno** (50/50 en total), con finalización y transporte WebSocket correctos e identificadores únicos en las cinco fuentes.

La prueba PF-13 comprueba **transporte y recepción**, no sustituye los estudios de precisión, recall, F1-score, falsos positivos, latencia o rendimiento experimental.

## Documentación

- [Instalación y puesta en marcha](docs/instalacion.md)
- [Artefactos de demostración y verificación SHA-256](docs/artefactos.md)
- [Autenticación y datos privados](docs/autenticacion.md)
- [Validación y alcance de las pruebas](docs/validacion.md)

## Alcance, derechos y uso

Proyecto de investigación académica, con datos reales y modelos previamente entrenados. No exponer este servidor de demostración directamente en Internet. Los datasets y artefactos de terceros están sujetos a sus propios términos y atribuciones; la publicación del código fuente **no autoriza por sí sola su redistribución**. La licencia del código de APETRIA no se declara hasta que el titular la seleccione expresamente.
