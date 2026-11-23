# Artefactos reales de demostración

APETRIA distribuye el código fuente por GitHub y los **artefactos de datos/modelos por separado**. El paquete demo permite evaluar las cinco fuentes sin descargar todo el material experimental.

## Paquete demo verificado funcionalmente

- **Nombre:** `APETRIA_Demo_5_fuentes_20261123.zip`
- **Contenido esperado:** 14 archivos.
- **Tamaño original de la selección:** aproximadamente **16,72 MiB**.
- **Tamaño ZIP observado:** aproximadamente **8 MB**.
- **SHA-256 del ZIP generado el 23/11/2026:**

```text
C0FF73F5960509596AD7985C8EA43F30322BC9BE38BC7D41F32BE04921B37148
```

**Descarga del ZIP demo:** enlace público **pendiente de publicación**. No debe inventarse ni incorporarse un enlace de Google Drive que no haya sido confirmado. El archivo generado localmente es el que debe publicarse, sujeto a la comprobación de derechos de redistribución de sus contenidos.

**Datos/artefactos completos:** enlace o directorio de Drive **pendiente de publicación y revisión de permisos**. Los archivos completos no son obligatorios para esta demostración mínima.

## Manifiesto de los 14 archivos

Extraer el ZIP en la raíz del repositorio sin modificar las rutas:

```text
data/processed/smd/machine-1-1.npz
models/isolation_forest/machine-1-1.joblib
models/isolation_forest/machine-1-1_metadata.json
models/lstm_autoencoder/machine-1-1.pt
results/calibration/machine-1-1_lstm_threshold.json

data/processed/nasa/A-1.npz
models/nasa/isolation_forest/A-1.joblib
models/nasa/isolation_forest/A-1_metadata.json
models/nasa/lstm_autoencoder/A-1.pt
results/calibration/nasa/A-1_lstm_threshold.json

data/processed/nab/aws_v1/ec2_cpu_utilization_24ae8d.csv
results/hybrid/nab/nab_hybrid_predictions.csv.gz
results/hybrid/hdfs/hdfs_hybrid_predictions.csv
results/hybrid/bgl/bgl_hybrid_predictions.csv.gz
```

La fuente NAB utiliza la serie AWS y las predicciones históricas correspondientes. HDFS y BGL usan los resultados de reproducción históricos. Para SMD y NASA se cargan observaciones reales, modelos entrenados y umbrales calibrados; no se recalibran modelos al iniciar la aplicación.

## Verificación de integridad del ZIP

Desde PowerShell:

```powershell
$zip = Join-Path $env:USERPROFILE 'Downloads\APETRIA_Demo_5_fuentes_20261123.zip'
Get-FileHash -LiteralPath $zip -Algorithm SHA256
```

La salida debe coincidir exactamente con el hash anterior. Un hash correcto demuestra que el archivo ZIP no cambió desde la versión registrada; no equivale a un análisis antimalware ni a una autorización de redistribución.

Para inspeccionar la lista de rutas **sin extraer**:

```powershell
Add-Type -AssemblyName System.IO.Compression
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
  $archive.Entries | Where-Object { $_.Name } | ForEach-Object { $_.FullName }
} finally {
  $archive.Dispose()
}
```

Verificar que solo existan las 14 rutas documentadas y que no se incluyan `.env`, `instance/`, contraseñas, credenciales ni archivos ajenos al paquete.

## Reproducibilidad y atribuciones

Las rutas contienen muestras y resultados de datasets reales. El paquete demo es una distribución técnica para evaluación local; **su publicación externa requiere comprobar los términos de cada dataset** y los permisos pertinentes sobre modelos y artefactos derivados. No equivale a redistribuir los datasets de entrenamiento completos.

SMD y NASA generan inferencias durante la reproducción; NAB, HDFS y BGL reproducen resultados históricos. El protocolo de mensajes es WebSocket, pero esa circunstancia no significa que todos los datasets ejecuten inferencia nueva.
