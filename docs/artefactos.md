# Artefactos reales de demostración

APETRIA distribuye el código fuente por GitHub y los **artefactos de datos/modelos por separado**. El paquete demo permite evaluar las cinco fuentes sin descargar todo el material experimental.

## Paquete demo publicado y verificado

- **Versión:** [`v1.0.0` — APETRIA E100](https://github.com/BocanegraKleyver/APETRIA/releases/tag/v1.0.0)
- **Nombre:** `APETRIA_E100_Demo_5_fuentes_v1.0.0.zip`
- **Descarga directa:** [APETRIA_E100_Demo_5_fuentes_v1.0.0.zip](https://github.com/BocanegraKleyver/APETRIA/releases/download/v1.0.0/APETRIA_E100_Demo_5_fuentes_v1.0.0.zip)
- **Contenido:** 20 archivos (14 artefactos técnicos + 4 avisos de terceros + `ATRIBUCIONES.md` + `MANIFEST_SHA256.txt`).
- **Tamaño original de los 14 artefactos:** aproximadamente **16,72 MiB**.
- **Tamaño ZIP publicado:** **8.392.165 bytes**, aproximadamente **8 MiB**.
- **SHA-256 del ZIP publicado:**

```text
35F4A53BA076350A4041BCB9EA6B47DC7D8220B0C0D9BCD4F32AFEDBE1BBA504
```

El digest SHA-256 informado por GitHub Releases coincide con el del ZIP local verificado. Los 14 artefactos técnicos se compararon individualmente y no presentaron diferencias respecto de la selección original. El código y los artefactos se distribuyen por separado. Los modelos, datos procesados y resultados históricos incluidos deben usarse respetando las atribuciones y condiciones originales de las fuentes.

**Datos y artefactos experimentales completos:** descarga por Drive todavía no publicada; no son necesarios para ejecutar esta demostración mínima.

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
$zip = Join-Path $env:USERPROFILE 'Downloads\APETRIA_E100_Demo_5_fuentes_v1.0.0.zip'
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

Verificar que existan las 14 rutas técnicas documentadas, los cuatro avisos bajo `THIRD_PARTY_LICENSES/`, `ATRIBUCIONES.md` y `MANIFEST_SHA256.txt` (20 archivos en total), sin `.env`, `instance/` ni credenciales. El manifiesto contiene los SHA-256 de los 14 artefactos técnicos.

## Reproducibilidad y atribuciones

Las rutas contienen muestras y resultados de datasets reales. El paquete demo es una distribución técnica para evaluación local; **su redistribución debe respetar los términos de cada dataset** y los permisos pertinentes sobre modelos y artefactos derivados. No equivale a redistribuir los datasets de entrenamiento completos.

SMD y NASA generan inferencias durante la reproducción; NAB, HDFS y BGL reproducen resultados históricos. El protocolo de mensajes es WebSocket, pero esa circunstancia no significa que todos los datasets ejecuten inferencia nueva.
