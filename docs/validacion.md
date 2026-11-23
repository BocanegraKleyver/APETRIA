# Validación funcional de la distribución independiente

## Instalación local

El **23/11/2026** se verificó en Windows con Python **3.11.1**:

- Entorno `.venv` nuevo e independiente del repositorio académico.
- Instalación de las 13 dependencias declaradas con `pip install -r requirements.txt`.
- `pip check`: `No broken requirements found`.
- Importación correcta de dependencias y `backend.app.main`.
- Creación de administrador local y acceso a la consola desde FastAPI en puerto `8000`.
- Rechazo HTTP 401 de una contraseña incorrecta y aceptación HTTP 200 de credenciales válidas.

## PF-13: transporte WebSocket

Se ejecutó el validador PF-13 contra la instancia independiente de APETRIA que utilizaba el paquete de demostración de 14 archivos.

| Fuente | Recibidas | Finalización | Transporte | Identificadores únicos |
|---|---:|---:|---|---|
| SMD | 10/10 | 10 | OK | Sí (`point_index`) |
| NASA | 10/10 | 10 | OK | Sí (`point_index`) |
| NAB | 10/10 | 10 | OK | Sí (`point_index`) |
| HDFS | 10/10 | 10 | OK | Sí (`block_id`) |
| BGL | 10/10 | 10 | OK | Sí (`line_number`) |

**Resultado global:** `Transporte de cinco fuentes: True`.

La evidencia JSON fue generada en el repositorio académico privado, en `results/functional_tests/pf13/PF13_five_sources_20261123_173610_450266.json`. Esa ruta no forma parte del repositorio público.

## Alcance de la prueba

PF-13 confirma autenticación de la sesión usada para prueba, recepción de diez observaciones por fuente, mensaje de finalización y verificaciones de identificadores. **No certifica** precisión, recall, F1, FPR, throughput, robustez de producción, seguridad integral ni trazabilidad completa de cada archivo del ZIP.

La prueba se ejecutó con un script ubicado en el repositorio experimental privado. Si el validador se incorpora más adelante al repositorio público, deberá conservarse su versión y documentarse el comando exacto.

## Verificación de integridad

El 23/11/2026 se verificó el SHA-256 del ZIP demo.
También se compararon individualmente los 14 archivos
extraídos con los de la instalación independiente.

**Resultado: 14 archivos verificados, 0 diferencias.**

La integridad no implica autorización de redistribución.
