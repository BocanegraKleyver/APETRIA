# Administrador local y protección de credenciales

APETRIA utiliza **una cuenta administradora por instalación**, persistida en SQLite local en `instance/auth.db`. No se proporciona una contraseña predeterminada ni hay envío de contraseñas por correo electrónico.

## Crear usuario

Desde la raíz del repositorio con las dependencias instaladas:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m backend.app.cli.create_admin
```

Elegir un nombre de usuario y una contraseña de al menos **12 caracteres**. La contraseña no se muestra mientras se introduce.

## Iniciar sesión

Arrancar el servidor:

```powershell
& $py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Acceder a <http://127.0.0.1:8000/>. Desde el menú de usuario se puede cambiar la contraseña o cerrar la sesión.

## Controles implementados

- Hash PBKDF2-HMAC-SHA256 para las contraseñas, con sal aleatoria.
- Cookie de sesión `HttpOnly`, `SameSite=Strict`; indicador `Secure` cuando se utiliza HTTPS.
- Sesiones de duración máxima de ocho horas.
- Protección CSRF para operaciones que alteran estado.
- Los endpoints de fuentes y configuración y las conexiones WebSocket exigen una sesión válida.
- Rechazo de credenciales incorrectas (`HTTP 401`) y bloqueo temporal ante varios intentos fallidos.

## Archivos que nunca se deben publicar

```text
instance/auth.db
instance/runtime_settings.json
.env
.env.*
.venv/
```

Todo `instance/` y el entorno virtual deben permanecer fuera de Git y fuera del ZIP de datos. Antes de hacer `git push`, inspeccionar siempre `git status` y `git ls-files`.

## Restricciones

La autenticación está pensada para evaluación **local**. No exponer la aplicación directamente en Internet, no utilizar `--host 0.0.0.0` como sustituto de un despliegue seguro y no compartir capturas con contraseñas, tokens o cookies.
