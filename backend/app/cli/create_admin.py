"""Crear el único administrador local: python -m backend.app.cli.create_admin."""
from getpass import getpass
from backend.app.auth_store import create_admin, has_admin, DB_PATH


def main():
    print("\nAPETRIA — Configuración inicial del administrador")
    if has_admin():
        print("Ya existe un administrador en esta instalación. No se creó otra cuenta.")
        return 1
    username = input("Usuario administrador [admin]: ").strip() or "admin"
    password = getpass("Contraseña (mínimo 12 caracteres): ")
    confirmation = getpass("Confirmar contraseña: ")
    if password != confirmation:
        print("Las contraseñas no coinciden.")
        return 1
    try:
        create_admin(username, password)
    except ValueError as exc:
        print("No se pudo crear el administrador:", exc)
        return 1
    print("Administrador creado correctamente.")
    print("Base local:", DB_PATH)
    print("Ingresá a http://127.0.0.1:8000/ y usá tus credenciales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
