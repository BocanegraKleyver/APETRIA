/* APETRIA — Autenticación real con cookie HttpOnly y sesión de FastAPI.
 * Los scripts de datos solo se cargan después de validar la sesión.
 */
(() => {
 'use strict';
 const $ = id => document.getElementById(id);
 const state = {csrf: null, username: null};
 window.APETRIA_AUTH = state;

 // La consola ya no debe consumirse mediante el servidor estático de desarrollo.
 if (location.port === '5173') {
   location.replace('http://127.0.0.1:8000/');
   return;
 }

 const explain = (message) => { $('auth-error').textContent = message; };
 async function readSession() {
   const response = await fetch('/api/auth/session', {cache: 'no-store'});
   if (!response.ok) throw new Error('No se pudo consultar el estado de la sesión.');
   return response.json();
 }
 function loadJS(src) {
   return new Promise((resolve, reject) => {
     const node = document.createElement('script');
     node.src=src;
     node.onload=resolve;
     node.onerror=()=>reject(new Error('No se pudo cargar '+src));
     document.body.appendChild(node);
   });
 }
 async function showApp(session) {
   state.csrf=session.csrf_token;
   state.username=session.username;
   $('profile-name').textContent=session.username;
   $('profile-username').textContent='Usuario: '+session.username+' · Administrador local';
   document.body.classList.add('authenticated');
   await loadJS('./js/app.js');
   await loadJS('./js/navigation.js');
 }
 async function initialize() {
   try {
     const session=await readSession();
     if (session.authenticated) {
       await showApp(session);
     } else if (!session.configured) {
       $('login-form').hidden=true;
       $('auth-title').textContent='Configuración inicial';
       $('auth-help').textContent='Todavía no existe un administrador en esta instalación. Abrí PowerShell en la raíz de APETRIA, ejecutá python -m backend.app.cli.create_admin y actualizá esta página.';
     }
   } catch (error) {
     explain(error.message || 'Error de conexión con FastAPI.');
   }
 }
 $('login-form').addEventListener('submit',async (event)=>{
   event.preventDefault();
   const button=$('login-button');
   button.disabled=true;explain('');
   try {
     const response=await fetch('/api/auth/login',{
       method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify({username:$('login-username').value.trim(),password:$('login-password').value})
     });
     if(!response.ok){const payload=await response.json().catch(()=>({}));throw new Error(payload.detail||'Acceso rechazado');}
     $('login-password').value='';
     location.reload();
   }catch(error){explain(error.message||'Error al iniciar sesión.');button.disabled=false;}
 });
 const overlay=$('profile-dialog');
 const closeProfile=()=>{overlay.hidden=true;};
 $('profile-button').addEventListener('click',()=>{overlay.hidden=false;$('old-password').focus();});
 $('profile-close').addEventListener('click',closeProfile);
 overlay.addEventListener('click',(event)=>{if(event.target===overlay)closeProfile();});
 document.addEventListener('keydown',(event)=>{if(event.key==='Escape'&&!overlay.hidden)closeProfile();});
 $('logout-button').addEventListener('click',async ()=>{
   try {
     const response=await fetch('/api/auth/logout',{method:'POST',headers:{'X-CSRF-Token':state.csrf||''}});
     if(!response.ok)throw new Error('No se pudo cerrar la sesión');
     // Recargar elimina del DOM todos los datos temporales de la reproducción.
     location.reload();
   }catch(error){$('profile-message').textContent=error.message||'Error al cerrar sesión';}
 });
 $('password-form').addEventListener('submit',async(event)=>{
   event.preventDefault();
   const old=$('old-password').value;
   const next=$('new-password').value;
   if(next!==$('confirm-password').value){$('profile-message').textContent='Las contraseñas no coinciden.';return;}
   const button=event.target.querySelector('[type="submit"]');button.disabled=true;
   try {
     const response=await fetch('/api/auth/change-password',{
       method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf||''},
       body:JSON.stringify({current_password:old,new_password:next})
     });
     if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(body.detail||'Error al actualizar contraseña');}
     $('profile-message').textContent='Contraseña actualizada. Volvé a iniciar sesión.';
     location.reload();
   }catch(error){$('profile-message').textContent=error.message||'Error al cambiar contraseña';button.disabled=false;}
 });
 void initialize();
})();
