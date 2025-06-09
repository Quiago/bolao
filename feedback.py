import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

def send_feedback_email(rating: str, comment: str) -> str:
    """
    Envía feedback por email a Gmail
    """
    try:
        # Configuración Gmail
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = os.getenv('GMAIL_USER')  # Tu email de Gmail
        EMAIL_PASS = os.getenv('GMAIL_APP_PASSWORD')  # Contraseña de aplicación
        RECIPIENT_EMAIL = os.getenv('FEEDBACK_EMAIL', EMAIL_USER)  # Donde recibir feedback
        
        # Validar configuración
        if not EMAIL_USER or not EMAIL_PASS:
            # Guardar localmente si no hay configuración de email
            return save_feedback_locally(rating, comment)
        
        # Crear mensaje de email
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"📝 Nuevo Feedback BOLAO - {rating if rating else 'Sin calificación'}"
        
        # Cuerpo del email en HTML
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px 10px 0 0;">
                    <h2 style="color: white; margin: 0;">🔍 BOLAO - Nuevo Feedback</h2>
                </div>
                
                <div style="background: #f9fafb; padding: 20px; border-radius: 0 0 10px 10px; border: 1px solid #e5e7eb;">
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h3 style="color: #1f2937; margin-top: 0;">📊 Calificación:</h3>
                        <p style="font-size: 18px; color: #3b82f6; margin: 5px 0;">
                            {rating if rating else "❌ Sin calificación"}
                        </p>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h3 style="color: #1f2937; margin-top: 0;">💭 Comentario:</h3>
                        <p style="color: #4b5563; line-height: 1.6; margin: 5px 0;">
                            {comment if comment.strip() else "❌ Sin comentario"}
                        </p>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <h3 style="color: #1f2937; margin-top: 0;">📅 Información:</h3>
                        <p style="color: #6b7280; margin: 5px 0;"><strong>Fecha y hora:</strong> {timestamp}</p>
                        <p style="color: #6b7280; margin: 5px 0;"><strong>Aplicación:</strong> BOLAO - Búsqueda Inteligente</p>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; color: #9ca3af; font-size: 12px;">
                    Este feedback fue enviado automáticamente desde la aplicación BOLAO
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Enviar email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()  # Habilitar seguridad
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
        server.quit()
        
        # También guardar localmente como backup
        save_feedback_locally(rating, comment)
        
        return "✅ ¡Gracias por tu feedback! Tu mensaje ha sido enviado correctamente."
        
    except smtplib.SMTPAuthenticationError:
        save_feedback_locally(rating, comment)
        return "❌ Error de autenticación. Verifica tu contraseña de aplicación de Gmail."
    except smtplib.SMTPException as e:
        save_feedback_locally(rating, comment)
        return f"❌ Error al enviar email: {str(e)}"
    except Exception as e:
        save_feedback_locally(rating, comment)
        return f"❌ Error inesperado: {str(e)}"

def save_feedback_locally(rating: str, comment: str) -> str:
    """Guarda feedback localmente como backup"""
    try:
        feedback_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'rating': rating,
            'comment': comment.strip()
        }
        
        with open('feedback_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{feedback_data}\n")
        
        if rating or comment.strip():
            return "✅ Feedback guardado localmente. Configuración de email pendiente."
        else:
            return "⚠️ Por favor selecciona una calificación o escribe un comentario."
    except Exception as e:
        return f"❌ Error al guardar feedback: {str(e)}"

def test_email_config():
    """Función para probar la configuración de email"""
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = os.getenv('GMAIL_USER', "test")
        EMAIL_PASS = os.getenv('GMAIL_APP_PASSWORD', "test")
        
        if not EMAIL_USER or not EMAIL_PASS:
            return "❌ Faltan variables de entorno: GMAIL_USER y GMAIL_APP_PASSWORD"
        
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.quit()
        
        return f"✅ Configuración de email correcta para: {EMAIL_USER}"
    except Exception as e:
        return f"❌ Error en configuración: {str(e)}"

# Ejemplo de uso y prueba
if __name__ == "__main__":
    print("🧪 Probando configuración de email...")
    print(test_email_config())
    
    print("\n📧 Enviando feedback de prueba...")
    result = send_feedback_email("⭐⭐⭐⭐⭐ Excelente", "Esta es una prueba del sistema de feedback")
    print(result)