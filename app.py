import streamlit as st
import os
import stable_whisper
import tempfile

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Culqi Transcriptor", page_icon="🎙️", layout="wide")

st.title("🎙️ Culqi Transcriptor Pro")
st.markdown("Carga tus audios de atención al cliente para obtener transcripciones precisas con el modelo Whisper Turbo.")

# --- CARGA DEL MODELO (Caché para evitar recargas constantes) ---
@st.cache_resource
def load_whisper_model():
    return stable_whisper.load_model('turbo')

model = load_whisper_model()

# --- BARRA LATERAL (Configuración) ---
with st.sidebar:
    st.header("Configuración")
    st.info("Este modelo detecta automáticamente pausas y terminología específica de Culqi.")
    gap_threshold = st.slider("Umbral de silencio (segundos)", 0.5, 3.0, 1.5)
    
# --- COMPONENTE DE CARGA DE ARCHIVOS ---
uploaded_files = st.file_uploader(
    "Arrastra y suelta tus archivos de audio aquí (.mp3, .wav)", 
    type=["mp3", "wav"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Archivos listos para procesar: {len(uploaded_files)}")
    
    if st.button("🚀 Iniciar Transcripción"):
        for uploaded_file in uploaded_files:
            with st.status(f"Procesando: {uploaded_file.name}...", expanded=True) as status:
                try:
                    # Crear un archivo temporal para que stable-whisper pueda leerlo
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Transcripción con tus parámetros específicos
                    result = model.transcribe(
                        tmp_path, 
                        language='es',
                        beam_size=1,
                        best_of=1,
                        no_speech_threshold=None, 
                        logprob_threshold=-1.0,
                        initial_prompt="Culqi, empresa, terminal, transacciones, soporte técnico, asesor, cliente, llamada, atención al cliente, KAM, plazos de abono, encuestas, RUC, número de whatsapp, abonos, consultas por transacciones, incidencias técnicas, problemas de señal, enviar evidencias, pruebas, contómetros, reclamos, CULQI, cuenta bancaria, cambio de contraseña, cambio de correo",
                        fp16=False,
                        vad=True 
                    )

                    # Generar contenido del texto
                    full_text = f"TRANSCRIPCIÓN COMPLETA: {uploaded_file.name}\n"
                    full_text += "="*40 + "\n\n"
                    
                    ultimo_fin = 0
                    for segment in result.segments:
                        if (segment.start - ultimo_fin) > gap_threshold:
                            full_text += f"\n{'-'*15} POSIBLE CAMBIO DE TURNO O PAUSA {'-'*15}\n\n"
                        
                        full_text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}\n"
                        ultimo_fin = segment.end

                    # Mostrar resultado en la interfaz
                    st.text_area(f"Resultado: {uploaded_file.name}", full_text, height=300)
                    
                    # Botón de descarga para el usuario
                    st.download_button(
                        label=f"📥 Descargar TXT - {uploaded_file.name}",
                        data=full_text,
                        file_name=f"Resultado_{uploaded_file.name.replace('.mp3', '.wav')}.txt",
                        mime="text/plain"
                    )
                    
                    os.remove(tmp_path) # Limpiar archivo temporal
                    status.update(label=f"✅ {uploaded_file.name} completado", state="complete")

                except Exception as e:
                    st.error(f"Error procesando {uploaded_file.name}: {e}")