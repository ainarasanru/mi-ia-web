import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Digit Recognizer")
st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro.")

# 1. Cargar el modelo guardado
@st.cache_resource
def load_my_model():
    # Asegúrate de que el archivo 'modelo_mnist.keras' esté en la misma carpeta
    return tf.keras.models.load_model('modelo_mnist.keras')

model = load_my_model()

# 2. Crear el lienzo (Canvas)
canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=20, 
    stroke_color="white", 
    background_color="black", 
    height=280, 
    width=280, 
    drawing_mode="freedraw", 
    key="canvas",
)

# 3. Procesar el dibujo y predecir
if canvas_result.image_data is not None:
    # Obtener la imagen del canvas
    img = canvas_result.image_data.astype('uint8')
    
    # Redimensionar a 28x28 (formato MNIST)
    img = cv2.resize(img, (28, 28))
    
    # Convertir a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalizar y cambiar forma para el modelo (1, 28, 28, 1)
    img = img / 255.0
    img_input = img.reshape(1, 28, 28, 1)

    # Predicción
    pred = model.predict(img_input)
    clase = np.argmax(pred)
    confianza = np.max(pred)

    # 4. Mostrar resultados
    st.subheader(f"Resultado: {clase}")
    if confianza < 0.80:
        st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
    else:
        st.success(f"Confianza alta: {confianza:.2%}")
    
    st.bar_chart(pred[0])
