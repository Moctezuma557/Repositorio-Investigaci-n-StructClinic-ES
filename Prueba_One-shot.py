import os
from openai import OpenAI

# Configuración del cliente para Hugging Face Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="",# Aqui debes poner tu clave de API
)

# 1. NOTA CLÍNICA EXTENSA (Ejemplo para prueba de estrés)
nota_extensa = """

Escolar de 11 años que es remitida a la consulta de Alergología Pediátrica por haber presentado a las 2 horas de la toma de un comprimido de lansoprazol 30mg por epigastralgia, cuadro de urticaria generalizada sin otra clínica acompañante, que remitió en 12 horas. En consulta se realizan pruebas cutáneas en prick e intradermorreacción con lansoprazol, omeprazol, pantoprazol, esomeprazol y rabeprazol a concentraciones descritas en la literatura: omeprazol 4mg/ ml, lansoprazol 1.5mg/ml, pantoprazol 4mg/ml, esomeprazol 4mg/ml y rabeprazol 1mg/ml.

Resultados:
En la lectura inmediata de las pruebas cutáneas obtenemos un resultado positivo en intradermorreacción para lansoprazol, siendo el resto de pruebas negativas. Ante el resultado de las pruebas y tras revisar los diferentes patrones de reactividad cruzada descritos en la literatura, decimos realizar prueba de provocación oral con omeprazol para poder ofrecer alternativas en caso de necesitar un IBP. Se realizó administrando una dosis total acumulada de 30mg, con buena tolerancia.
Con estos resultados, hacemos el diagnostico de urticaria aguda por alergia IgE mediada a lansoprazol, recomendamos su evitación y permitimos la administración de omeprazol si lo precisa.

Conclusiones:
Ante la sospecha de alergia a un fármaco de los IBP, recomendamos realizar el estudio con todos los fármacos del grupo para valorar la existencia de posibles reacciones cruzadas y buscar posibles alternativas seguras en estos pacientes.




"""

# 2. EJECUCIÓN DEL MODELO CON PROMPT ESTRUCTURADO
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # <-- Aqui va el modelo que se usara
    messages=[
        {
            "role": "system",
            "content": """Eres un transcriptor médico especializado en el formato SOAP.
            Tu única función es SEGMENTAR el texto que recibas.
            REGLAS CRÍTICAS:
            1. No modifiques ni una sola palabra del texto original.
            2. No resumas, no corrijas ortografía ni añadidas comentarios.
            3. Organiza el texto original exactamente como está escrito dentro de las categorías:
               - S (Subjetivo): Síntomas, antecedentes y relato del paciente.
               - O (Objetivo): Signos vitales, exploración física y resultados de estudios.
               - A (Evaluación): Diagnósticos e impresiones clínicas.
               - P (Plan): Tratamientos, órdenes médicas y seguimiento.
            4. Si una sección no contiene información, escribe 'Información no disponible'.

            A continuación te paso un ejemplo de como debes de estrcuturar las respuestas:


            Dada la nota:

            El segundo caso que se presenta es el de una lactante de 2 meses que acudió a urgencias por febrícula y un cuadro de urticaria aguda, aparentemente pruriginoso, de 4 días de evolución.

            Inicialmente afectaba la cara y las extremidades superiores, extendiéndose en pocas horas al tronco y las extremidades inferiores. Exantema urticariforme. No había afectación palmoplantar.

            Estas manifestaciones no se acompañaban de angioedema acral, labial ni lingual.

            Como antecedente epidemiológico destacable, la paciente convivía con 2 personas con COVID-19 demostrada, por lo que se realizó PCR a SARS-CoV-2 en aspirado nasofaríngeo, que fue positiva.

            Se pautó tratamiento sintomático vía oral con buena respuesta. La duración de la mayoría de las lesiones fue inferior a 24h, resolviéndose la clínica cutánea en 5 días,
            sin otras manifestaciones asociadas.


            Se estructura de la siguiente forma:

            S (Subjetive):

            El segundo caso que se presenta es el de una lactante de 2 meses que acudió a urgencias por febrícula y un cuadro de urticaria aguda, aparentemente pruriginoso, de 4 días de evolución.

            O (Objective):

            Inicialmente afectaba la cara y las extremidades superiores, extendiéndose en pocas horas al tronco y las extremidades inferiores. Exantema urticariforme.

            No había afectación palmoplantar. Estas manifestaciones no se acompañaban de angioedema acral, labial ni lingual.

            A (Assessment):

            Como antecedente epidemiológico destacable, la paciente convivía con 2 personas con COVID-19 demostrada, por lo que se realizó PCR a SARS-CoV-2 en aspirado nasofaríngeo, que fue positiva.

            P (Plan):

            Se pautó tratamiento sintomático vía oral con buena respuesta. La duración de la mayoría de las lesiones fue inferior a 24h, resolviéndose la clínica cutánea en 5 días, sin otras manifestaciones asociadas.

            """
        },
        {
            "role": "user",
            "content": f"Segmenta la siguiente nota clínica en formato SOAP sin alterar el texto:\n\n{nota_extensa}"
        }
    ],
    temperature=0.1, # Temperatura baja para máxima fidelidad y cero creatividad
)

print("-" * 30)
print("RESULTADO STRUCTCLINIC-ES ()")
print("-" * 30)
print(completion.choices[0].message.content)