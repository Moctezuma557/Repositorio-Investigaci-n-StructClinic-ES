import os
from openai import OpenAI

# Configuración del cliente para Hugging Face Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="", # Aqui debes poner tu clave de API
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

            A continuación te paso un ejemplos de como debes de estrcuturar las respuestas:

            (1)

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

            Como antecedente epidemiológico destacable, la paciente convivía con 2 personas con COVID-19 demostrada, por lo que se realizó PCR a SARS-CoV-2 en aspirado naso faríngeo, que fue positiva.

            P (Plan):

            Se pautó tratamiento sintomático vía oral con buena respuesta. La duración de la mayoría de las lesiones fue inferior a 24h, resolviéndose la clínica cutánea en 5 días, sin otras manifestaciones asociadas.


            (2)

            Dada la nota:


            Varón de 48 años que acude a Urgencias por presentar un episodio de dolor torácico punzante no irradiado, de inicio súbito mientras conducía. No se acompaña de disnea, palpitaciones, ni síncope.

            Antecedentes, enfermedad actual y exploración física
            Varón de 48 años, sin alergias conocidas, con varios factores de riesgo cardiovascular entre los que destaca tabaquismo importante (2-3 paquetes/día), hipertensión arterial no tratada y obesidad moderada, sin otros antecedentes de interés. De profesión, labrador. Acude a Urgencias por presentar un episodio de dolor torácico punzante no irradiado, de inicio súbito mientras conducía. No se acompaña de disnea, palpitaciones ni síncope. A su llegada, presenta mal estado general, con sudoración profusa y taquipnea. La tensión arterial es de 200/81 mmHg, frecuencia cardiaca de 95 latidos por minuto y la saturación de oxigeno del 86%. A la exploración física destaca un soplo diastólico IV/VI, más llamativo en borde esternal izquierdo, hipoventilación generalizada con crepitantes en ambas bases pulmonares, edema tibiomaleolar con fóvea y pulsos periféricos presentes y simétricos. El resto de exploración física resulta anodina.

            Pruebas complementarias
            • ECG al ingreso: Ritmo sinusal a 90 lpm. PR normal. QRS estrecho. Criterios de voltaje para hipertrofia ventricular izquierda. T negativa en cara lateral.
            • Analítica Urgencias: Destaca glucemia 185 mg/dl, PCR 60 y marcadores de necrosis miocárdica elevados (TnT US máxima 180 ng/ml, sin curva típica).
            • Rx de tórax: Cardiomegalia. Elongación aórtica. Engrosamiento hiliar.
            • Angio TAC: Raíz aórtica y segmento proximal de aorta ascendente en el límite alto de la normalidad, con el resto de aorta de calibre normal, sin signos sugerentes de síndrome aórtico. Cardiomegalia ligera y derrame pericárdico de escasa cuantía. A nivel abdominal, únicamente destaca discreta hepatomegalia con esteatosis hepática.
            • Ecocardiograma transtorácico: Destaca como hallazgo principal la presencia de una insuficiencia aórtica moderada-severa sobre válvula trivalva, secundaria a dilatación de anillo aórtico. El ventrículo izquierdo está ligeramente dilatado (diámetro telediastólico de 60 mm) con hipertrofia concéntrica severa (15-16mm) y función sistólica global y segmentaria conservadas. Tanto el ventrículo derecho como el resto de válvulas no presentan hallazgos patológicos.
            • Resonancia magnética cardiaca: El estudio fue de calidad técnica subóptima por falta de colaboración por parte del paciente para realizar correctamente apneas. El ventrículo izquierdo mostraba una hipertrofia excéntrica moderada, dilatación (volumen telediástólico de 124 ml/m2 y telesistólico de 50 ml/m2) y función sistólica segmentaria y global dentro de la normalidad (fracción de eyección del 60%). El ventrículo derecho era normal en dimensiones y función sistólica. No se evidenció la presencia de edema ni alteraciones en la perfusión ni en el realce tardío con gadolinio. Se detectó una insuficiencia aórtica de apariencia significativa, pero la mala calidad del estudio no permitió una cuantificación adecuada. Se observó una dilatación de la raíz aórtica (senos de Valsalva: 50 mm, 21 mm/m2) y porción tubular de aorta ascendente (48 mm, 20 mm/m2).
            • Cateterismo cardiaco diagnóstico: El hallazgo principal es en la aortografía, donde se evidencia dilatación de aorta ascendente desde la raíz, con borramiento de los senos de Valsalva, e insuficiencia aórtica de grado IV. Las arterias coronarias no presentaban lesiones angiográficas significativas.
            • Ecocardiograma transesofágico: Se interrumpe por mala tolerancia dada la situación clínica del paciente (insuficiencia cardiaca). Únicamente se consigue visualizar una válvula aórtica trivalva con correcta apertura, sin poder valorar raíz aórtica ni la insuficiencia valvular.
            • Análisis en planta de Hospitalización: Destaca una insuficiencia renal moderada, con urea de 78 mg/dL, creatinina de 1,53 mg/dL y un filtrado glomerular estimado de 48,82 mL/minuto. Las enzimas hepáticas estaban ligeramente elevadas, y la hemoglobina glicosilada era de 7,8%. El resto de la bioquímica, así como el hemograma, eran normales.

            Evolución clínica
            Durante su estancia en Urgencias se realiza angio TAC para descartar síndrome aórtico agudo dada la presentación del paciente con emergencia hipertensiva, dolor punzante y elevación de troponina. El angio TAC es informado como normal, por lo que el paciente es ingresado en cardiología para estudio de dolor torácico con elevación de marcadores de necrosis miocárdica e insuficiencia cardiaca. Al segundo día de ingreso en Cardiología el paciente presenta un pico febril de 38,5o, sin foco infeccioso aparente (ni clínico ni en pruebas complementarias), y con hemocultivos repetidos negativos. Ante la presencia en el ecocardiograma transtorácico de una insuficiencia aórtica moderada-severa, se realiza un estudio transesofágico por sospecha endocarditis como posible etiología del cuadro clínico. La situación clínica del paciente hace que no tolere adecuadamente el procedimiento, y aunque no se observan signos claros de endocarditis, el estudio es de mala calidad y se interrumpe precozmente.
            En los días siguientes, el paciente no vuelve a presentar fiebre ni clínica infecciosa. No tiene tampoco nuevos episodios de dolor torácico; sin embargo, en los ECG seriados existe una negativización de ondas T en cara anterior, por lo que se solicita una resonancia magnética cardiaca de estrés, para valorar la presencia de necrosis o isquemia inducible. La resonancia resulta negativa, descartándose signos de miocarditis o necrosis miocárdica, pero de nuevo se observa la insuficiencia aórtica de apariencia severa.
            No obstante, dado que el paciente continúa presentando elevación progresiva de troponina y persisten los cambios eléctricos (profundización de onda T en cara lateral) se solicita cateterismo cardiaco diagnóstico, que objetiva una lesión no significativa en ACD y confirma la severidad de la insuficiencia aórtica. El paciente al 5o día de ingreso presenta un nuevo pico febril de 38o, y episodio de fibrilación auricular rápida con cardioversión farmacológica y eléctrica fallidas, en el contexto de descompensación leve de insuficiencia cardiaca. De nuevo se solicita batería de pruebas para descartar foco infeccioso, que se descarta, con hemocultivos nuevamente negativos. El 7o día de ingreso en Cardiología, presenta un episodio de desaturación, taquipnea e hipotensión, compatible con edema agudo de pulmón por lo que ingresa en la Unidad Coronaria y finalmente requiere intubación orotraqueal y ventilación mecánica, así como soporte inotrópico con drogas vasoactivas. Se realiza ecocardiograma transesofácico con el paciente intubado, descartándose endocarditis y finalmente se objetiva una disección localizada de la raíz aórtica responsable de la insuficiencia aórtica severa, por lo que se contacta con Cirugía Cardiovascular para intervención urgente. Intraoperatoriamente se objetiva disección tipo A, apreciándose rotura intimal circunferencial a nivel de la unión sinotubular originando descolgamiento de todas las comisuras, siendo ese el mecanismo de la insuficiencia aórtica observada.
            Durante su ingreso en la Unidad Coronaria tras la cirugía presenta una evolución lenta, aunque favorable, sin nuevos picos febriles y con mejoría de la clínica de insuficiencia cardiaca, siendo dado de alta al 7o día post-cirugía. Al mes, el paciente está clínicamente estable y asintomático desde el punto de vista cardiológico, sin insuficiencia aórtica en el ecocardiograma de control ambulatorio.

            Diagnóstico
            Disección localizada de raíz aórtica con insuficiencia aórtica severa aguda.


            Se estructura de la siguiente forma:

            S (Subjetive):

            Varón de 48 años que acude a Urgencias por presentar un episodio de dolor torácico punzante no irradiado, de inicio súbito mientras conducía.
            No se acompaña de disnea, palpitaciones, ni síncope.
            Antecedentes, enfermedad actual y exploración física Varón de 48 años, sin alergias conocidas, con varios factores de riesgo cardiovascular entre los que destaca tabaquismo importante (2-3 paquetes/día), hipertensión arterial no tratada y obesidad moderada, sin otros antecedentes de interés.
            De profesión, labrador. Acude a Urgencias por presentar un episodio de dolor torácico punzante no irradiado, de inicio súbito mientras conducía.


            O (Objective):



            No se acompaña de disnea, palpitaciones ni síncope. A su llegada, presenta mal estado general, con sudoración profusa y taquipnea.
             La tensión arterial es de 200/81 mmHg, frecuencia cardiaca de 95 latidos por minuto y la saturación de oxigeno del 86\%.
             A la exploración física destaca un soplo diastólico IV/VI, más llamativo en borde esternal izquierdo, hipoventilación generalizada con crepitantes en ambas bases pulmonares, edema tibiomaleolar con fóvea y pulsos periféricos presentes y simétricos.
             El resto de exploración física resulta anodina.
             Pruebas complementarias • ECG al ingreso: Ritmo sinusal a 90 lpm. PR normal. QRS estrecho. Criterios de voltaje para hipertrofia ventricular izquierda. T negativa en cara lateral.
             • Analítica Urgencias: Destaca glucemia 185 mg/dl, PCR 60 y marcadores de necrosis miocárdica elevados (TnT US máxima 180 ng/ml, sin curva típica).
             • Rx de tórax: Cardiomegalia. Elongación aórtica. Engrosamiento hiliar.
             Angio TAC: Raíz aórtica y segmento proximal de aorta ascendente en el límite alto de la normalidad, con el resto de aorta de calibre normal, sin signos sugerentes de síndrome aórtico.
             Cardiomegalia ligera y derrame pericárdico de escasa cuantía. A nivel abdominal, únicamente destaca discreta hepatomegalia con esteatosis hepática.
             • Ecocardiograma transtorácico: Destaca como hallazgo principal la presencia de una insuficiencia aórtica moderada-severa sobre válvula trivalva, secundaria a dilatación de anillo aórtico.
             El ventrículo izquierdo está ligeramente dilatado (diámetro telediastólico de 60 mm) con hipertrofia concéntrica severa (15-16mm) y función sistólica global y segmentaria conservadas.
             Tanto el ventrículo derecho como el resto de válvulas no presentan hallazgos patológicos.
             • Resonancia magnética cardiaca: El estudio fue de calidad técnica subóptima por falta de colaboración por parte del paciente para realizar correctamente apneas.
             El ventrículo izquierdo mostraba una hipertrofia excéntrica moderada, dilatación (volumen telediástólico de 124 ml/m2 y telesistólico de 50 ml/m2) y función sistólica segmentaria y global dentro de la normalidad (fracción de eyección del 60\%).
             El ventrículo derecho era normal en dimensiones y función sistólica.
             No se evidenció la presencia de edema ni alteraciones en la perfusión ni en el realce tardío con gadolinio.
             Se detectó una insuficiencia aórtica de apariencia significativa, pero la mala calidad del estudio no permitió una cuantificación adecuada.
             Se observó una dilatación de la raíz aórtica (senos de Valsalva: 50 mm, 21 mm/m2) y porción tubular de aorta ascendente (48 mm, 20 mm/m2).
             • Cateterismo cardiaco diagnóstico: El hallazgo principal es en la aortografía, donde se evidencia dilatación de aorta ascendente desde la raíz, con borramiento de los senos de Valsalva, e insuficiencia aórtica de grado IV.
             Las arterias coronarias no presentaban lesiones angiográficas significativas.
             Ecocardiograma transesofágico: Se interrumpe por mala tolerancia dada la situación clínica del paciente (insuficiencia cardiaca).
             Únicamente se consigue visualizar una válvula aórtica trivalva con correcta apertura, sin poder valorar raíz aórtica ni la insuficiencia valvular.
             • Análisis en planta de Hospitalización: Destaca una insuficiencia renal moderada, con urea de 78 mg/dL, creatinina de 1,53 mg/dL y un filtrado glomerular estimado de 48,82 mL/minuto.
             Las enzimas hepáticas estaban ligeramente elevadas, y la hemoglobina glicosilada era de 7,8\%.
             El resto de la bioquímica, así como el hemograma, eran normales.


            A(assessment):

             Evolución clínica Durante su estancia en Urgencias se realiza angio TAC para descartar síndrome aórtico agudo dada la presentación del paciente con emergencia hipertensiva, dolor punzante y elevación de troponina.
             El angio TAC es informado como normal, por lo que el paciente es ingresado en cardiología para estudio de dolor torácico con elevación de marcadores de necrosis miocárdica e insuficiencia cardiaca.
             Al segundo día de ingreso en Cardiología el paciente presenta un pico febril de 38,5o, sin foco infeccioso aparente (ni clínico ni en pruebas complementarias), y con hemocultivos repetidos negativos.
             Ante la presencia en el ecocardiograma transtorácico de una insuficiencia aórtica moderada-severa, se realiza un estudio transesofágico por sospecha endocarditis como posible etiología del cuadro clínico.
             La situación clínica del paciente hace que no tolere adecuadamente el procedimiento, y aunque no se observan signos claros de endocarditis, el estudio es de mala calidad y se interrumpe precozmente.
             En los días siguientes, el paciente no vuelve a presentar fiebre ni clínica infecciosa.
             No tiene tampoco nuevos episodios de dolor torácico; sin embargo, en los ECG seriados existe una negativización de ondas T en cara anterior, por lo que se solicita una resonancia magnética cardiaca de estrés, para valorar la presencia de necrosis o isquemia inducible.
             La resonancia resulta negativa, descartándose signos de miocarditis o necrosis miocárdica, pero de nuevo se observa la insuficiencia aórtica de apariencia severa.
             No obstante, dado que el paciente continúa presentando elevación progresiva de troponina y persisten los cambios eléctricos (profundización de onda T en cara lateral) se solicita cateterismo cardiaco diagnóstico, que objetiva una lesión no significativa en ACD y confirma la severidad de la insuficiencia aórtica.
             El paciente al 5o día de ingreso presenta un nuevo pico febril de 38o, y episodio de fibrilación auricular rápida con cardioversión farmacológica y eléctrica fallidas, en el contexto de descompensación leve de insuficiencia cardiaca.
             De nuevo se solicita batería de pruebas para descartar foco infeccioso, que se descarta, con hemocultivos nuevamente negativos.
             El 7o día de ingreso en Cardiología, presenta un episodio de desaturación, taquipnea e hipotensión, compatible con edema agudo de pulmón por lo que ingresa en la Unidad Coronaria y finalmente requiere intubación orotraqueal y ventilación mecánica, así como soporte inotrópico con drogas vasoactivas.
             Se realiza ecocardiograma transesofácico con el paciente intubado, descartándose endocarditis y finalmente se objetiva una disección localizada de la raíz aórtica responsable de la insuficiencia aórtica severa.
             Diagnóstico Disección localizada de raíz aórtica con insuficiencia aórtica severa aguda.


            P(plan):

             ...por lo que se contacta con Cirugía Cardiovascular para intervención urgente.
             Intraoperatoriamente se objetiva disección tipo A, apreciándose rotura intimal circunferencial a nivel de la unión sinotubular originando descolgamiento de todas las comisuras, siendo ese el mecanismo de la insuficiencia aórtica observada.
             Durante su ingreso en la Unidad Coronaria tras la cirugía presenta una evolución lenta, aunque favorable, sin nuevos picos febriles y con mejoría de la clínica de insuficiencia cardiaca, siendo dado de alta al 7o día post-cirugía.
             Al mes, el paciente está clínicamente estable y asintomático desde el punto de vista cardiológico, sin insuficiencia aórtica en el ecocardiograma de control ambulatorio.


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