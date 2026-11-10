<p align="center"><b><font size="5">Predicción del S&P 500 con SARIMAX y LSTM</font></b></p>

**Descripción general:**

La meta del proyecto es crear una aplicación para la predicción de SP500 mediante dos enfoques distintos de modelado temporal, Sarimax y LSTM. El motivo de incluir estos dos enfoques es para comparar su desempeño, ya que cada una captura estructuras diferentes, siendo que Sarimax modela relaciones estadísticas explícitas y estructuras más lineales, mientras que LSTM captura relaciones no lineales. De esta manera un modelo podría ser más preciso en momentos de alta volatilidad, por ejemplo. 

El resultado de este desarrollo es una aplicación con dos modelos que se reentrenan cada 15 días. Se incluyen las métricas de los modelos cuando fueron entrenados y cuando es su próximo entrenamiento. Se incluye la volatilidad, la predicción de los precios y retornos, así como el ratio de retorno, todo esto con los dos modelos. 
 
<img width="589" height="274" alt="image" src="https://github.com/user-attachments/assets/ccba0cd2-1c9b-4cc7-b125-fdab4956e572" />
<img width="589" height="249" alt="image" src="https://github.com/user-attachments/assets/083945f9-6bcb-4e49-b71d-6bac150c3c09" />

**Dataset:**

El dataset del sp500 se descarga de la librería yfinance y se actualiza automáticamente mediante la aplicación. Para el proyecto las columnas utilizadas fueron el índice, que es la fecha, y los precios de cierre.

**Métricas:**

Para el modelado de precios se usó MAPE como métrica principal. Para retornos se usó RMSE y MAE, dejando fuera a MAPE ya que los retornos suelen encontrarse próximos a cero, lo que hace que MAPE pierda fiabilidad. En la aplicación para la volatilidad se usó GARCH y con ella se calculó el ratio de retorno-riesgo.

**Características de la aplicación:**

Cada 15 días se descarga un nuevo conjunto de datos históricos del S&P 500. Posteriormente se realizan las transformaciones necesarias sobre precios y retornos, y se reentrenan los modelos SARIMAX y LSTM utilizando los parámetros seleccionados durante la fase de validación. Durante este proceso se almacenan las métricas de seguimiento del modelo. Se decidió que el modelo GARCH se actualizara en cada predicción, ya que su cálculo tomaba milisegundos.

Entre reentrenamientos, la aplicación descarga únicamente los datos más recientes y los incorpora a la serie histórica. Estos datos actualizados se almacenan en caché para evitar descargas y cálculos innecesarios cada vez que el usuario solicita una nueva predicción.

Cuando el usuario selecciona un horizonte de predicción, la aplicación utiliza los modelos previamente entrenados para estimar:

- Precio esperado.
- Retorno diario esperado.
- Retorno acumulado esperado.
- Volatilidad esperada mediante GARCH.
- Ratio retorno-riesgo.

Además, se generan intervalos de confianza para precios y retornos, permitiendo visualizar tanto la predicción central como la incertidumbre asociada.

**Archivos principales:**

-	sp500_despliegue.py: aplicación Streamlit desplegada
-	Analisis_sp500.ipynb: análisis completo y desarrollo de modelos.

**Tecnologías utilizadas:**

-	Python
-	Pandas
-	NumPy
-	Statsmodels
-	arch
-	PyTorch
-	yfinance
-	Plotly
-	Streamlit
-	GitHub

**Conclusiones:**

-	SARIMAX y LSTM mostraron puntajes similares en donde ninguna presentó una superioridad clara y consistente.
-	Los modelos de SARIMAX entrenados con un año mostraron mejores métricas que con tres, lo que sugiere que para predicciones de días inmediatos los datos recientes son más relevantes.
-	Las pruebas realizadas con ventanas móviles mostraron que el desempeño de SARIMAX varía considerablemente dependiendo del periodo utilizado para la predicción, lo que indica que hay periodos cuya estructura es más compleja y por lo tanto menos propensa a predicciones, algo esperable en mercados financieros debido a la influencia de factores externos que no se encuentran contenidos en la serie temporal analizada. Lo anterior implica la necesidad de reentrenamiento constante con datos actualizados y el monitoreo de métricas claves.
-	Aunque fue entrenada con alrededor de 250 datos LSTM mostró buenos resultados, caso que no se esperaba ya que estas redes suelen beneficiarse de una mayor cantidad de entradas.
-	El entrenamiento de LSTM con ventanas de 21 días, transformaciones logarítmicas y horizonte de predicción de un día, la hizo competente a niveles de modelos clásicos.

**Aplicación desplegada**

La aplicación de los modelos se puede ver en esta liga: https://predicciones-del-sp500-con-sarimax-y-lstm-o9p8nwhhfnw5cc6sw4ap.streamlit.app/

