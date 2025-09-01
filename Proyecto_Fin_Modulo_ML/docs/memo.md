Proyecto Fin de Modulo - ML

28-07-2025: Comenzamos con la limpieza y el análisis descriptivo de los datos. Parece ser un dataset de clientes correspondientes a diversas compañías de crédito y las quejas y asuntos correspondientes a dichas quejas.

01-08-2025: Se han eliminado todos los valores faltantes y se ha reordenado la columna State utilizando los prefijos postales, con el objetivo de corregir errores y completar valores nulos.

La columna "Consumer disputed?" ha sido limpiada en base a "Company response": si la respuesta de la empresa indica que el caso está cerrado y el valor es nulo, se ha asumido que no hubo disputa, por lo que se ha rellenado como "No". Si el caso sigue "In progress", se ha marcado como "Pending".

Se procedió a agrupar y limpiar la categoría "Company" debido al elevado número de empresas; se han seleccionado aquellas que tienen una frecuencia mayor de 50 y el resto se han agrupado en una categoría "cajón de sastre".

04-08-2025: Se imputaron valores faltantes en la columna "Issue" siempre que existía una correspondencia clara con "Product" y "Sub-product", mejorando así la coherencia categórica del dataset. La misma estrategia se aplicó a "Sub-product" cuando fue posible.

Dado el escaso interés analítico y el alto porcentaje de valores nulos (más del 60%), la columna "Sub-issue" fue eliminada por aportar información redundante y marginal.

Se generó una nueva variable numérica llamada "Difference in days", que representa el tiempo en días entre la recepción y la respuesta de una queja.

06-08-2025: Realización de gráficos de frecuencia y de tipo pie chart para analizar la distribución de quejas por "Product". También se generaron visualizaciones jerárquicas (gráficos sunburst) para explorar la relación entre "Product", "Sub-product" e "Issue".

Construcción de gráficos de barras para visualizar las empresas con mayor volumen de quejas y se realizó un análisis detallado de sus patrones de respuesta.

La variable "Difference in days" fue evaluada estadísticamente, incluyendo medidas de tendencia central, dispersión, asimetría, curtosis y normalidad (prueba de Shapiro-Wilk), complementado con su respectiva visualización.

Para los tiempos de respuesta por tipo de producto, se emplearon diagramas de caja (boxplots), lo que permitió identificar productos con mayor variabilidad y presencia de outliers.

Se generó un mapa coroplético para visualizar la distribución geográfica del volumen de quejas por estado.

Se analizaron variables asociadas a la satisfacción y resolución del cliente ("Timely response?", "Consumer disputed?" y "Company response") mediante gráficos circulares y un mapa de calor cruzado.

Se elaboró un gráfico por tipo de respuesta para las 10 empresas con más quejas, incluyendo la categoría "Low Count Companies".

07-08-2025: Empezamos con los modelos, primero un modelo de clasificación para ver si el consumidor disputará la queja.

Primero hemos hecho un modelo de Random Forest. Como teníamos duda sobre "Sub-product", se han diseñado dos conjuntos: uno con dicha columna y otro sin ella. El modelo parece tener problemas para clasificar correctamente "Yes", así que se ha probado con SMOTE para ver si el overfitting de la clase podía mejorar el modelo. No parece ser el caso; además, dado que las mejores métricas se encuentran si se incluye la clase "Sub-product", vamos a optar por conservarla.

Procedemos a hiperparametrizar con GridSearchCV; de no obtener buenos resultados probaremos con XGBoost.

10-08-2025: Tras probar con XGBoost, vemos que a pesar de la hiperparametrización y el ajuste de los pesos, el modelo tiene dificultad para determinar la clase "Yes", lo que supone un problema. Tras intentar mejorarlo, la mejora obtenida es marginal, por lo que nos quedamos con el último modelo obtenido, con los parámetros dados por el GridSearch y el umbral de 0.28.

18-08-2025: Para nuestro segundo modelo vamos a intentar predecir si la respuesta va a ser a tiempo o no. Para este modelo de clasificación binomial se van a desarrollar modelos empleando Random Forest y XGBoost de la misma forma que el modelo anterior.

Tras realizar la función del pipeline, el modelo muestra un rendimiento muy sólido en términos generales; la clase mayoritaria es predicha casi a la perfección, sin embargo, para la clase minoritaria presenta un desempeño desigual. Con el fin de equilibrar dichos desempeños vamos a ajustar el modelo mediante SMOTE para evaluar de nuevo su rendimiento.

Gracias a SMOTE, el modelo fue capaz de mejorar el equilibrio en la detección de ambas clases.

19-08-2025: Se ha completado el tercer modelo. Tras los malos resultados obtenidos al intentar crear un modelo de regresión logística, se optó por convertir el problema a un modelo de clasificación con 4 clases. Tras probar Random Forest, Random Forest + SMOTE, XGBoost y XGBoost con ajuste de pesos, el mejor modelo resultó ser XGBoost con Randomized Search.

22-08-2025: Se ha procedido a la realización del dashboard en PowerBI de la estadística descriptiva para la presentación.

26-08-2025: Se han limpiado los datos de nuevo para intentar solventar algún valor faltante y mejorar las métricas obtenidas; sin embargo, debido a la calidad de los datos el margen de mejora ha sido mínimo. Se ha reducido el último modelo a un modelo de clasificación de 3 clases con el fin de mejorar el rendimiento; algo se ha conseguido. Se han redactado los notebooks incluyendo nuevas gráficas comparativas de los modelos.

27-08-2025: Se va a proceder a generar los archivos src y pkl para desarrollar posteriormente la app con la interfaz predictiva. Se ha creado la app por Streamlit y funciona correctamente.



