# SentimentumAI 🐦
### Análisis de Sentimientos de Odio en Twitter/X

Plataforma inteligente de auditoría social que clasifica automáticamente el discurso de odio en publicaciones de Twitter/X mediante Procesamiento de Lenguaje Natural (NLP) y Machine Learning supervisado.

---

## Contexto

En el período 2020–2026, la conversación pública digital ha experimentado niveles críticos de polarización ideológica y social. Organizaciones como las Naciones Unidas señalan que la propagación del discurso de odio en línea constituye una de las mayores amenazas contemporáneas para la cohesión social, actuando como detonante de crímenes en el plano físico.

SentimentumAI nace como respuesta a la falta de herramientas accesibles de moderación automatizada, ofreciendo un motor analítico capaz de clasificar flujos de texto y proveer inteligencia útil para tomadores de decisiones, activistas y defensores de derechos humanos.

---

## Propuesta de Valor

> **Para** organizaciones de derechos humanos, investigadores sociales y moderadores de comunidades virtuales **que** enfrentan una creciente polarización digital, **SentimentumAI** es una plataforma inteligente de auditoría social que extrae publicaciones de Twitter/X, aplica modelos avanzados de NLP para detectar discurso de odio y evalúa su impacto con métricas de clasificación claras. **A diferencia de** herramientas tradicionales basadas en conteos manuales, nuestro producto actúa como un escudo analítico digital que aporta una evaluación objetiva y cuantitativa de la polarización en el entorno digital.

---

## Dataset

- **Fuente:** Kaggle — Twitter Hate Speech Dataset
- **Archivo:** `twitter_train.csv`
- **Registros:** 31,192 tweets en inglés
- **Columnas:** `id`, `label`, `tweet`
- **Distribución:** 92.99% neutro (label=0) / 7.01% odio (label=1)
- **Nota:** El desbalance significativo (ratio 13:1) requiere estrategia de balanceo antes del entrenamiento.

---

## Tecnologías

- Python 3.x
- Pandas / NumPy
- Scikit-learn (TF-IDF, clasificadores supervisados)
- NLTK (tokenización, stopwords, stemming)
- Matplotlib
- WordCloud

---

## Arquitectura del Pipeline

```
Dataset CSV
    ↓
HU-01: Importación y exploración
    ↓
HU-02: Limpieza y normalización de texto
    ↓
HU-03: Análisis de distribución de clases + balanceo
    ↓
HU-04: Vectorización TF-IDF → Matriz X / Vector y
    ↓
HU-05: WordClouds y análisis léxico
    ↓
Sprint 2: Entrenamiento → Evaluación → Dashboard
```

---

## Estructura del Proyecto

```
SentimentumAI/
│
├── twitter_train.csv             # Dataset original
├── Sprint1_Documentacion.md      # Documentación técnica Sprint 1
│
├── notebooks/
│   ├── HU01_exploracion.ipynb
│   ├── HU02_limpieza.ipynb
│   ├── HU03_distribucion.ipynb
│   ├── HU04_vectorizacion.ipynb
│   └── HU05_wordclouds.ipynb
│
└── outputs/
    ├── distribucion_clases.png
    ├── wordcloud_odio.png
    └── wordcloud_neutro.png
```

---

## Roadmap

| Sprint | Semanas | Bloques | Entregable |
|--------|---------|---------|------------|
| Sprint 1 | 1–2 (1–14 jun) | A: Preparación de datos · B: Procesamiento NLP | Pipeline de preprocesamiento + matriz limpia y balanceada |
| Sprint 2 | 3–4 (15–28 jun) | C: Modelado predictivo · D: Análisis visual y documentación | MVP: modelo entrenado + dashboard analítico |

**Hitos de validación:**
- Hito 1 (fin semana 2): Pipeline de preprocesamiento funcional
- Hito 2 (fin semana 3): Modelo entrenado con precisión superior a baseline
- Hito Final (fin semana 4): Entrega del MVP bajo metodología POL

---

## Metodología

El proyecto aplica el marco ágil **Scrum** en combinación con la metodología académica **Project Oriented Learning (POL)**, gestionado visualmente en **Trello** con un tablero de 6 listas:

`Product Backlog → Sprint Backlog → In Progress → AI Raw Code → Adapted Code → Done`

La priorización del Product Backlog se realizó mediante la combinación de **MoSCoW** y análisis **Impacto vs Esfuerzo**, concentrando el desarrollo del MVP en las funcionalidades de mayor valor con menor costo técnico.

---

## Equipo

| Nombre | Rol |
|--------|-----|
| Héctor Ramón Jiménez Esquivel | Product Owner |
| Nikesha Sonalhy Mercado Escoto | Scrum Master |
| Carlo Eugenio Leviaguirre Chavez | Dev Team |
| Denisse Martínez Ruíz | Dev Team |

---

*Materia: Administración del Desarrollo de Software · Tecnológico de Monterrey*
