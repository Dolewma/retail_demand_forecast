# Retail Demand Forecasting App

Dieses Projekt beinhaltet die Entwicklung einer Anwendung zur Prognose der Verkaufszahlen im Einzelhandel. Die zugrunde liegenden Modelle wurden im Rahmen eines Machine Learning-Projekts erstellt, trainiert und in einer interaktiven Streamlit-Anwendung eingebunden.

Projektziel
Ziel war es, die Nachfrage nach Produkten in verschiedenen Filialen vorherzusagen. Dies unterstützt Geschäftsentscheidungen im Bereich Lagerhaltung, Beschaffung und Planung.

Die Webanwendung ermöglicht es, Store, Item und ein Datum (im Zeitraum Januar bis März 2014) auszuwählen. Darauf basierend wird eine Absatzprognose mit einem von drei verfügbaren Modellen erstellt.

Explorative Datenanalyse (Week 1)
In der EDA wurden u. a. folgende Aspekte untersucht:

Saisonale und wöchentliche Muster in den Verkaufszahlen

Einfluss von Feiertagen und Promotionen

Entwicklung des Ölpreises im Zeitverlauf

Unterschiede im Kaufverhalten zwischen Stores und Produktfamilien

Verwendete Bibliotheken: pandas, matplotlib, seaborn, plotly

Detailliert dokumentiert in: course_project_week_1.ipynb

Modellierung & Optimierung (Week 2 & 3)
Im Fokus standen drei Modellansätze:

XGBoost
Gradient Boosted Trees. Liefert tendenziell höhere Prognosen (aggressiver Charakter).

LSTM (Long Short-Term Memory)
Deep Learning Modell, speziell zur Modellierung von Zeitreihen. Liefert konservativere, stabilere Prognosen.

Hybrid-Modell (XGBoost + LSTM)
Kombination beider Modelle. Die Gewichtung wird dynamisch angepasst, abhängig von der Differenz beider Einzelprognosen.

Feature Engineering beinhaltete unter anderem:

Lag-Features (1, 7, 14, 30 Tage)

Durchschnittswerte

Zeitmerkmale wie Wochentag, Monat, Feiertag

Ölpreis-bezogene Merkmale

Trainings- und Optimierungsprozesse dokumentiert in:
course_project_week_2_and_3_.ipynb

Streamlit App
Die entwickelte Webanwendung bietet folgende Funktionen:

Auswahl eines Prognosemodells (XGBoost, LSTM oder Hybrid)

Eingabe von Store, Item und Datum (beschränkt auf Jan–März 2014)

Automatische Vorbereitung der Daten (inkl. Lag-Berechnung, Feature-Erweiterung)

Vorhersage der Verkaufsmenge mit Fehlerbehandlung

retail_demand_forecast/
├── app/
│   ├── main.py
│   ├── config.py
│   
├── model/
│   ├── model_utils.py
│   
├── data/
│   ├── data_utils.py
│   └── __init__.py
├── models/                    # Modell-Dateien (.pkl, .h5)
├── requirements.txt
├── README.md
└── course_project_week_*.ipynb


Fazit
Das Projekt verbindet klassische ML-Verfahren (XGBoost) mit modernen Deep-Learning-Techniken (LSTM) in einer produktionsnahen Streamlit-Oberfläche. Durch intelligentes Feature Engineering und datengetriebene Kombination der Modelle entsteht eine realistische Absatzprognose, die sich in der Praxis vielfältig einsetzen lässt.


