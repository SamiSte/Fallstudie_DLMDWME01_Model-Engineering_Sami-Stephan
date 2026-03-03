# Fallstudie\_DLMDWME01\_Model Engineering\_Sami Stephan



\*\*Optimierung von Zahlungstransaktionen durch dynamisches PSP-Routing\*\*



Dieses Repository enthält den Python-Code und die Auswertungen für die Fallstudie im Modul "Model Engineering" (IU Internationale Hochschule). 



Ziel des Projekts ist die Entwicklung eines Machine-Learning-Systems, das für einen Online-Shop bei jedem Checkout dynamisch den optimalen Zahlungsdienstleister (Payment Service Provider, PSP) auswählt. Das Modell bewertet dabei nicht nur die technische Erfolgswahrscheinlichkeit, sondern maximiert den \*\*Business Value\*\* unter Berücksichtigung von Transaktionsgebühren und Opportunitätskosten bei Kaufabbrüchen.



\## Methodik

\* \*\*Vorgehensmodell:\*\* CRISP-DM (Cross-Industry Standard Process for Data Mining)

\* \*\*Algorithmus:\*\* Kalibrierte Entscheidungsbäume (CART - Classification and Regression Trees) pro PSP

\* \*\*Evaluierung:\*\* Sequenzielle Replay-Simulation zur Ermittlung des monetären Mehrgewinns (Profit-Maximierung) gegenüber einer historischen Baseline.



\## Repository-Struktur



Das Projekt folgt einer standardisierten Data-Science-Struktur:



```text

├── data/

│   ├── raw/                 <- Originaler Rohdatensatz (muss manuell hinzugefügt werden)

│   └── processed/           <- Bereinigte Daten inkl. Feature Engineering

├── models/                  <- Serialisierte, trainierte Modelle (.pkl)

├── reports/                 <- Automatisch generierte Auswertungen

│   ├── figures/             <- EDA-Plots und Entscheidungsbaum-Visualisierungen

│   ├── logs/                <- Konsolen-Protokolle der Modelläufe

│   └── results/             <- Tabellarische CSV-Ergebnisse der Simulation

├── src/

│   └── main\_cart\_analysis.py <- Das zentrale ausführbare Python-Skript

├── requirements.txt         <- Benötigte Python-Bibliotheken

└── README.md                <- Diese Projektbeschreibung

