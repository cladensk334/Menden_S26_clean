# Menden_S26_clean
**Ziel dieses Projekts**
Ziel dieses Projekts ist die Analyse von Wetterzeitreihen der DWD-Wetterstation Würzburg (Stations-ID: 05705) mithilfe statistischer Zeitreihenmodelle. Hier ist wichtig, dass nicht Würzburg/Main oder Physikalisches Institut als Ort genommen werden, da diese Stationen seite Jahrzehnten nicht mehr existieren (keine aktuelle Datenlage) oder zb in dem Datensatz der Luftfeutigkeit nicht repräsentiert sind. Die Daten werden automatisiert über den CDC Open Data Server des Deutschen Wetterdienstes bezogen und umfassen tägliche Messungen von Windgeschwindigkeit, Luftdruck und Temperatur.
Im Rahmen der univariaten Zeitreihenanalyse wird für jede der drei Zeitreihen mithilfe der Box-Jenkins-Methode ein geeignetes ARIMA-Modell identifiziert, geschätzt und validiert. Dabei werden Stationaritätstests durchgeführt, ACF und PACF analysiert sowie Residualdiagnostiken vorgenommen. Abschließend werden Prognosen für die nächsten zehn Perioden inklusive Konfidenzintervallen berechnet.
Im Rahmen der multivariaten Analyse wird eine automatisierte Pipeline entwickelt, die verschiedene Modelle über alle drei Zeitreihen evaluiert, das jeweils beste Modell anhand geeigneter Metriken auswählt und Prognosen für den gesamten Datensatz erstellt. Die Daten wurden von der offiziellen DWD-Website bezogen: https://www.dwd.de/DE/leistungen/cdc/cdc_ueberblick-klimadaten.html
Präsentation: https://canva.link/5pkqlfakx89zx7t

## Repo Überblick 
**Welche Dateien gibt es?**
1. Diese "ReadMe" Datei in der die wichtigsten Sachen erklärt sind
2. Die "src" Datei, in welcher notwendige Python Funktionen aufzählt sind (diese wird je nach fortschritt des Projektes angepasst
3. Die "aufgabenstellung", in welcher unsere Eckdaten zum Projekt stehen (halt die Aufgabenstellung)
4. Den data Ordner, in welchem die Rohdaten und bereinigten Daten liegen
5. Die "to-do" liste, wo steht was noch gemacht werden muss, sodass dies nicht vergessen wird 

**Wie klont man dieses Respiratory lokal ?**
[] bei diesen klammer ist nur der inhalt zu kopieren, nicht die Klammern 
1. Nutze das Terminal auf deinem Laptop
2. Nutzte [git --version] um zu schauen, dass Github installiert ist 
3. Falls es nicht installiert ist: git-scm.com
4. Tippe dann ein [git clone https://github.com/cladensk334/Menden_S26_clean.git ~/Menden_S26_clean]

**Branches:**
main (mit Ordnerstrukur), dev ("test" branch um nicht das "orginal" im main zu schrotten), datenbereinigung, datenimport?, datenvisualiserung, deskriptive analyse, model
Hierraus können weitere Branches je nach Temperatur/Wind/Wetter entstehen

**Regeln:**
1. Jeder Uplaod wird direkt in den richtigen Ordner im dev, dann main (ausnahme am Anfang) hochgeladen 
2. Whatsapp Gruppe zur Kommunikation plus 1-2 Meetings pro Woche nach Aufgabenverteilung (Beispiel Kommunikation: struktur veränderung durch neuer Branch/Ordner, Information dieht hier zur Fehler vermeidung)
3. Vor einem upload muss die aktuelle Version von GitHub runtergeladen werden und genutzt/ angepasst werden um Konfikte mit anderen updates zu vermeiden
4. Code (oder andere nicht eindeutige Dateien) sollten mit Kommentar hochgeladen werden um verwirrung zu vermeiden


## Aufgaben
**Was macht ein professionelles Repo aus?:**
- So aufgebaut, dass jemand Fremdes es klonen, verstehen und sofort ausführen kann     ohne Rückfragen zu haben
- Aussagekräftiges README
- Saubere Ordnerstruktur
- .gitignore damit keine unnötigen Dateien getrackt werden (können wir hier nicht      machen)
- Requirements.txt damit die Umgebung reproduzierbar ist
- Commits die eine nachvollziehbare Geschichte des Projekts erzählen
- Hier findet man weitere Ideen: https://github.com/orgs/community/discussions/88715

**Termin Übersicht:**
| Termin | Beschreibung |
|------|-------------|
| **Termin 1** | **Professionelles Git Repository** — Ordnerstruktur, Branching-Strategie, Commit-Konventionen, README, `.gitignore` und `requirements.txt` |
| **Termin 2** | **Univariate Zeitreihenanalyse (ARIMA)** — Stationaritätstests (ADF & KPSS), ACF/PACF Analyse, Modellselektion nach Box-Jenkins, Residualdiagnostik, Prognose für 10 Perioden |
| **Termin 3** | **Multivariate Zeitreihenanalyse** — Automatisierte Pipeline über alle Zeitreihen, Evaluationsmetriken (RMSE, MAE, MAPE), Auswahl des besten Modells, Prognose für alle Zeitreihen |

**Aufgabenverteilung:** 
| Teammitglied | Zeitreihe | Parameter | Einheit |
|--------------|-----------|-----------|---------|
| **Kenia** | Wind | Windgeschwindigkeit (`FM`) | m/s |
| **Jonas** | Luftdruck | Luftdruck (`PM`) | hPa |
| **Clara** | Temperatur | Tagesmitteltemperatur (`TMK`) | °C |
