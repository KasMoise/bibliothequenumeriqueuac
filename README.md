# 📚 Système de Recommandation — Bibliothèque Numérique UAC

> **Université de l'Assomption au Congo (UAC) — Butembo, Nord-Kivu, RDC**
> Cadre doctoral BAEL (Behavior-Aware Explainability Loop)

---

## 🧠 Architecture

```
uac_scraper.ipynb          ← Collecte les données du site UAC
        ↓
data/uac_bibliotheque.csv  ← Dataset (~3 659 ouvrages)
        ↓
uac_recommender.ipynb      ← Entraîne le modèle TF-IDF + cosine
        ↓
artifacts/                 ← Modèle sérialisé (pickle)
        ↓
uac_app.py                 ← Interface Streamlit
```

---

## ⚙️ Installation

```bash
pip install -r uac_requirements.txt
```

---

## 🚀 Utilisation

### Étape 1 — Collecter les données
```bash
jupyter notebook uac_scraper.ipynb
# Exécuter toutes les cellules → génère data/uac_bibliotheque.csv
```

### Étape 2 — Entraîner le modèle
```bash
jupyter notebook uac_recommender.ipynb
# Exécuter toutes les cellules → génère artifacts/
```

### Étape 3 — Lancer l'application
```bash
streamlit run uac_app.py
```

---

## 📁 Structure des fichiers

```
├── uac_scraper.ipynb          # Scraper du site UAC
├── uac_recommender.ipynb      # Modèle TF-IDF + cosine similarity
├── uac_app.py                 # Application Streamlit
├── uac_requirements.txt       # Dépendances Python
├── data/
│   └── uac_bibliotheque.csv   # Dataset collecté (généré)
└── artifacts/
    ├── uac_tfidf.pkl           # Vectoriseur TF-IDF
    ├── uac_tfidf_matrix.pkl    # Matrice TF-IDF
    ├── uac_cosine_sim.pkl      # Matrice de similarité cosinus
    └── uac_df.pkl              # DataFrame des ouvrages
```

---

## 🔬 Méthode

| Aspect | Choix | Justification |
|---|---|---|
| Algorithme | Content-Based Filtering | Pas de ratings utilisateurs disponibles |
| Vectorisation | TF-IDF (1-2 grammes) | Capture les termes spécifiques du corpus |
| Similarité | Cosinus | Invariante à la longueur des textes |
| Pondération | Titre ×3, Auteur ×2, Domaine ×2 | Le titre est le signal le plus fort |

---

## 🖥️ Fonctionnalités de l'application

- 🔍 Recherche par mot-clé avec autocomplétion
- ✨ Top-N recommandations configurables (3 à 10)
- 📂 Filtre par domaine académique
- 📊 Score de similarité affiché pour chaque recommandation
- 🖼️ Couvertures des ouvrages
- 🔗 Lien vers la fiche complète sur le site UAC
- ⬇️ Export CSV du catalogue filtré
