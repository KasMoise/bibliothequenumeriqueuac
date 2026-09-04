"""
============================================================
UAC Bibliothèque Numérique — Application Streamlit v6
============================================================
Logique d'accès v6 :
  - Accès public  : Recommandation, Catalogue, À propos
  - Bouton "Se connecter" dans la sidebar
  - Après connexion admin : onglets ➕ Nouvel ouvrage + 🛠️ Admin
  - Seuls les admins ont des identifiants (dans uac_config.yaml)
============================================================
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import hashlib
import yaml
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# ── Chemins ──────────────────────────────────────────────
ARTIFACTS_DIR = 'artifacts'
FEEDBACK_CSV  = 'data/feedback.csv'
NOUVEAUX_CSV  = 'data/nouveaux_ouvrages.csv'
CONFIG_FILE   = 'uac_config.yaml'
COSINE_PKL    = f'{ARTIFACTS_DIR}/uac_cosine_sim.pkl'
DF_PKL        = f'{ARTIFACTS_DIR}/uac_df.pkl'
TFIDF_PKL     = f'{ARTIFACTS_DIR}/uac_tfidf.pkl'
TFIDF_MAT_PKL = f'{ARTIFACTS_DIR}/uac_tfidf_matrix.pkl'

os.makedirs('data', exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Configuration de la page ─────────────────────────────
st.set_page_config(
    page_title="UAC — Bibliothèque Numérique",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1a3a5c; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8f9fa; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #1a3a5c; margin-bottom: 0.8rem;
    }
    .card-new {
        background-color: #fff8e1; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #f59e0b; margin-bottom: 0.8rem;
    }
    .score-badge {
        background-color: #1a3a5c; color: white;
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.85rem; font-weight: 600;
    }
    .domaine-tag {
        background-color: #e8f0fe; color: #1a3a5c;
        border-radius: 8px; padding: 2px 8px; font-size: 0.8rem;
    }
    .new-badge {
        background-color: #f59e0b; color: white;
        border-radius: 8px; padding: 2px 8px; font-size: 0.8rem;
    }
    .admin-badge {
        background-color: #dc2626; color: white;
        border-radius: 8px; padding: 3px 10px; font-size: 0.85rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# CONFIGURATION & AUTHENTIFICATION
# ════════════════════════════════════════════════════════

def hasher(mdp: str) -> str:
    return hashlib.sha256(mdp.encode()).hexdigest()


def charger_config() -> dict:
    """Charge uac_config.yaml. Crée une config admin par défaut si absente."""
    if not os.path.exists(CONFIG_FILE):
        config_defaut = {
            'admins': {
                'admin': {
                    'nom': 'Administrateur UAC',
                    'role':'admin',
                    'password_hash': hasher('admin123'),
                },
                'libraire1': {
                    'nom': 'Bibliothécaire UAC',
                    'role':'libraire',
                    'password_hash': hasher('libraire123'),
                },
            }
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config_defaut, f, allow_unicode=True)
        return config_defaut

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def verifier_admin(username: str, password: str, config: dict):
    """
    Vérifie les credentials admin.
    Retourne le nom si OK, sinon None.
    """
    admins = config.get('admins', {})
    if username in admins:
        if admins[username]['password_hash'] == hasher(password):
            return admins[username]['nom']
    return None


def est_connecte() -> bool:
    return st.session_state.get('connecte', False)


# ════════════════════════════════════════════════════════
# CHARGEMENT DES ARTIFACTS
# ════════════════════════════════════════════════════════

@st.cache_resource
def charger_artifacts():
    try:
        cosine_sim   = pickle.load(open(COSINE_PKL,    'rb'))
        df           = pickle.load(open(DF_PKL,         'rb'))
        tfidf        = pickle.load(open(TFIDF_PKL,      'rb'))
        tfidf_matrix = pickle.load(open(TFIDF_MAT_PKL,  'rb'))

        df['titre']       = df['titre'].fillna('').str.lower().str.strip()
        df['auteur']      = df['auteur'].fillna('').str.lower().str.strip()
        df['domaine']     = df['domaine'].fillna('').str.lower().str.strip()
        df['description'] = df['description'].fillna('')
        for col in ['categorie', 'edition', 'annee', 'image_url', 'url_detail', 'source']:
            if col not in df.columns:
                df[col] = ''
        return cosine_sim, df, tfidf, tfidf_matrix
    except FileNotFoundError:
        return None, None, None, None


# ════════════════════════════════════════════════════════
# FEEDBACK
# ════════════════════════════════════════════════════════

def sauvegarder_feedback(titre_query, titre_reco, score, vote):
    ligne = {
        'date':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'titre_query':  titre_query,
        'titre_reco':   titre_reco,
        'score_cosine': score,
        'vote':         vote,
    }
    if os.path.exists(FEEDBACK_CSV):
        fb = pd.read_csv(FEEDBACK_CSV)
        fb = pd.concat([fb, pd.DataFrame([ligne])], ignore_index=True)
    else:
        fb = pd.DataFrame([ligne])
    fb.to_csv(FEEDBACK_CSV, index=False, encoding='utf-8-sig')


def charger_feedback():
    if os.path.exists(FEEDBACK_CSV):
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame(columns=['date', 'titre_query', 'titre_reco',
                                  'score_cosine', 'vote'])


# ════════════════════════════════════════════════════════
# COLD START B — AJOUT INCRÉMENTAL
# ════════════════════════════════════════════════════════

def construire_contenu(titre, auteur, domaine, description, categorie=''):
    t    = str(titre).lower().strip()
    a    = str(auteur).lower().strip()
    d    = str(domaine).lower().strip()
    desc = str(description).lower().strip()
    cat  = str(categorie).lower().strip()
    return f"{t} {t} {t} {a} {a} {d} {d} {desc} {cat}".strip()


def ajouter_ouvrage_incrementalement(nouvel_ouvrage: dict):
    from scipy.sparse import vstack as sp_vstack

    df_live  = st.session_state.df_live
    cos_live = st.session_state.cosine_live
    mat_live = st.session_state.tfidf_mat_live
    tfidf    = st.session_state.tfidf_obj

    if tfidf is None:
        raise ValueError("TF-IDF non chargé.")

    contenu  = construire_contenu(
        nouvel_ouvrage.get('titre', ''),
        nouvel_ouvrage.get('auteur', ''),
        nouvel_ouvrage.get('domaine', ''),
        nouvel_ouvrage.get('description', ''),
        nouvel_ouvrage.get('categorie', ''),
    )
    new_vec      = tfidf.transform([contenu])
    sims_new     = cosine_similarity(new_vec, mat_live).flatten()
    col_new      = sims_new.reshape(-1, 1)
    cos_updated  = np.hstack([cos_live, col_new])
    row_new      = np.append(sims_new, 1.0).reshape(1, -1)
    cos_updated  = np.vstack([cos_updated, row_new])
    mat_updated  = sp_vstack([mat_live, new_vec])

    nouveau_row = {
        'titre':       str(nouvel_ouvrage.get('titre', '')).lower().strip(),
        'auteur':      str(nouvel_ouvrage.get('auteur', '')).lower().strip(),
        'domaine':     str(nouvel_ouvrage.get('domaine', '')).lower().strip(),
        'description': str(nouvel_ouvrage.get('description', '')),
        'categorie':   str(nouvel_ouvrage.get('categorie', '')),
        'isbn':        str(nouvel_ouvrage.get('isbn', '')),
        'annee':       str(nouvel_ouvrage.get('annee', '')),
        'image_url':   str(nouvel_ouvrage.get('image_url', '')),
        'url_detail':  str(nouvel_ouvrage.get('url_detail', '')),
        'contenu':     contenu,
        'source':      'ajout_manuel',
        'ajoute_par':  st.session_state.get('username', ''),
        'date_ajout':  datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    df_updated = pd.concat([df_live, pd.DataFrame([nouveau_row])], ignore_index=True)

    pickle.dump(cos_updated, open(COSINE_PKL,    'wb'))
    pickle.dump(df_updated,  open(DF_PKL,         'wb'))
    pickle.dump(mat_updated, open(TFIDF_MAT_PKL,  'wb'))

    df_log = pd.DataFrame([{**nouvel_ouvrage,
                             'ajoute_par': st.session_state.get('username', ''),
                             'date_ajout': nouveau_row['date_ajout']}])
    if os.path.exists(NOUVEAUX_CSV):
        existing = pd.read_csv(NOUVEAUX_CSV)
        df_log = pd.concat([existing, df_log], ignore_index=True)
    df_log.to_csv(NOUVEAUX_CSV, index=False, encoding='utf-8-sig')

    return df_updated, cos_updated, mat_updated


# ════════════════════════════════════════════════════════
# COLD START A — RE-ENTRAÎNEMENT COMPLET
# ════════════════════════════════════════════════════════

def re_entrainer_modele():
    import subprocess
    with st.spinner('🔄 Re-scraping en cours...'):
        r = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook',
             '--execute', 'uac_scraper.ipynb'],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            st.error(f'Erreur scraping : {r.stderr[:300]}')
            return False
    with st.spinner('🧠 Re-entraînement du modèle...'):
        r = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook',
             '--execute', 'uac_recommender.ipynb'],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            st.error(f'Erreur entraînement : {r.stderr[:300]}')
            return False
    st.cache_resource.clear()
    return True


# ════════════════════════════════════════════════════════
# RECOMMANDATION
# ════════════════════════════════════════════════════════

def get_suggestions(query: str, df_local, n: int = 10) -> list:
    if df_local is None or query.strip() == '':
        return []
    q = query.lower().strip()
    matches = df_local[
        df_local['titre'].str.contains(q, na=False, regex=False)
    ]['titre'].tolist()
    return [t.title() for t in matches[:n]]


def recommander(titre_display, df_local, cosine_local, n=5, filtre_domaine=None):
    titre_norm = titre_display.lower().strip()
    indices    = df_local[df_local['titre'] == titre_norm].index
    if len(indices) == 0:
        return pd.DataFrame()

    idx    = indices[0]
    scores = sorted(enumerate(cosine_local[idx]), key=lambda x: x[1], reverse=True)
    filtre = filtre_domaine.lower().strip() \
             if filtre_domaine and filtre_domaine != 'Tous' else None

    resultats = []
    rang = 1
    for i, score in scores:
        if i == idx:
            continue
        row = df_local.iloc[i]
        if filtre and row['domaine'] != filtre:
            continue
        desc = str(row['description']) if row['description'] else ''
        resultats.append({
            'rang':        rang,
            'titre':       row['titre'].title(),
            'auteur':      row['auteur'].title() if row['auteur'] else 'Inconnu',
            'domaine':     row['domaine'].title(),
            'score':       round(float(score), 4),
            'image_url':   str(row.get('image_url', '')),
            'url_detail':  str(row.get('url_detail', '')),
            'description': desc[:120] + '...' if len(desc) > 120 else desc,
            'nouveau':     str(row.get('source', '')) == 'ajout_manuel',
        })
        rang += 1
        if rang > n:
            break
    return pd.DataFrame(resultats)


# ════════════════════════════════════════════════════════
# APPLICATION PRINCIPALE
# ════════════════════════════════════════════════════════

def main():
    config = charger_config()

    # Initialisation des données
    if 'df_live' not in st.session_state:
        cosine_sim, df, tfidf, tfidf_matrix = charger_artifacts()
        st.session_state.df_live        = df
        st.session_state.cosine_live    = cosine_sim
        st.session_state.tfidf_mat_live = tfidf_matrix
        st.session_state.tfidf_obj      = tfidf

    if 'feedback_pending' not in st.session_state:
        st.session_state.feedback_pending = {}

    if 'show_login' not in st.session_state:
        st.session_state.show_login = False

    df_live     = st.session_state.df_live
    cosine_live = st.session_state.cosine_live
    admin_ok    = est_connecte()

    # ── En-tête ───────────────────────────────────────────
    st.markdown('<div class="main-title">📚 Bibliothèque Numérique UAC</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Université de l\'Assomption au Congo — Butembo, RDC</div>',
                unsafe_allow_html=True)
    st.markdown('---')

    if df_live is None:
        st.error('⚠️ Artifacts introuvables. Exécute d\'abord `uac_recommender.ipynb`.')
        return

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        # Bloc connexion / déconnexion
        if admin_ok:
            nom = st.session_state.get('nom', '')
            st.markdown(f'<span class="admin-badge">🔑 ADMIN</span>', unsafe_allow_html=True)
            st.markdown(f'👤 **{nom}**')
            if st.button('🚪 Se déconnecter', use_container_width=True):
                for k in ['connecte', 'username', 'nom',
                          'show_login', 'login_erreur',
                          'derniers_recos', 'derniere_query',
                          'feedback_pending']:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.markdown('👤 **Accès public**')
            if st.button('🔑 Se connecter', use_container_width=True):
                st.session_state.show_login = True
                st.rerun()

        st.markdown('---')
        st.markdown('### ⚙️ Paramètres')
        n_recos = st.slider('Recommandations', 3, 10, 5)

        domaines_bruts   = sorted(df_live['domaine'].dropna().unique().tolist())
        domaines_display = ['Tous'] + [d.title() for d in domaines_bruts]
        domaine_choisi   = st.selectbox('Filtrer par domaine', domaines_display)

        st.markdown('---')
        st.markdown('### 📊 Statistiques')
        n_manuels = int(
            (df_live['source'] == 'ajout_manuel').sum()
            if 'source' in df_live.columns else 0
        )
        st.metric('Ouvrages indexés',     len(df_live))
        st.metric('Ajoutés manuellement', n_manuels)
        st.metric('Domaines',             df_live['domaine'].nunique())

    # ════════════════════════════════════════════════════
    # MODAL DE CONNEXION (affiché par-dessus le contenu)
    # ════════════════════════════════════════════════════
    if st.session_state.show_login and not admin_ok:
        st.markdown('### 🔐 Connexion administrateur')
        st.info('Seuls les administrateurs disposent d\'identifiants de connexion.')

        with st.form('login_form'):
            username = st.text_input('Identifiant', placeholder='Votre identifiant admin')
            password = st.text_input('Mot de passe', type='password')
            c1, c2   = st.columns(2)
            with c1:
                soumis = st.form_submit_button('✅ Se connecter', use_container_width=True)
            with c2:
                annule = st.form_submit_button('✖️ Annuler', use_container_width=True)

        if soumis:
            nom = verifier_admin(username, password, config)
            if nom:
                st.session_state['connecte']   = True
                st.session_state['username']   = username
                st.session_state['nom']        = nom
                st.session_state['show_login'] = False
                st.success(f'✅ Bienvenue, {nom} !')
                st.rerun()
            else:
                st.error('❌ Identifiant ou mot de passe incorrect.')

        if annule:
            st.session_state.show_login = False
            st.rerun()

        st.markdown('---')
        return   # Ne pas afficher les onglets pendant la connexion

    # ════════════════════════════════════════════════════
    # ONGLETS — publics + admin si connecté
    # ════════════════════════════════════════════════════
    onglets_noms = ['🔍 Recommandation', '📂 Catalogue', 'ℹ️ À propos']
    if admin_ok:
        onglets_noms += ['➕ Nouvel ouvrage', '🛠️ Admin']

    onglets = st.tabs(onglets_noms)

    # ════════════════════════════════════════════════════
    # ONGLET 1 — RECOMMANDATION (public)
    # ════════════════════════════════════════════════════
    with onglets[0]:
        st.subheader('Trouver des ouvrages similaires')

        recherche   = st.text_input('🔎 Taper le titre d\'un ouvrage',
                                    placeholder='Ex : deep learning...')
        suggestions = get_suggestions(recherche, df_live) if recherche else []

        if suggestions:
            titre_sel = st.selectbox('Sélectionner un ouvrage', suggestions)
        elif recherche:
            st.warning('Aucun ouvrage trouvé. Essaie un autre mot-clé.')
            titre_sel = None
        else:
            tous_titres = sorted([t.title() for t in df_live['titre'].dropna()])
            titre_sel   = st.selectbox('Ou choisir dans la liste complète', tous_titres)

        if st.button('✨ Voir les recommandations', use_container_width=True):
            if titre_sel:
                with st.spinner('Recherche en cours...'):
                    recos = recommander(titre_sel, df_live, cosine_live,
                                        n_recos, domaine_choisi)
                if recos.empty:
                    st.warning('Aucune recommandation. Essaie sans filtre de domaine.')
                else:
                    st.session_state['derniere_query'] = titre_sel
                    st.session_state['derniers_recos'] = recos
                    st.session_state.feedback_pending  = {}

        # Affichage résultats + feedback
        if 'derniers_recos' in st.session_state:
            recos = st.session_state['derniers_recos']
            query = st.session_state.get('derniere_query', '')

            st.markdown(f'#### 📖 Similaires à : *{query}*')
            st.markdown(f'_{len(recos)} résultat(s) — domaine : {domaine_choisi}_')
            st.markdown('---')

            for _, row in recos.iterrows():
                col_img, col_info = st.columns([1, 4])

                with col_img:
                    if row['image_url'] and row['image_url'] not in ('', 'nan'):
                        try:
                            st.image(row['image_url'], width=90)
                        except Exception:
                            st.markdown('📘')
                    else:
                        st.markdown('📘')

                with col_info:
                    badge = '<span class="new-badge">🆕</span>' if row.get('nouveau') else ''
                    st.markdown(f"""
                    <div class="{'card-new' if row.get('nouveau') else 'card'}">
                        <b>{row['rang']}. {row['titre']}</b> {badge}<br>
                        👤 {row['auteur']} &nbsp;|&nbsp;
                        <span class="domaine-tag">{row['domaine']}</span> &nbsp;|&nbsp;
                        <span class="score-badge">Score : {row['score']}</span>
                        <br><small>{row['description']}</small>
                        <br><a href="{row['url_detail']}" target="_blank">🔗 Voir la fiche</a>
                    </div>
                    """, unsafe_allow_html=True)

                    fb_key = f"fb_{row['rang']}_{row['titre'][:15]}"
                    c1, c2, c3 = st.columns([1, 1, 3])
                    with c1:
                        if st.button('👍 Utile', key=f'u_{fb_key}'):
                            sauvegarder_feedback(query, row['titre'],
                                                 row['score'], 'utile')
                            st.session_state.feedback_pending[row['titre']] = 'utile'
                    with c2:
                        if st.button('👎 Non utile', key=f'nu_{fb_key}'):
                            sauvegarder_feedback(query, row['titre'],
                                                 row['score'], 'non_utile')
                            st.session_state.feedback_pending[row['titre']] = 'non_utile'
                    with c3:
                        vote = st.session_state.feedback_pending.get(row['titre'])
                        if vote == 'utile':
                            st.success('✅ Merci !')
                        elif vote == 'non_utile':
                            st.info('📝 Noté.')

    # ════════════════════════════════════════════════════
    # ONGLET 2 — CATALOGUE (public)
    # ════════════════════════════════════════════════════
    with onglets[1]:
        st.subheader('📂 Catalogue complet')
        c1, c2 = st.columns(2)
        with c1:
            fd = st.selectbox('Domaine', domaines_display, key='cat_dom')
        with c2:
            ft = st.text_input('Rechercher dans les titres', key='cat_txt')

        df_aff = df_live.copy()
        df_aff['titre']   = df_aff['titre'].apply(str.title)
        df_aff['auteur']  = df_aff['auteur'].apply(lambda x: x.title() if x else 'Inconnu')
        df_aff['domaine'] = df_aff['domaine'].apply(str.title)

        if fd != 'Tous':
            df_aff = df_aff[df_aff['domaine'] == fd]
        if ft:
            df_aff = df_aff[df_aff['titre'].str.contains(ft, case=False, na=False)]

        st.markdown(f'**{len(df_aff)} ouvrage(s) trouvé(s)**')
        cols_aff = [c for c in ['titre', 'auteur', 'domaine', 'isbn', 'annee']
                    if c in df_aff.columns]
        st.dataframe(df_aff[cols_aff].reset_index(drop=True),
                     use_container_width=True, height=400)

        csv = df_aff.to_csv(index=False, encoding='utf-8-sig')
        st.download_button('⬇️ Télécharger (CSV)', data=csv,
                           file_name='catalogue_uac.csv', mime='text/csv')

    # ════════════════════════════════════════════════════
    # ONGLET 3 — À PROPOS (public)
    # ════════════════════════════════════════════════════
    with onglets[2]:
        st.subheader('ℹ️ À propos')
        st.markdown("""
        ### Système de Recommandation — Bibliothèque Numérique UAC

        Ce système recommande des ouvrages similaires à partir des métadonnées :
        titre, auteur, domaine et description.

        **Méthode :** TF-IDF + Similarité cosinus (Content-Based Filtering)

        | Champ | Poids |
        |---|---|
        | Titre | × 3 |
        | Auteur | × 2 |
        | Domaine | × 2 |
        | Description | × 1 |

        **Source des données :** https://bibliothequenumerique.uaconline.edu.cd

        **Cadre doctoral :** BAEL (Behavior-Aware Explainability Loop)
        — UAC Butembo, Nord-Kivu, RDC
        """)

    # ════════════════════════════════════════════════════
    # ONGLET 4 — NOUVEL OUVRAGE (admin uniquement)
    # ════════════════════════════════════════════════════
    if admin_ok:
        with onglets[3]:
            st.subheader('➕ Ajouter un nouvel ouvrage')
            st.info(
                '**Cold Start B — Ajout incrémental**  \n'
                'L\'ouvrage est immédiatement disponible sans relancer '
                'l\'entraînement complet.'
            )

            with st.form('form_nouvel_ouvrage'):
                c1, c2 = st.columns(2)
                with c1:
                    nv_titre  = st.text_input('Titre *')
                    nv_auteur = st.text_input('Auteur *')
                    nv_isbn   = st.text_input('ISBN')
                    nv_annee  = st.text_input('Année')
                with c2:
                    nv_domaine = st.selectbox('Domaine *',
                                              [d.title() for d in domaines_bruts])
                    nv_cat     = st.selectbox('Catégorie',
                                              ['Livre', 'Article', 'Mémoire UAC', 'Autre'])
                    nv_url  = st.text_input('URL fiche')
                    nv_img  = st.text_input('URL couverture')
                nv_desc = st.text_area('Description', height=100)
                soumis  = st.form_submit_button('✅ Ajouter', use_container_width=True)

            if soumis:
                if not nv_titre or not nv_auteur:
                    st.error('Titre et auteur obligatoires.')
                elif st.session_state.tfidf_obj is None:
                    st.error('TF-IDF non chargé. Vérifie les artifacts.')
                else:
                    nouvel = {
                        'titre': nv_titre, 'auteur': nv_auteur,
                        'domaine': nv_domaine, 'description': nv_desc,
                        'categorie': nv_cat, 'isbn': nv_isbn,
                        'annee': nv_annee, 'url_detail': nv_url,
                        'image_url': nv_img,
                    }
                    with st.spinner('Ajout en cours...'):
                        try:
                            df_upd, cos_upd, mat_upd = \
                                ajouter_ouvrage_incrementalement(nouvel)
                            st.session_state.df_live        = df_upd
                            st.session_state.cosine_live    = cos_upd
                            st.session_state.tfidf_mat_live = mat_upd
                            st.success(
                                f'✅ **{nv_titre}** ajouté ! '
                                f'Modèle : {len(df_upd)} ouvrages.'
                            )
                            st.balloons()
                        except Exception as e:
                            st.error(f'Erreur : {e}')

        # ════════════════════════════════════════════════
        # ONGLET 5 — ADMIN (admin uniquement)
        # ════════════════════════════════════════════════
        with onglets[4]:
            st.subheader('🛠️ Administration')

            # Feedback analytics
            st.markdown('### 📊 Analyse des feedbacks')
            fb = charger_feedback()

            if fb.empty:
                st.info('Aucun feedback enregistré.')
            else:
                total  = len(fb)
                utiles = (fb['vote'] == 'utile').sum()
                taux   = round(utiles / total * 100, 1) if total > 0 else 0

                m1, m2, m3 = st.columns(3)
                m1.metric('Total feedbacks',    total)
                m2.metric('👍 Utiles',           utiles)
                m3.metric('Taux de pertinence', f'{taux}%')

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('**Top recommandations utiles**')
                    top = (fb[fb['vote'] == 'utile']
                           .groupby('titre_reco').size()
                           .reset_index(name='nb')
                           .sort_values('nb', ascending=False).head(8))
                    st.dataframe(top, use_container_width=True)
                with c2:
                    st.markdown('**Recommandations rejetées**')
                    flop = (fb[fb['vote'] == 'non_utile']
                            .groupby('titre_reco').size()
                            .reset_index(name='nb')
                            .sort_values('nb', ascending=False).head(8))
                    st.dataframe(flop, use_container_width=True)

                st.markdown('**Historique complet**')
                st.dataframe(fb.sort_values('date', ascending=False),
                             use_container_width=True, height=250)
                csv_fb = fb.to_csv(index=False, encoding='utf-8-sig')
                st.download_button('⬇️ Exporter feedbacks', data=csv_fb,
                                   file_name='feedbacks_uac.csv', mime='text/csv')

            st.markdown('---')

            # Re-entraînement complet
            st.markdown('### 🔄 Re-entraînement complet (Cold Start A)')
            st.warning(
                'Relance le scraping complet + recalcule tout le modèle. '
                '**Durée : 30–60 min.** Recommandé une fois par mois.'
            )
            cb1, cb2 = st.columns(2)
            with cb1:
                if st.button('🚀 Lancer le re-entraînement',
                             use_container_width=True, type='primary'):
                    if re_entrainer_modele():
                        st.success('✅ Terminé ! Recharge la page.')
            with cb2:
                if os.path.exists(NOUVEAUX_CSV):
                    df_new = pd.read_csv(NOUVEAUX_CSV)
                    st.info(f'{len(df_new)} ouvrage(s) en attente d\'intégration.')
                    cols_n = [c for c in ['titre', 'auteur', 'domaine',
                                          'ajoute_par', 'date_ajout']
                              if c in df_new.columns]
                    st.dataframe(df_new[cols_n], use_container_width=True)
                else:
                    st.info('Aucun ouvrage en attente.')

            st.markdown('---')
            st.markdown('### 👥 Comptes administrateurs')
            config_loaded = charger_config()
            users_data = [
                {'identifiant': u, 'nom': v['nom']}
                for u, v in config_loaded.get('admins', {}).items()
            ]
            st.dataframe(pd.DataFrame(users_data), use_container_width=True)
            st.caption('Pour modifier les comptes, édite `uac_config.yaml`.')


if __name__ == '__main__':
    main()
