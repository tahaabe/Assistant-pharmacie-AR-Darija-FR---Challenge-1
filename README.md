# Assistant patient–pharmacie · Arabe / Darija / Français (Challenge #1)

Demande texte **ou** vocale → détection de langue → intention → entités (médicament, dosage, forme,
quantité, ville, pharmacie) → **ancrage dans le référentiel CNOPS réel** (5 917 produits) et l'annuaire
des pharmacies → JSON structuré + réponse dans la langue du patient, via une API FastAPI.
Un **LLM de secours (Groq)** n'intervient que lorsque les règles ne sont pas sûres d'elles.

```
 audio ─▶ ASR Whisper (Groq ou faster-whisper) ─┐
 texte ─────────────────────────────────────────┴─▶ normalisation (variantes de lettres arabes, ٠-٩, accents, "500mg")
                                                       │
                     ┌─────────────────────────────────┼──────────────────────────────────┐
                     ▼                                 ▼                                  ▼
              détection de langue          intention (lexique pondéré FR/AR/Darija    entités : médicaments via
          (darija / ar / fr / arabizi,      + règles de contexte, dont un garde-fou     squelette consonantique
           mélange)                         « conseil médical » → hors périmètre)       multi-alphabet vs CNOPS,
                                                                                         quantité, dosage, forme, ville
                     └─────────────────────────────────┬──────────────────────────────────┘
                                                       ▼
                         confiance faible ? ──oui──▶ LLM Groq : intention + réécriture française
                                │                    └─▶ réécriture repassée dans l'extracteur à règles
                                │                        (les médicaments restent vérifiés dans CNOPS)
                                ▼
            gestionnaire de dialogue (session : variante → ville → pharmacie)
                                ▼
            JSON + brouillon `pharmacy_request` + réponse ─▶ POST /v1/requests (envoi à la pharmacie)
```

## Lancer le projet (Windows / VS Code)

Ouvre un terminal **dans le dossier qui contient `app/`** (si tu as `pharma_assistant\pharma_assistant`, c'est le dossier intérieur).
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
* Démo chat (texte + micro) : http://127.0.0.1:8000/  ·  Swagger : http://127.0.0.1:8000/docs
* `/health` indique les fichiers de données chargés et si le LLM est actif.
* Tests : `pytest -q` (22 tests)  ·  Évaluation : `python scripts/evaluate.py --file eval/heldout_v2.jsonl --tag mon_run`
* Compatible Python **3.9 → 3.13** (testé en 3.9 et 3.13).

### Activer le LLM de secours et la voix (clé Groq gratuite : console.groq.com)
PowerShell :
```powershell
$env:GROQ_API_KEY="gsk_..."
uvicorn app.main:app --reload
```
(dans `cmd` : `set GROQ_API_KEY=gsk_...`). La même clé active la transcription vocale (Whisper) et le LLM.
Options : `GROQ_LLM_MODEL` (défaut `llama-3.3-70b-versatile`), `LLM_CONFIDENCE_THRESHOLD` (défaut 0.45),
`LLM_FALLBACK=0` pour le désactiver. Sans clé, le texte fonctionne en mode règles seules (la voix demande alors `pip install faster-whisper`).

## Données

### Médicaments — référentiel CNOPS réel ✅
`data/raw/ref-des-medicaments-cnops-2014.xlsx` : « Référentiel des médicaments » publié par la **CNOPS**
sur le portail open data **data.gov.ma** (jeu `referentiel-des-medicaments`, version 2014 — la plus récente
publiée sur le portail ; SHA-256 `e625e9f1…50d3`). 5 917 produits, 2 189 marques ; colonnes CODE (EAN),
NOM, DCI, DOSAGE, FORME, PRESENTATION, PPV, PH, PRIX_BR, PRINCEPS/GÉNÉRIQUE, TAUX_REMBOURSEMENT.
Converti par `scripts/import_cnops.py` en `data/medicaments.csv` (chargé automatiquement) :
* marque = premier mot du nom (« DOLIPRANE VITAMINE C » → DOLIPRANE), sauf mots génériques (ACIDE, VITAMINE…) ;
* chaque molécule d'une association est indexée (« SULFATE FERREUX / ACIDE FOLIQUE ») ;
* remboursable = taux > 0 % (3 940 produits à 70 %, 1 977 à 0 %) ;
* **limite** : prix et remboursements de 2014 ; des produits plus récents (ex. Advil, Nexium) n'y sont pas.

Pour une version plus récente du fichier : `python scripts/import_cnops.py nouveau_fichier.xlsx`.

### Pharmacies — Assaydaliya ⚠️ (encore l'échantillon)
Le site Assaydaliya n'a pas pu être atteint depuis l'environnement de développement (ni le site, ni une
alternative OpenStreetMap). L'app utilise donc toujours `data/sample_pharmacies.csv` (10 pharmacies
**fictives** « Pharmacie Démo … »). Dès que vous avez l'export (JSON / CSV / XLSX) :
```powershell
python scripts/import_pharmacies.py export_assaydaliya.json
```
Champs reconnus : nom, ville, quartier, adresse, téléphone, latitude, longitude, garde, horaires
(villes normalisées « rbat / الرباط → Rabat », téléphones au format +212). Les gardes tournent chaque jour :
l'import doit être relancé régulièrement.

## Le LLM de secours (app/llm.py)
* **Quand** : intention inconnue, confiance < 0,45, phrase avec un médicament mais sans verbe reconnu,
  ou question sur un médicament que les règles n'ont pas pu identifier. Jamais pour une réponse courte de
  suivi de dialogue (« 1000 », « f rbat »), ni quand les règles sont sûres.
* **Quoi** : le LLM renvoie un JSON contraint — intention (liste fermée + `out_of_scope`), langue,
  **réécriture française** de la demande. La réécriture repasse par l'extracteur à règles :
  un médicament absent du référentiel CNOPS reste « non trouvé » (testé).
* **Sécurité** : le LLM ne répond jamais au patient. Les questions de conseil médical (posologie, effets
  secondaires, grossesse, alcool…) sont détectées par des règles **et** par le LLM → réponse fixe
  « demandez à votre pharmacien / médecin ».
* En cas d'erreur réseau / quota / JSON invalide → on garde silencieusement le résultat des règles.
* La sortie API indique `nlu_source: "rules"` ou `"rules+llm"` et le détail dans `llm`.

## API
| méthode | chemin | rôle |
|---|---|---|
| POST | `/v1/analyze` | `{text, session_id?, lat?, lon?}` → interprétation complète |
| POST | `/v1/analyze/voice` | audio → transcription → même sortie (+ `input.asr`) |
| GET | `/v1/medications/search?q=` | recherche approximative dans le référentiel |
| GET | `/v1/medications/{code}` | produit + équivalents (même DCI, même dosage) |
| GET | `/v1/pharmacies?city=&garde=&lat=&lon=` | annuaire / plus proches |
| POST | `/v1/requests` | confirme un `pharmacy_request` pour un `pharmacy_id` |

Intentions : `order_medication`, `check_availability`, `ask_price`, `ask_reimbursement`, `ask_alternative`,
`find_pharmacy`, `pharmacy_hours`, `pharmacy_contact`, `greeting`, `out_of_scope`, `unknown`.

Exemple de dialogue sur le vrai référentiel :
```
> je voudrais 2 boites de doliprane   → quel dosage ? 500 / 150 MG / MG ; 300 MG ; 150 MG ; 1 G ; 200 MG ; 100 MG ; 500 MG
> 1000                                 → quelle forme ? comprime secable ; suppositoire ; comprime effervescent
> comprimé sécable                     → Demande préparée : 2 × DOLIPRANE 1 G comprime secable. Dans quelle ville ?
> f rbat                               → pharmacies proposées à Rabat … (pharmacy_request avec le code EAN 6118000040972)
```

## Évaluation (scripts/evaluate.py)
Trois jeux écrits par nous (pas de vrais utilisateurs) :
`eval/scenarios.jsonl` (53, **dev** — règles ajustées dessus), `eval/heldout.jsonl` (30, held-out v1),
`eval/heldout_v2.jsonl` (30, held-out v2 écrit **après** l'import CNOPS, avec de nouvelles marques et
4 questions de conseil médical). Chaque held-out a été évalué **une fois avant toute correction** :
c'est ce premier passage qui compte.

| jeu | référentiel | langue | intention | F1 médic. | dosage | exact |
|---|---|---|---|---|---|---|
| held-out v1, **1er passage** | échantillon | 0.93 | 1.00 | 0.90 | 0.71 | 0.80 |
| held-out v2, **1er passage** (règles seules) | CNOPS réel | 0.93 | **0.87** | **0.94** | 1.00 | **0.77** |
| held-out v2 après corrections (plus indépendant) | CNOPS réel | 0.93 | 1.00 | 1.00 | 1.00 | 0.93 |
| dev (ajusté) | CNOPS réel | 0.96 | 1.00 | 0.99 | 1.00 | 0.94 |

Ce que le 1er passage v2 a révélé (puis corrigé) : les 4 questions médicales recevaient une réponse
prix/disponibilité (→ garde-fou `out_of_scope`) ; « البيتادين » lu comme OLAPATADINE à cause de l'article
ال (→ préférence pour la lecture sans clitique) ; « تاهور » (Tahor) non trouvé car squelette trop court.
Sur le vrai référentiel, l'évaluation a aussi fait apparaître des faux positifs (« votre » → VITAIR,
« dafalgan » → DIFLUCAN, « sécable » → ASCABIOL), corrigés par un contrôle sur les voyelles et une liste de
mots courants. Fichiers : `eval/results_*.json`, erreurs dans `eval/errors_*.jsonl`.
Latence ≈ 2 ms / message (CPU, règles). **Le LLM n'a été testé qu'avec des réponses simulées** :
lancer l'évaluation avec `GROQ_API_KEY` défini pour mesurer son apport réel (`llm_fallback_calls` dans le résultat).

## Limites
* Jeux de test petits et écrits par nous → montrent que le pipeline fonctionne, pas sa précision réelle.
* Référentiel CNOPS de 2014 (prix / produits datés) ; pharmacies encore fictives.
* « Disponibilité » ≠ stock réel : on prépare une demande que la pharmacie confirme.
* ASR sur de la darija parlée non mesurée (WER à calculer sur des enregistrements).
* Pas de conseil médical, par conception.

## Arborescence
```
app/normalize.py   normalisation + squelette multi-alphabet
app/lexicon.py     intentions, nombres, unités, formes, villes, marqueurs de langue, mots courants
app/data.py        référentiel CNOPS + annuaire, recherche approximative, pharmacies proches
app/nlu.py         langue, intention, entités, ancrage
app/llm.py         LLM de secours Groq (JSON contraint, validé)
app/dialog.py      sessions, questions de précision, demande structurée, réponses multilingues
app/asr.py         Whisper (Groq / faster-whisper)
app/main.py        API FastAPI  ·  app/demo.html  page de démo
scripts/           import_cnops.py, import_pharmacies.py, evaluate.py
data/raw/          fichier CNOPS d'origine · data/medicaments.csv  référentiel importé
eval/  tests/
```
