"""Multilingual lexicons: French, MSA Arabic, Darija in Arabic script, Darija in Arabizi.

All entries are written in their *normalized* form conceptually; they are passed
through `normalize()` at load time, so accents / hamza variants don't matter here.
"""

# intent -> {keyword: weight}.  Multi-word phrases are fine.
INTENT_KEYWORDS: dict[str, dict[str, float]] = {
    "order_medication": {
        # fr
        "commander": 2, "commande": 2, "reserver": 2, "reservation": 2, "je veux": 1.5,
        "je voudrais": 1.5, "j aimerais": 1.5, "preparer": 1.5, "mettre de cote": 2,
        "livrer": 2, "livraison": 2, "acheter": 1.5, "il me faut": 1.5, "besoin de": 1.2,
        # arabizi
        "bghit": 1.5, "bghina": 1.5, "nbghi": 1.5, "khlli liya": 2, "khalli lia": 2, "khelli lia": 2,
        "wjjd liya": 2, "wajjed lia": 2, "wajdo": 2, "sift": 2, "sifet": 2, "jib": 1.5, "jibli": 2,
        "khasni": 1.5, "5asni": 1.5, "hjz": 2,
        # darija arabic script
        "بغيت": 1.5, "بغينا": 1.5, "نبغي": 1.5, "خلي ليا": 2, "خليو ليا": 2, "وجد ليا": 2, "وجدو ليا": 2,
        "صيفط": 2, "صيفطو": 2, "جيب": 1.5, "جيبو": 1.5, "خاصني": 1.5, "محتاج": 1.2,
        # msa
        "اريد": 1.5, "اطلب": 2, "طلب": 1.5, "احجز": 2, "حجز": 2, "اود": 1.5, "توصيل": 2, "شراء": 1.5,
    },
    "check_availability": {
        "disponible": 2, "dispo": 2, "disponibilite": 2, "avez vous": 2, "vous avez": 2, "en stock": 2,
        "il y a": 1, "rupture": 2, "trouver": 1, "trouve": 1, "est ce que vous avez": 2.5,
        "kayn": 2, "kayna": 2, "kaynin": 2, "wach kayn": 2.5, "3ndkom": 2, "3endkom": 2, "andkom": 2,
        "3andkom": 2, "mojod": 2, "mawjoud": 2, "nl9a": 1.5, "nlqa": 1.5,
        "كاين": 2, "كاينة": 2, "كاينين": 2, "واش كاين": 2.5, "عندكم": 2, "عندك": 1.5, "موجود": 2,
        "متوفر": 2, "متوفرة": 2, "يوجد": 2, "هل يوجد": 2.5, "نلقى": 1.5, "مقطوع": 2,
    },
    "ask_price": {
        "prix": 2.5, "combien": 2, "coute": 2.5, "tarif": 2.5, "ca fait combien": 3, "cher": 1,
        "ch7al": 2.5, "chhal": 2.5, "chal": 2, "bch7al": 2.5, "bchhal": 2.5, "taman": 2.5, "tmen": 2.5,
        "شحال": 2.5, "بشحال": 2.5, "ثمن": 2.5, "تمن": 2.5, "سعر": 2.5, "بكم": 2.5, "كم": 1, "غالي": 1,
    },
    "ask_reimbursement": {
        "remboursable": 3, "rembourse": 3, "remboursement": 3, "cnops": 3, "cnss": 3, "amo": 3,
        "mutuelle": 3, "tiers payant": 3, "remboursi": 3, "kayt3wed": 3, "t3wid": 3,
        "تعويض": 3, "كيتعوض": 3, "يتعوض": 3, "التعاضدية": 3, "تعاضدية": 3, "كنوبس": 3, "الضمان": 2,
        "استرجاع": 2,
    },
    "ask_alternative": {
        "generique": 3, "equivalent": 3, "remplacer": 2.5, "alternative": 3, "a la place": 2.5,
        "remplacant": 3, "bdil": 3, "blasto": 2.5, "f blasto": 3,
        "بديل": 3, "جنيس": 3, "بلاصتو": 2.5, "عوض": 2, "مكانه": 2,
    },
    "find_pharmacy": {
        "pharmacie la plus proche": 3, "plus proche": 2.5, "ou se trouve": 2.5, "adresse": 2,
        "localisation": 2.5, "proche": 1.5, "pres de": 1.5, "a cote": 1.5, "ou est": 2, "pharmacies": 1,
        "fin": 1.5, "fen": 1.5, "fin kayna": 2.5, "qrib": 2, "9rib": 2, "qriba": 2, "9riba": 2,
        "فين": 1.5, "فين كاينة": 2.5, "قريب": 2, "قريبة": 2, "اقرب": 2.5, "عنوان": 2, "اين": 2,
        "موقع": 2, "الموقع": 2, "صيدلية قريبة": 3,
    },
    "pharmacy_contact": {
        "numero": 3, "telephone": 3, "tel": 2.5, "appeler": 2.5, "contacter": 2.5, "joindre": 2,
        "nmra": 3, "nemra": 3, "ra9m": 3, "raqm": 3, "ntasl": 2.5, "n3iyet": 2.5,
        "نمرة": 3, "رقم": 3, "هاتف": 3, "تيليفون": 3, "نتاصل": 2.5, "اتصل": 2.5, "نعيط": 2.5,
    },
    "pharmacy_hours": {
        "garde": 3, "de garde": 3.5, "ouverte": 2, "ouvert": 2, "horaire": 2.5, "horaires": 2.5,
        "heures d ouverture": 3, "nuit": 2, "dimanche": 1.5, "ferme": 1.5,
        "m7loula": 2, "mhloula": 2, "ma7loula": 2, "f lil": 2, "llil": 2,
        "حراسة": 3, "الحراسة": 3, "كارد": 3, "مفتوحة": 2, "محلولة": 2, "الليل": 2, "توقيت": 2.5,
        "ساعات": 1.5,
    },
    # medical-advice questions: the assistant must NOT answer them (safety), only redirect
    "out_of_scope": {
        "effets secondaires": 4, "effet secondaire": 4, "posologie": 4, "combien de comprimes": 4,
        "combien de fois": 3, "je peux prendre": 3, "peut on prendre": 3, "prendre avec": 3, "enceinte": 3,
        "allaitement": 3, "alcool": 3, "dangereux": 3, "contre indication": 4, "surdosage": 4, "que dois je prendre": 4,
        "quoi prendre": 4, "chno nakhod": 4, "chno nakhd": 4, "chno n3tih": 4, "chno n3tiha": 4, "ch7al mn 7ba": 4,
        "chhal mn 7ba": 4, "nakhod m3a": 3, "wach n9der nakhod": 4, "bl3ani": 2, "mra 7amla": 4, "7amla": 3,
        "شحال من حبة": 4, "شحال من مرة": 3, "ناخد": 3, "نشرب": 2, "شنو ناخد": 4, "شنو نعطيه": 4, "شنو نعطيها": 4,
        "الاثار الجانبية": 4, "الاعراض الجانبية": 4, "الجرعة": 4, "جرعة": 3, "حامل": 3, "الحمل": 3, "الرضاعة": 3,
        "هل يمكنني تناول": 4, "تناول": 2, "خطير": 3, "مع الكحول": 4,
    },
    "greeting": {
        "bonjour": 1, "bonsoir": 1, "salut": 1, "salam": 1, "slm": 1, "salam alaykom": 1,
        "سلام": 1, "السلام عليكم": 1, "مرحبا": 1, "صباح الخير": 1, "مساء الخير": 1,
    },
}

# when several intents tie, prefer the more "actionable" one
INTENT_PRIORITY = ["out_of_scope", "order_medication", "ask_price", "ask_reimbursement", "ask_alternative",
                   "check_availability", "pharmacy_contact", "pharmacy_hours", "find_pharmacy", "greeting"]

# words signalling a pharmacy
PHARMACY_WORDS = ["pharmacie", "pharmacies", "pharma", "farmasyan", "farmasya", "fermasyan", "saydaliya",
                  "صيدلية", "الصيدلية", "فرمسيان", "فارماسي", "صيدليات"]

NEAR_ME = ["pres de moi", "proche de moi", "autour de moi", "a cote de moi", "qrib liya", "9rib liya",
           "qrib mni", "9rib mni", "qriba liya", "9riba liya", "qriba lia", "9riba lia", "qrib lia", "9rib lia", "قريبة ليا", "قريب ليا", "قريب مني", "قريبة مني", "حدايا", "hdaya"]

# ---------------------------------------------------------------- numbers & units
NUMBER_WORDS = {
    # fr ("un/une" only counts before a unit word, see nlu)
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    # arabizi
    "wahed": 1, "wa7ed": 1, "wahd": 1, "wa7d": 1, "jouj": 2, "zouj": 2, "joj": 2, "zoj": 2, "jooj": 2,
    "tlata": 3, "tlta": 3, "tlat": 3, "rb3a": 4, "reb3a": 4, "rab3a": 4, "khamsa": 5, "5amsa": 5,
    "stta": 6, "setta": 6, "sb3a": 7, "seb3a": 7, "tmnya": 8, "tmenya": 8, "3achra": 10, "3ashra": 10,
    # arabic (darija + msa)
    "واحد": 1, "وحدة": 1, "جوج": 2, "زوج": 2, "اثنين": 2, "اثنان": 2, "تلاتة": 3, "ثلاثة": 3, "تلاثة": 3,
    "ربعة": 4, "اربعة": 4, "خمسة": 5, "ستة": 6, "سبعة": 7, "تمنية": 8, "ثمانية": 8, "عشرة": 10,
}
FR_ARTICLE_ONE = {"un", "une"}
DUAL_UNITS = {"علبتين": 2, "3lbtin": 2, "3elbtin": 2, "قرعتين": 2, "9er3tin": 2, "حبتين": 2}

UNIT_WORDS = {
    "boite": "box", "boites": "box", "bte": "box", "btes": "box", "3lba": "box", "3elba": "box", "3lab": "box",
    "l3lba": "box", "علبة": "box", "علب": "box", "لعلبة": "box", "plaquette": "blister", "plaquettes": "blister",
    "بلاكيت": "blister", "flacon": "bottle", "flacons": "bottle", "9er3a": "bottle", "قرعة": "bottle",
    "tube": "tube", "tubes": "tube", "sachet": "sachet", "sachets": "sachet", "comprime": "tablet",
    "comprimes": "tablet", "7ba": "tablet", "حبة": "tablet", "حبات": "tablet",
}

DOSAGE_UNITS = {"mg": "mg", "g": "g", "gr": "g", "ml": "ml", "mcg": "mcg", "ug": "mcg", "ui": "ui", "%": "%",
                "ملغ": "mg", "مغ": "mg", "غرام": "g", "ميليغرام": "mg"}

FORM_WORDS = {
    "comprime": "comprimé", "comprimes": "comprimé", "cp": "comprimé", "cpr": "comprimé", "7bob": "comprimé",
    "7ab": "comprimé", "حبوب": "comprimé", "كينة": "comprimé", "gelule": "gélule", "gelules": "gélule",
    "sirop": "sirop", "chrab": "sirop", "شراب": "sirop", "سيرو": "sirop", "siro": "sirop",
    "injectable": "injectable", "injection": "injectable", "piqure": "injectable", "ibra": "injectable",
    "ابرة": "injectable", "شوكة": "injectable", "suppositoire": "suppositoire", "suppo": "suppositoire",
    "تحميلة": "suppositoire", "sachet": "sachet", "sachets": "sachet", "pommade": "pommade",
    "creme": "crème", "دهينة": "pommade", "مرهم": "pommade", "collyre": "collyre", "قطرة": "collyre",
    "inhalateur": "inhalateur", "spray": "spray", "بخاخ": "inhalateur", "effervescent": "effervescent",
}

# canonical city -> spellings
CITIES: dict[str, list[str]] = {
    "Rabat": ["rabat", "rbat", "الرباط", "رباط"],
    "Salé": ["sale", "sla", "سلا"],
    "Témara": ["temara", "tmara", "تمارة"],
    "Casablanca": ["casablanca", "casa", "dar lbida", "dar el beida", "الدار البيضاء", "البيضاء", "كازا",
                   "كازابلانكا", "الدارالبيضاء"],
    "Mohammedia": ["mohammedia", "mohamedia", "المحمدية"],
    "Fès": ["fes", "fez", "fas", "فاس"],
    "Meknès": ["meknes", "mknas", "مكناس"],
    "Marrakech": ["marrakech", "marrakesh", "mrakch", "مراكش"],
    "Tanger": ["tanger", "tangier", "tanja", "طنجة"],
    "Tétouan": ["tetouan", "tetwan", "تطوان"],
    "Agadir": ["agadir", "akadir", "اكادير", "أكادير"],
    "Oujda": ["oujda", "wjda", "وجدة"],
    "Kénitra": ["kenitra", "qnitra", "knitra", "القنيطرة", "قنيطرة"],
    "El Jadida": ["el jadida", "jadida", "ljadida", "الجديدة"],
    "Safi": ["safi", "asfi", "اسفي", "آسفي"],
    "Nador": ["nador", "الناظور", "ناظور"],
    "Beni Mellal": ["beni mellal", "bni mellal", "بني ملال"],
    "Khouribga": ["khouribga", "خريبكة"],
    "Settat": ["settat", "سطات"],
    "Taza": ["taza", "تازة"],
    "Larache": ["larache", "l3rayech", "العرائش"],
    "Khémisset": ["khemisset", "الخميسات"],
    "Tiflet": ["tiflet", "تيفلت"],
    "Ifrane": ["ifrane", "افران", "إفران"],
    "Essaouira": ["essaouira", "sawira", "الصويرة"],
    "Ouarzazate": ["ouarzazate", "ورزازات"],
    "Errachidia": ["errachidia", "الرشيدية"],
    "Laâyoune": ["laayoune", "layoune", "العيون"],
    "Dakhla": ["dakhla", "الداخلة"],
    "Berkane": ["berkane", "بركان"],
    "Guelmim": ["guelmim", "كلميم"],
}

# language markers (normalized)
DARIJA_AR_MARKERS = {"بغيت", "واش", "شحال", "بشحال", "فين", "ديال", "ديالي", "عندكم", "كاين", "كاينة", "دابا",
                     "شي", "علاش", "كيفاش", "خاصني", "ليا", "جوج", "صيفط", "مزيان", "بزاف", "غير", "هادا",
                     "هادي", "نتوما", "واخا", "حتى", "باش", "كنقلب", "نلقى", "محلولة", "فرمسيان"}
MSA_MARKERS = {"اريد", "هل", "يوجد", "اين", "الذي", "التي", "كم", "متوفر", "ارجو", "لدي", "لديكم", "ثمن",
               "سعر", "من فضلك", "هذا", "هذه", "اقرب", "هاتف", "دواء", "توجد", "لدواء", "قابل", "ماذا", "الآن", "اود", "شراء", "هي", "ارغب"}
ARABIZI_MARKERS = {"bghit", "wach", "ch7al", "chhal", "fin", "dyal", "dial", "3ndkom", "3endkom", "kayn",
                   "kayna", "daba", "chi", "khasni", "liya", "lia", "jouj", "zouj", "mzyan", "bzaf", "ghir",
                   "wakha", "bach", "9rib", "qrib", "f", "had", "hadi", "hada", "3afak", "3fak", "salam",
                   "sift", "nl9a", "m7loula", "farmasyan", "fermasyan", "qriba", "9riba", "kayn", "taman", "bghina", "3lba"}
FRENCH_MARKERS = {"je", "vous", "est", "le", "la", "les", "des", "du", "de", "pour", "avez", "une", "un",
                  "bonjour", "merci", "pharmacie", "prix", "combien", "est ce", "svp", "s il", "plait",
                  "voudrais", "veux", "ou", "quel", "quelle", "avec", "mon", "ma", "mes"}

STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "a", "au", "aux", "en", "pour", "je", "j",
    "vous", "est", "ce", "il", "elle", "me", "moi", "mon", "ma", "mes", "svp", "merci", "avec", "sans",
    "bonjour", "bonsoir", "salut", "pharmacie", "comment", "joindre", "besoin", "chez", "cher", "moins", "pharmacies", "medicament", "medicaments", "dwa", "dwaya",
    "f", "fi", "l", "d", "dyal", "dial", "had", "hada", "hadi", "wach", "chi", "liya", "lia", "3afak", "3fak",
    "daba", "ana", "salam", "mg", "ml", "g",
    "في", "من", "على", "ديال", "دواء", "الدواء", "دوا", "هاد", "هادا", "شي", "واش", "عافاك", "الله", "ليا",
    "انا", "دابا", "هل", "الان", "الآن", "ليوم", "اليوم", "دبا", "عفاك", "maintenant", "aujourd", "hui", "lyoum", "dba", "ce", "soir", "او", "مع", "عند", "صيدلية", "الصيدلية", "سلام",
}

# frequent function / chat words; with ~2 000 real brands, fuzzy matching would otherwise hit some of them
COMMON_WORDS = {
    "merci", "beaucoup", "votre", "notre", "vos", "nos", "aide", "aider", "cherche", "chercher", "avoir",
    "faire", "peux", "pouvez", "voudrais", "voulez", "veux", "bien", "tres", "aussi", "encore", "comme", "quoi",
    "quand", "pourquoi", "comment", "alors", "donc", "mais", "parce", "merci", "madame", "monsieur", "docteur",
    "ordonnance", "medecin", "enfant", "bebe", "adulte", "malade", "douleur", "mal", "tete", "ventre", "fievre",
    "toux", "grippe", "rhume", "allergie", "demain", "matin", "soir", "nuit", "semaine", "jour", "heure",
    "ville", "quartier", "adresse", "numero", "telephone", "ouverte", "fermee", "garde", "proche", "loin",
    "combien", "prix", "cher", "rembourse", "remboursable", "generique", "equivalent", "disponible", "stock",
    "boite", "boites", "flacon", "sachet", "comprime", "comprimes", "sirop", "gelule", "creme", "pommade",
    "secable", "effervescent", "effervescents", "pellicule", "enrobe", "dispersible", "orodispersible",
    "buvable", "injectable", "suppositoire", "suppositoires", "solution", "suspension", "poudre", "ovule",
    "lyophilisat", "collyre", "gouttes", "inhalation", "gastro", "resistante", "retard", "forme",
    "cava", "labas", "bikhir", "hamdoulah", "hamdullah", "safi", "wakha", "mzyan", "bzaf", "chwiya", "chi",
    "wach", "kayn", "kayna", "bghit", "khasni", "3afak", "baraka", "lah", "llah", "inchallah", "nchallah",
    "شكرا", "بزاف", "مزيان", "الحمد", "لله", "الله", "بخير", "لاباس", "عافاك", "واخا", "صافي", "شوية", "ولدي",
    "بنتي", "الراس", "الكرش", "السخانة", "الكحة", "الطبيب", "الدكتور", "الوصفة", "ورقة", "غدا", "الصباح",
}

