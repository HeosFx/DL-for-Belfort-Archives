import os
import re
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
from tqdm import tqdm


# Load the model on the available device(s)
with torch.no_grad():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

print("Model and Processor Loaded Successfully!")

# Start timer
start_time = time.time()

input_dir = Path("../data/inputs")
output_dir = Path("../data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Filtrer les fichiers image
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
# Trouver toutes les images dans tous les sous-dossiers
image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_extensions]

system_prompts = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "../data/inputs/A-B/IMG_1198.jpg"},
            {
                "type": "text",
                "text": """Voici un exemple de la tâche que vous devez accomplir. Vous devez extraire les informations du document manuscrit et les structurer en XML."""
            }
        ]
    },
    {
      "role": "assistant",
      "content": [
          {
                "type": "text", "text": """Voici la transcription correspondant à l'exemple ci-dessus (XML) :
                    <?xml version="1.0" encoding="utf-8"?>
<Document>
  <Nom>Bouteiller Hélène</Nom>
  <Genre>Née</Genre>
  <NomDuConjoint>
    <added-above>
      <sprited>Vve</sprited>
    </added-above>
     
    <sprited>fme Rayot</sprited>
     Fme Braun
  </NomDuConjoint>
  <DateDeNaissance>6 Octobre 1899</DateDeNaissance>
  <LieuDeNaissance>Hérimoncourt</LieuDeNaissance>
  <Nationalité>française</Nationalité>
  <Ville>
    <pence>Belfort</pence>
  </Ville>
  <Adresse>
    <pence>du Berger, 19</pence>
  </Adresse>
  <Table>
    <Ligne>
      <Occupation>
        Fge 
        <pence>cartonnage</pence>
      </Occupation>
      <Entrée>31 Mars 1924</Entrée>
      <Sortie>5 Juillet 1924</Sortie>
      <Présence>3</Présence>
      <Observations>
        5j 
        <printed>
          <coloured-ink>Gratification 1925</coloured-ink>
        </printed>
      </Observations>
      <Divers>89.-</Divers>
    </Ligne>
    <Ligne>
      <Occupation>- d° -</Occupation>
      <Entrée>21 Juillet 1924</Entrée>
      <Sortie>10 Janvier 1930</Sortie>
      <Présence>55</Présence>
      <Observations>
        21 jours 
        <bis/>
        180e ann.
      </Observations>
      <Divers>100.-</Divers>
    </Ligne>
    <Ligne>
      <Observations>
        <bis/>
        1926
      </Observations>
      <Divers>117.-</Divers>
    </Ligne>
    <Ligne>
      <Observations>
        <bis/>
        1927
      </Observations>
      <Divers>126-</Divers>
    </Ligne>
    <Ligne>
      <Observations>
        <bis/>
        10° an. Arm
      </Observations>
      <Divers>140-</Divers>
    </Ligne>
    <Ligne>
      <Observations>
        <bis/>
        1928
      </Observations>
      <Divers>159-</Divers>
    </Ligne>
    <Ligne>
      <Observations>
        <bis/>
        1929
      </Observations>
      <Divers>182-</Divers>
    </Ligne>
    <Ligne>
      <Observations>quitte Belfort</Observations>
    </Ligne>
  </Table>
  <Annotation>
    <pence>Réfectoire 24</pence>
  </Annotation>
</Document>
"""
            },
      ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "../data/inputs/A-B/IMG_0690.jpg"},
            {
                "type": "text",
                "text": """Voici un second exemple de la tâche que vous devez accomplir. Vous devez extraire les informations du document manuscrit et les structurer en XML."""
            }
        ]
    },
    {
      "role": "assistant",
      "content": [
          {
                "type": "text", "text": """Voici la transcription correspondant au second exemple ci-dessus (XML) :
                    <?xml version="1.0" encoding="utf-8"?>
                    <Document>
                    <Nom>Bernhard Jeanne Berthe</Nom>
                    <Genre>Née</Genre>
                    <NomDuConjoint>Fme Blind Georg</NomDuConjoint>
                    <DateDeNaissance>19 Mars 1907</DateDeNaissance>
                    <LieuDeNaissance>Etueffont-Bas</LieuDeNaissance>
                    <Nationalité>française</Nationalité>
                    <Ville>
                        <pence>Belfort</pence>
                    </Ville>
                    <Adresse>
                        <pence>de la Marseillaise N°21</pence>
                    </Adresse>
                    <Table>
                        <Ligne>
                        <Occupation>Fge pliage</Occupation>
                        <Entrée>30 Janvier 1922</Entrée>
                        <Sortie>
                            24 Nov
                            <added-above>bre</added-above>
                            1923
                        </Sortie>
                        <Présence>1 9</Présence>
                        <Observations>
                            <added-above>
                            28 jours 
                            <pence>annonce par nous</pence>
                            </added-above>
                            
                            <printed>
                            <coloured-ink>Gratification 1925</coloured-ink>
                            </printed>
                        </Observations>
                        <Divers>61. -</Divers>
                        </Ligne>
                        <Ligne>
                        <Occupation>- M -</Occupation>
                        <Entrée>11 Août 1924</Entrée>
                        <Sortie>7 Janvier 1928</Sortie>
                        <Présence>3 4</Présence>
                        <Observations>
                            <added-above>Pour garder son enfant</added-above>
                            
                            <coloured-ink>Alsthom</coloured-ink>
                            
                            <bis/>
                            180e ann
                        </Observations>
                        <Divers>70. -</Divers>
                        </Ligne>
                        <Ligne>
                        <Occupation>Fin. Pliage</Occupation>
                        <Entrée>1er Avril 1930</Entrée>
                        <Sortie>14 Juin 1930</Sortie>
                        <Présence>- 2</Présence>
                        <Observations>
                            <added-above>le compte a été demandé par le Cx</added-above>
                            14 j. 
                            <bis/>
                            1920
                        </Observations>
                        <Divers>77. -</Divers>
                        </Ligne>
                        <Ligne>
                        <Observations>
                            <bis/>
                            1927
                        </Observations>
                        <Divers>87 -</Divers>
                        </Ligne>
                    </Table>
                    <Annotation>
                        <pence>Touche l'allocation familiale</pence>
                        
                    Pupille de la Nation
                    </Annotation>
                    </Document>
"""
            },
      ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": None},
            {
                "type": "text",
                "text": """Vous êtes une IA spécialisée dans l'extraction d'informations de documents historiques manuscrits en français.
Dans un premier temps, vous lisez les informations de l'entête du document en reconnaissant les éléments sous la forme ("first level", "second level") :
```plaintext
[("Nom complet", ""),
("Nom complet du Mari", ""),
("Marqueur", ""),
("Date de naissance", ""),
("Lieu de naissance", ""),
("Nationalité", ""),
("Statut marital", ""),
("Genre", ""),
("Adresse", ""),
("Ville", "")]
```
Contexte :
- Le document est une fiche d'entreprise manuscrite en français datant des années 1900.
- Le nom de la personne est composé d'un ou plusieurs prénoms et d'un nom de famille.
- Les noms des personnes et des villes ne sont pas toujours français.
- Les marqueurs peuvent être 'fme', 'fe', 'vve', 've', 'femme', 'divorcée', etc.
- Les marqueurs précèdent le nom du mari si la personne est une femme mariée.
- Le nom du conjoint est composé d'un ou plusieurs marqueurs et du ou des noms. ex : "fme Jean Dupont", "ve Marie Curie".
- La date de naissance peut être sous la forme "jj/mm/aaaa" ou "jj mois aaaa".
- Le lieu de naissance peut être une ville, un village, un pays, etc.
- Un individu peut avoir plusieurs nationalités (toutes les indiquer).
- Les statuts maritaux doit être 'célibataire', 'marié', 'divorcé', 'veuf', etc ou "" si indéfini.
- Le nom de la ville est optionnel.
- Si la ligne "Etat civil" est présente, l'adresse s'y trouve obligatoirement.
- L'adresse doit contenir le nom de la rue et un numéro.	
- Le genre peut être 'Née' ou 'Né' et est déterminé par la présence d'un e à 'née'.
Tâche :
En vous inspirant des exemples donnés, reconstruisez, s'il vous plait, l'entête du document en remplissant toutes les information dans le dictionnaire.
Faites attention à bien lire les mots et chiffres correctement.
"""
            },
            {
                "type": "text",
                "text": """Vous êtes une IA spécialisée dans l'extraction d'informations de documents historiques manuscrits en français.
Dans un premier temps, vous lisez l'entête à deux niveau du tableau dans le document en reconnaissant les éléments sous la forme ("first level", "second level") :
```plaintext
[("Atelier", ""),
("Occupation", ""),
("Entrée", ""),
("Sortie", ""),
("Présence: Années", ""),
("Présence: Mois", ""),
("Présence: Jours", ""),
("Observations", ""),
("Divers", "")]
```
Contexte :
- Le document est une fiche d'entreprise textile manuscrite en français datant des années 1900.
- La colonne 'Atelier' peut être absente.
- Pour chaque ligne du tableau, créer une entrée dans le dictionnaire.
- Il est possible que certaines lignes du tableau ne contiennent qu'une observation. Dans ce cas, remplir uniquement la colonne 'Observations' d'une nouvelle entrée.
- Des lignes peuvent avoir des informations manquantes.
- Tu dois ecrire <bis> à la place de '"'.
- "Divers" contient des informations supplémentaires sur le montant de la gratifiaction par exemple (Francs et centimes).
Tâche :
En vous inspirant des exemples donnés, reconstruisez, s'il vous plait, le tableau en remplissant toutes les information dans le dictionnaire.
Faites attention à bien lire les mots et chiffres correctement. Ne confond pas '"' et '11'. Tu dois egalement faire attention à ne pas confondre une ligne est une annotation marginale.
"""
            },
            {
                "type": "text",
                "text": """Vous êtes une IA spécialisée dans l'extraction d'informations de documents historiques manuscrits en français.
Dans un premier temps, vous prenez connaissance des informations du document en essayant de reperer les informations marginales sous la forme ("first level", "second level") :
```plaintext
[("Annotation", "")]
```
Contexte :
- Le document est une fiche d'entreprise textile manuscrite en français datant des années 1900.
- Les annotations sont des informations supplémentaires sur le document.
- Les annotations peuvent être des notes, des remarques, des précisions, etc.
- Les annotations sont essentiellement manuscrites.
- Les annotations peuvent être sur plusieurs lignes.
Tâche :
En vous inspirant des exemples donnés, reconstruisez, s'il vous plait, la liste des annotations en remplissant toutes les information dans le dictionnaire.
Faites attention à bien lire les mots et chiffres correctement. Veillez également à ne pas mélanger les annotations.
"""
            },
            {
                "type": "text",
                "text": """Vous êtes un agent spécialisé dans l'adaptation des informations précédemment extraites.
En utilisant les exemples donnés, votre mission est de transformer les informations extraites en XML. Pour cela vous avez a votre disposition les tags suivants :
```xml
<Document>
<Nom>
<Genre>
<Statut>
<DateDeNaissance>
<LieuDeNaissance>
<Nationalité>
<Adresse>
<Ville>
<Table>
<Ligne>
<Atelier>
<Occupation>
<Entrée>
<Sortie>
<Présence>
<Années>
<Mois>
<Jours>
<Observations>
<Divers>
<Annotation>
```
Si une balise est vide alors ne pas l'inclure.
"""
            },
        ],
    }
]

def extract_xml(text):
    # Recherche d'un bloc XML entre des balises markdown ```xml ... ```
    match = re.search(r"```xml\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

for image_path in tqdm(image_files, desc="Processing images", unit="image"):

    torch.cuda.empty_cache()

    # Générer le chemin de sortie correspondant
    relative_path = image_path.relative_to(input_dir)
    output_subdir = output_dir / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / (image_path.stem + ".xml")

    # Construire le message
    messages = system_prompts.copy()
    # Change the position of the placeholder in the prompt
    pos = 4
    messages[pos]["content"][0]["image"] = str(image_path)

    # Préparer le texte d'entrée pour le modèle
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assistant_response = output_text[0].split("assistant\n", 1)[-1]

    xml_content = extract_xml(assistant_response)
    if not xml_content:
        print(f"[⚠️] Aucun bloc XML détecté pour : {relative_path}")
        continue

    # Sauvegarde
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Nettoyage GPU
    del generated_ids, generated_ids_trimmed, inputs, output_text, assistant_response, xml_content
    torch.cuda.empty_cache()

# Fin du traitement
end_time = time.time()
# Print time in hh:mm:ss
elapsed_time = end_time - start_time
days = int(elapsed_time // 86400)  # 86400 seconds in a day
time_part = time.strftime("%H:%M:%S", time.gmtime(elapsed_time % 86400))

elapsed_time_str = f"{days}d {time_part}"
print(f"Traitement terminé en {elapsed_time_str} !")
