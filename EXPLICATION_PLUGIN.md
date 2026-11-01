# Explication du code du plugin DSPy

Ce document explique l'implementation actuelle du plugin, et pas seulement
comment remplir un flow Kestra.

L'idee principale : **Java gere l'integration Kestra et prepare les donnees ;
Python execute les operations qui dependent de DSPy.**

## 1. A quoi sert ce plugin ?

Le plugin expose deux types de taches :

| Tache | Role |
|---|---|
| `io.kestra.plugin.dspy.Predict` | Executer un programme DSPy sur un dataset. |
| `io.kestra.plugin.dspy.Optimize` | Optimiser un programme DSPy, l'evaluer et sauvegarder son etat. |

Le programme peut etre defini par une signature declarative ou par un fichier
Python fourni par l'utilisateur.

Le plugin ne choisit pas de domaine metier. Il n'impose ni classification,
ni traduction, ni pipeline RAG. Les champs d'entree et de sortie viennent du
programme utilisateur.

**Generique ne signifie pas que tout est interchangeable :** l'implementation
actuelle propose `Predict` et `ChainOfThought` pour les signatures declaratives,
et `BootstrapFewShot` pour l'optimisation. Un module personnalise peut avoir une
autre structure DSPy, mais le choix d'optimiseur reste actuellement limite.

## 2. Les termes DSPy importants

| Terme | Signification dans ce plugin |
|---|---|
| Signature | Declaration des champs d'entree et de sortie, par exemple `source -> resultat`. |
| Module | Programme Python construit avec DSPy. Il peut contenir plusieurs predictors. |
| Predictor | Composant qui demande au LLM de produire les sorties d'une signature. |
| LM / LLM | Modele de langage utilise pour les appels au fournisseur. |
| Student | Programme et modele dont on veut ameliorer les predictions. |
| Teacher | Programme et modele utilises pour produire les demonstrations candidates. |
| Train / trainset | Exemples que l'optimiseur peut utiliser pour construire les demonstrations. |
| Validation | Exemples reserves a l'evaluation du programme obtenu. |
| Metric | Fonction qui attribue un score a une prediction. |
| Demo | Exemple ajoute au programme pour guider les predictions suivantes. |
| Artifact | Fichier JSON contenant la definition et l'etat sauvegarde du programme. |
| Manifest | Fichier JSON de suivi : versions, parametres, hashes et resultat d'evaluation. |

Avec `BootstrapFewShot`, **optimiser ne signifie pas reentrainer les poids du
LLM**. L'optimiseur compile un programme en lui associant des demonstrations.

Les mots `train` et `validation` designent des roles techniques. Ils ne
designent pas des fichiers d'exemple ou des champs metier imposes.

## 3. Vue d'ensemble du parcours

```text
Flow YAML Kestra
    |
    v
Predict.run() ou Optimize.run()
    |
    v
AbstractDspy : proprietes, chemins, lancement du TaskRunner
    |
    v
DspyInputs + DspyDataset : recuperation, lecture et validation Java
    |
    v
Configuration JSON stockee, avec les enregistrements prepares
    |
    v
TaskRunner : execution de dspy_task.py
    |
    v
DSPy : construction du programme, appels LLM, prediction ou optimisation
    |
    v
Fichiers de sortie recuperes par Kestra
    |
    +--> Predict : URI des predictions et du resume
    |
    +--> Optimize : URI de l'artefact + manifeste construit en Java
```

Le script Python n'est donc pas une application autonome qui lit les proprietes
du flow. Il recoit une configuration preparee par les classes Java.

## 4. Les classes Java

Les classes se trouvent dans
[`src/main/java/io/kestra/plugin/dspy`](src/main/java/io/kestra/plugin/dspy).

### 4.1. `AbstractDspy` : le socle commun

Fichier : [AbstractDspy.java](src/main/java/io/kestra/plugin/dspy/AbstractDspy.java).

Cette classe herite d'`AbstractExecScript`, fourni par les plugins de scripts
Kestra. Elle reutilise les TaskRunners au lieu d'implementer un moteur
d'execution de processus ou de conteneurs.

Elle declare les proprietes communes :

| Propriete | Utilisation |
|---|---|
| `signature` | Definition declarative du programme. |
| `instructions` | Instructions associees a la signature. |
| `predictor` | `PREDICT` ou `CHAIN_OF_THOUGHT`. |
| `programUri` | URI du fichier Python personnalise. |
| `programFactory` | Nom de la fonction qui construit le module ; `build_program` par defaut. |
| `filePaths` | Chemins relatifs des fichiers dans le runner, par role. |
| `containerImage` | Image contenant Python et les dependances DSPy. |
| `pythonExecutable` | Executable lance dans le runner. |
| `maxLlmCalls` | Budget d'appels aux modeles fournis par le plugin. |
| `requestTimeoutSeconds` | Timeout d'une requete au modele. |
| `maxTokens` | Limite de tokens de sortie par requete. |

Une `Property<T>` peut contenir une valeur ou une expression Kestra.
`runContext.render(...)` resout la propriete avant utilisation. Par exemple,
une URI peut venir de `{{ inputs.dataset }}` plutot que d'une valeur fixe.

Les principales methodes sont :

| Methode | Ce qu'elle fait |
|---|---|
| `configuration(...)` | Construit les parametres communs et verifie les limites positives. |
| `programFiles(...)` | Prepare le fichier personnalise et interdit son association avec une signature ou des instructions declaratives. |
| `filePaths(...)` | Fusionne les chemins par defaut avec les substitutions utilisateur. |
| `runnerPath(...)` | Normalise et valide un chemin relatif, sans glob ni sortie du repertoire de travail. |
| `required(...)` | Verifie qu'une chaine obligatoire est presente et non blanche. |
| `uri(...)` | Verifie la propriete et son format avec `URIFetcher`. |
| `execute(...)` | Prepare les fichiers, lance le runner et verifie les sorties obligatoires. |
| `outputUri(...)` | Retrouve l'URI d'un fichier capture, ou echoue s'il manque. |

`runnerPath(...)` recherche directement les caracteres interdits. Il n'utilise
pas de regex pour cette recherche.

Dans `execute(...)`, le plugin :

1. Recupere les `inputFiles` supplementaires.
2. Refuse les collisions avec les chemins reserves.
3. Appelle `DspyInputs.prepare(...)`.
4. Prepare le script embarque et la configuration JSON.
5. Configure l'environnement et les fichiers a capturer.
6. Lance `python3 dspy_task.py dspy_config.json`, avec l'executable configurable.
7. Verifie le code de sortie et la presence des fichiers attendus.

**Pourquoi stocker la configuration avant de la transmettre ?** Elle contient
les instructions utilisateur et les metadonnees. Les donnees restent dans les
fichiers JSONL stockes. Ces fichiers ne sont pas rendus comme des expressions
Kestra : une chaine telle que `{{ ceci est une donnee }}` reste litterale.

### 4.2. `DspyDataset` : lecture et validation JSONL

Fichier : [DspyDataset.java](src/main/java/io/kestra/plugin/dspy/DspyDataset.java).

C'est un `record` Java qui regroupe les informations obtenues apres lecture :

| Composant | Contenu |
|---|---|
| `ids` | Identifiants uniques des lignes. |
| `referenceFields` | Ensembles de champs observes dans les references. |
| `sha256` | Hash du fichier source. |

`read(...)` parcourt le fichier ligne par ligne. Chaque ligne doit contenir un
objet JSON valide, sans second document JSON apres lui.

La classe verifie notamment :

- `schemaVersion` doit etre l'entier `1`.
- `id` doit etre une chaine non vide, unique dans le dataset.
- `inputs` doit etre un objet JSON.
- `reference`, si present, doit etre un objet JSON.
- En mode supervise, `reference` doit etre present et non vide.
- Les champs d'`inputs` et de `reference` ne doivent pas se chevaucher.
- Les nombres flottants doivent etre finis.
- Le dataset ne doit pas etre vide.

`metadata()` retourne seulement le hash et le nombre de lignes.
`digest(...)` calcule le SHA-256 en parcourant les octets du fichier.

Java ne conserve pas les enregistrements. Les identifiants restent en memoire
pour detecter les doublons. `maxDatasetBytes` limite chaque fichier a 64 MiB et
`maxDatasetRows` a 10000 lignes par defaut. `copy(...)` controle les octets pendant
la copie ; `read(...)` controle la taille et le nombre de lignes.

### 4.3. `DspyInputs` : preparation des entrees

Fichier : [DspyInputs.java](src/main/java/io/kestra/plugin/dspy/DspyInputs.java).

`prepare(...)` recupere les fichiers via `URIFetcher`, puis les copie dans des
fichiers temporaires. La validation et le hash portent ainsi sur les copies
qui seront transmises au runner.

Sans artefact existant, `programDescriptor(...)` construit une description :

- Mode declaratif : type de programme, signature, predictor et instructions.
- Mode Python : type de programme, nom de la factory et hash du code source.

Pour `Optimize`, `prepareOptimization(...)` :

1. Lit les datasets d'entrainement et de validation.
2. Verifie que leurs identifiants sont disjoints.
3. Avec la metrique par defaut, exige les memes champs de reference partout.
4. Ajoute seulement les metadonnees a la configuration.

Pour `Predict`, la classe valide le fichier et prepare ses metadonnees. Avec
`evaluate: true`, elle exige des references non vides et coherentes.

Enfin, elle stocke les copies des fichiers et retourne la correspondance entre
leurs chemins dans le runner et leurs URI Kestra. Les fichiers source restent
disponibles dans le runner : Python les lit via les chemins configures.

### 4.4. `Predict` : executer un programme

Fichier : [Predict.java](src/main/java/io/kestra/plugin/dspy/Predict.java).

Cette classe implemente `RunnableTask<Predict.Output>`. Kestra appelle sa
methode `run(...)`.

Deux modes sont possibles :

| Mode | Informations necessaires |
|---|---|
| Prediction directe | `datasetUri`, `model` et soit `signature`, soit `programUri`. |
| Prediction depuis un artefact | `datasetUri` et `artifactUri` ; `model` peut remplacer celui de l'artefact. |

Un artefact de programme personnalise necessite aussi son fichier source
original via `programUri`.

`run(...)` valide le choix du mode, prepare les chemins et demande au runner
de produire les predictions et le resume.

L'objet `Output` contient :

- `predictionsUri` : fichier JSONL des predictions.
- `summaryUri` : fichier JSON avec le modele, le nombre de lignes, la duree
  de prediction, les appels aux modeles et la version de l'artefact.

### 4.5. `Optimize` : construire un candidat

Fichier : [Optimize.java](src/main/java/io/kestra/plugin/dspy/Optimize.java).

Cette classe implemente `RunnableTask<Optimize.Output>`.

Elle ajoute notamment `trainsetUri`, `validationUri`, `studentModel`,
`teacherModel`, `optimizer`, `seed`, les limites de demonstrations,
`metricFunction` et `metricThreshold`.

La factory et la metrique personnalisee ne sont pas appelees en Java :
Java prepare leurs noms, puis Python les cherche dans le module utilisateur.

La methode `run(...)` :

1. Construit et valide les parametres d'optimisation.
2. Exige une signature ou un programme personnalise.
3. Verifie qu'une metrique personnalisee dispose d'un `programUri`.
4. Lance le runner pour obtenir le candidat et le resultat technique.
5. Appelle `DspyManifest.write(...)`.
6. Retourne `candidateArtifactUri` et `manifestUri`.

`metricThreshold` doit etre dans l'intervalle `]0, 1]`. Les limites de
demonstrations doivent etre strictement positives dans cette implementation.

### 4.6. `DspyManifest` : tracer le resultat

Fichier : [DspyManifest.java](src/main/java/io/kestra/plugin/dspy/DspyManifest.java).

Cette classe lit l'artefact produit par Python et son fichier de resultat.
Elle les combine avec les parametres Java et les metadonnees des datasets.

Le manifeste contient notamment :

- La definition du programme et sa version d'artefact.
- Le SHA-256 du fichier d'artefact.
- Les versions Python, DSPy et des dependances.
- Les modeles student et teacher.
- L'optimiseur, le seed, les seuils et les limites.
- Les hashes et tailles des datasets.
- Le score moyen sur la validation.
- Les compteurs d'appels aux modeles.

Un champ technique requis manquant provoque une erreur. Le manifeste est
stocke par Kestra ; il n'est pas le fichier recharge par `Predict`.

### 4.7. Les annotations Java

`@SuperBuilder`, `@Builder`, `@Getter` et les autres annotations Lombok
generent les builders, accesseurs et methodes usuelles.

`@Schema` fournit la documentation des taches et proprietes.
`@Plugin` associe des exemples de flows au plugin.
`@NotNull` exprime les contraintes de presence, completees par les controles
effectues pendant `run(...)`.

## 5. Le script Python, fonction par fonction

Fichier : [dspy_task.py](src/main/resources/scripts/dspy_task.py).

Python reste necessaire parce que DSPy est utilise via son API Python.
La validation de l'enveloppe JSONL est faite en Java. Python lit les fichiers
valides et verifie ce qui depend de la signature DSPy ou du module.

### 5.1. `main(...)` : point d'entree

`main(...)` charge la configuration JSON preparee par Java, verifie la version
DSPy attendue, initialise le budget et configure DSPy.

`responseFormat` choisit `ChatAdapter` ou `JSONAdapter`, sans repli automatique
apres une erreur de parsing. Le format est conserve dans l'artefact et reutilise
par `Predict`, sauf surcharge explicite. Les anciens artefacts utilisent CHAT.
Le script desactive l'historique DSPy et limite certains logs des bibliotheques.

Il appelle ensuite `predict(config)` ou `optimize(config)` selon `operation`.

### 5.2. `DeclarativeProgram` : transformer une signature en module

Cette classe herite de `dspy.Module`.

Son constructeur cree une `dspy.Signature`, applique les instructions
eventuelles et choisit `dspy.Predict` ou `dspy.ChainOfThought`.

`forward(**inputs)` transmet les champs d'entree au predictor.
DSPy appelle cette methode lorsque le programme est execute.

### 5.3. `load_user_module(...)` et `build_program(...)`

`load_user_module(path)` importe le fichier Python dont le chemin a ete
transmis par Java. Sans fichier personnalise, il retourne `None`.

`build_program(descriptor, module)` :

- Construit `DeclarativeProgram` pour une definition declarative.
- Pour une definition personnalisee, verifie le hash du fichier source.
- Cherche la factory configuree et l'appelle sans arguments.
- Verifie que le resultat est un `dspy.Module`.

La comparaison du hash permet de refuser un artefact personnalise si le code
fourni n'est plus celui qui a servi a le construire.

### 5.4. `validate_fields(...)` et `examples(...)`

`validate_fields(...)` exige des noms de champs qui soient des identifiants
Python publics, sans prefixe `_`.

Pour une signature declarative, il compare exactement les champs d'entree
du dataset avec ceux de la signature. Si une reference non vide existe, ses
champs doivent correspondre aux sorties declarees.

Pour un module personnalise, le plugin ne peut pas deduire une signature
unique de toutes ses operations : ce controle declaratif n'est pas applique.

`examples(...)` construit des `dspy.Example` avec les entrees et references.
L'appel `with_inputs(...)` indique a DSPy quels champs sont les entrees :
les references ne sont pas transmises comme arguments d'entree du programme.

### 5.5. `make_metric(...)` : evaluer les predictions

Sans `metricFunction`, la metrique compare tous les champs de reference.

Pour les chaines, `normalized(...)` applique la normalisation Unicode NFKC,
le case folding et la normalisation des espaces. Pour les autres valeurs JSON,
il compare leur representation JSON canonique.

La metrique par defaut retourne `1.0` si tous les champs correspondent,
sinon `0.0`. Des sorties supplementaires, comme un raisonnement, sont acceptees ;
l'absence d'un champ de reference provoque une erreur.

Avec une metrique personnalisee, le plugin appelle la fonction du module avec
`(example, prediction, trace)`. `trace` correspond aux informations d'execution
transmises pendant le bootstrap ; il peut etre `None`.

Dans les deux cas, le score doit etre un booleen ou un nombre fini entre `0` et
`1`. Une erreur n'est pas convertie silencieusement en score nul.

### 5.6. `Budget`, `BudgetLM` et `make_lm(...)`

`Budget` compte les appels par nom de modele et impose une limite totale.
`BudgetLM` herite de `dspy.LM` et consomme une unite avant chaque tentative.
Un verrou protege le compteur si un programme lance des appels concurrents.

Le budget est partage au niveau du processus pour couvrir les copies de LM
effectuees pendant le bootstrap. Un appel tente compte meme s'il echoue ensuite.

`make_lm(...)` configure notamment :

- Le fournisseur/modele choisi par l'utilisateur.
- Le moteur LiteLLM.
- Le timeout et la limite de tokens.
- L'absence de cache et de retries internes au fournisseur.
- `maxRetries`, par defaut 2 retries geres par le plugin pour les erreurs
  temporaires, avec attente exponentielle plafonnee a 8 secondes.
- Les variables d'environnement propres au student ou au teacher.

Le budget ne couvre pas les appels externes que le code utilisateur effectuerait
independamment de ces LM.

Le budget couvre une tentative de tache. Une reprise par Kestra recommence le
lot avec un nouveau budget ; un timeout fournisseur peut avoir ete facture meme
si sa reponse n'a pas ete recue.

### 5.7. `optimize(...)` : le coeur DSPy de l'optimisation

Le debut de la fonction ressemble a ceci :

```python
train = list(read_dataset(files["train"]))
validation = list(read_dataset(files["validation"]))
descriptor = config["program"]
validate_fields(train + validation, descriptor)
module = load_user_module(files.get("program"))
```

Les fichiers ont deja ete bornes et valides par Java. DSPy demande des listes
pour l'optimisation :

| Instruction | Explication |
|---|---|
| `read_dataset(files["train"])` | Lit le fichier d'entrainement valide et borne. |
| `read_dataset(files["validation"])` | Lit le fichier de validation valide et borne. |
| `config["program"]` | Recupere la definition du programme utilisateur. |
| `validate_fields(...)` | Verifie la compatibilite avec les champs DSPy. |
| `load_user_module(...)` | Importe le module personnalise s'il existe. |

Ensuite, la fonction melange les exemples d'entrainement avec le `seed`, construit
les LM et deux instances du programme : student et teacher.

Le teacher recoit explicitement son LM. Il est passe a
`BootstrapFewShot.compile(...)`, avec les exemples d'entrainement.

Le candidat obtenu est execute avec le student sur la validation. La moyenne
des scores devient `validationScore`.

Enfin, Python sauvegarde l'artefact et un resultat technique que Java utilisera
pour construire le manifeste.

L'implementation fixe actuellement `max_rounds=1` et `max_errors=1`.
Le seed rend l'ordre des donnees reproductible, mais ne garantit pas que
les reponses d'un fournisseur distant soient deterministes.

### 5.8. `predict(...)` : produire les sorties

La fonction lit les enregistrements depuis le fichier prepare par Java. Un
premier passage valide tous les champs DSPy avant tout appel LLM. Un second
passage produit les predictions ligne par ligne, sans liste de dataset.
Avec `evaluate: true`, elle calcule aussi un score moyen sans stocker les scores
individuels ; Java expose ce resultat dans `Predict.Output.evaluationScore`.

Avec un artefact, elle verifie sa version de schema, sa version DSPy et son hash
logique, reconstruit le programme puis recharge son etat avec `load_state(...)`.

Sans artefact, elle construit directement le programme et calcule une version
d'artefact en memoire. Cela permet aux predictions de porter une version meme
en mode direct ; ce mode n'expose pas de fichier candidat.

Apres avoir attache le LM, elle execute chaque ligne :

```python
prediction = program(**row["inputs"])
```

`**` transforme l'objet `inputs` en arguments nommes Python. Par exemple,
`{"source": "Texte"}` devient un appel avec `source="Texte"`.

`prediction_outputs(...)` convertit les sorties DSPy en valeurs JSON et exige
au moins un champ de sortie.

Les predictions sont ecrites dans un fichier temporaire derive du chemin de
sortie, avec le suffixe `.partial`. Le fichier final n'est remplace qu'une fois
le lot termine. Le resume est ensuite ecrit separement.

### 5.9. Artefacts, hashes et erreurs

`canonical(...)` produit un JSON UTF-8 avec les cles triees et sans valeurs
numeriques non finies.

`artifact_version(...)` calcule le hash du contenu logique de l'artefact,
en excluant le champ `artifactVersion` lui-meme.

`make_artifact(...)` detache les LM avec `set_lm(None)`, puis sauvegarde la
definition du programme, les versions, le modele student et `dump_state()`.
`validate_state(...)` refuse un etat de predictor contenant un LM serialise.

Il faut distinguer :

| Champ | Ce qu'il identifie |
|---|---|
| `artifactVersion` | Le contenu JSON logique de l'artefact. |
| `artifactSha256` | Les octets exacts du fichier d'artefact, dans le manifeste. |

Un hash detecte une modification, mais **ne prouve pas l'identite de l'auteur** :
ce n'est pas une signature cryptographique.

`report_error(...)` est installe comme `sys.excepthook`. Il affiche les erreurs
de contrat ou de budget. `ProviderError` expose uniquement la classe d'erreur
DSPy et le statut HTTP disponible. Pour les autres exceptions, le detail est
remplace par un message generique afin de ne pas afficher les donnees fournisseur.
Ses trois arguments correspondent a l'API Python de ce callback.

## 6. Donnees, chemins et configuration : trois choses differentes

Exemple de ligne JSONL :

```json
{"schemaVersion":1,"id":"exemple-1","inputs":{"source":"Texte fourni"},"reference":{"resultat":"Resultat attendu"}}
```

`source` et `resultat` sont ici de simples exemples de champs utilisateur,
coherents avec une signature `source -> resultat`. Ce ne sont pas des noms
reserves par le plugin.

Il faut distinguer :

| Element | Exemple | Role |
|---|---|---|
| URI source | `datasetUri: "{{ inputs.dataset }}"` | Indique a Java ou recuperer le fichier. |
| Chemin runner | `filePaths.dataset: inputs/lot.jsonl` | Indique ou placer sa copie dans le runner. |
| Metadonnees | `config["dataset"]` | Contient seulement le hash et le nombre de lignes. |
| Donnees | Fichier JSONL au chemin `config["files"]["dataset"]` | Lu ligne par ligne par Python. |
| Definition du programme | `config["program"]` | Decrit la signature ou la factory, pas le dataset. |

Pour `Predict`, `filePaths` accepte `program`, `dataset`, `artifact`,
`predictions` et `summary`.

Pour `Optimize`, il accepte `program`, `train`, `validation`, `candidate`
et `result`.

Les anciens noms de fichiers restent les valeurs par defaut. Ils ne sont pas
codes en dur dans Python pour les datasets, le programme ou les artefacts.
Les noms du script embarque et de sa configuration sont reserves par Java.

`config["train"]` contient des metadonnees pour le manifeste ;
les exemples eux-memes restent dans le fichier `config["files"]["train"]`.

## 7. Runtime, dependances et limites

[`build.gradle`](build.gradle) configure un module Java cible JDK 25, les
dependances Kestra, Lombok, les tests et la construction du JAR avec Shadow.

[`python/Dockerfile`](python/Dockerfile) construit le runtime Python et installe
les dependances de [`python/requirements.lock`](python/requirements.lock).
Le script exige actuellement DSPy `3.4.1`.

Le runner par defaut est Docker. Le runner Process suppose que Python et les
dependances compatibles sont deja installes sur le worker.

Les identifiants fournisseur passent par l'environnement, par exemple
`OPENAI_API_KEY`. Les variables `DSPY_STUDENT_API_KEY`,
`DSPY_TEACHER_API_KEY`, `DSPY_STUDENT_API_BASE` et `DSPY_TEACHER_API_BASE`
permettent des connexions distinctes.

**Le programme personnalise est du code de confiance, pas du code sandboxe.**
Il peut acceder aux fichiers et a l'environnement accessibles au runner.
Les protections du plugin ne controlent pas ses propres logs ou appels externes.

Les artefacts peuvent contenir des demonstrations issues des datasets.
La configuration stockee contient les enregistrements prepares. Ces fichiers
doivent donc etre traites comme les donnees source, pas comme des fichiers
automatiquement depourvus d'informations sensibles.

## 8. Comment les tests correspondent au code

| Fichier | Responsabilites couvertes |
|---|---|
| [DspyTest.java](src/test/java/io/kestra/plugin/dspy/DspyTest.java) | Contrat JSONL, references, donnees preparees, hashes, chemins invalides et validations avant lancement. |
| [DspyDockerTest.java](src/test/java/io/kestra/plugin/dspy/DspyDockerTest.java) | Parcours Optimize puis Predict via Kestra et Docker, teacher explicite, manifeste, chemins personnalises et conservation des chaines litterales. |
| [test_dspy_task.py](python/test_dspy_task.py) | Budget, metriques, trace, signatures typees, modules personnalises, integrite des artefacts et prediction sans lecture JSONL. |

Les tests d'integration utilisent un serveur LLM simule. Ils ne necessitent
pas de cle fournisseur reelle.

Les tests Docker Java sont actives par `DSPY_DOCKER_TESTS=true`.
Les commandes de verification documentees pour ce projet sont :

```bash
docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
DSPY_DOCKER_TESTS=true ./gradlew test lintPluginDocs shadowJar
```

Les tests Python peuvent etre executes avec le meme runtime :

```bash
docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  plugin-dspy:3.4.1 \
  python -m unittest discover -s python -p 'test_dspy_task.py'
```

## 9. Ordre conseille pour lire les sources

Commencer par `Predict.run(...)` et `Optimize.run(...)` pour comprendre
l'interface exposee a Kestra.

Lire ensuite `AbstractDspy.execute(...)`, puis `DspyInputs.prepare(...)` et
`DspyDataset.read(...)` pour suivre la preparation Java.

Terminer par `main(...)`, `predict(...)` et `optimize(...)` dans le script
Python, puis `DspyManifest.write(...)` pour comprendre les sorties.

La separation a retenir est :
**Kestra et preparation des donnees en Java ; execution DSPy en Python ;
programme metier fourni par l'utilisateur.**
