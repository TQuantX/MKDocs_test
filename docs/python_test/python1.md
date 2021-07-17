# Terminal et IDE dans Mkdocs

## Introduction

Afin d'écrire un cours interactif utilisant Mkdocs, le besoin s'est fait sentir de pouvoir écrire directement des scripts en ligne :

- dans le navigateur
- sans iframe
- sans cookie
- sans inscription

!!! info "Solution"
    La solution existe et s'appelle [Pyodide](https://pyodide.org/en/stable/ "Pyodide, Python with the scientific stack, compiled to WebAssembly").
    
    Pyodide utilise WebAssembly pour faire le lien entre Python et Javascript et proposer un environnement permettant de manipuler le DOM avec Python, ou de manipuler Python avec Javascript.

??? warning "Un hic"
    La documentation quasi-absente ou réservée aux initiés du projet Pyodide...


## Prise en main

Je vous propose ici des commandes Markdown permettant de créer un terminal ainsi qu'un IDE grâce au **plugin macro**.

### Syntaxe Markdown

Vite vite ! Le résultat, histoire d'appâter le chaland.

!!! summary "La syntaxe"

    === "Terminal"
        ```markdown
        {% raw %}
        {{ terminal() }}
        {% endraw %}
        ```
        Cette commande crée un terminal vide. L'auto-complétion avec ++tab++ et le rappel de l'historique (avec ++ctrl+"R"++ ) sont possibles.

        {{ terminal () }}

    === "IDE vide"
        ```markdown
        {% raw %}
        {{ IDE() }}
        {% endraw %}
        ```
        Cette commande crée un IDE (~ Thonny) vide. La flèche permet de lancer le code tapé dans la zone de saisie (avec les numéros de ligne). La zone de saisie se redimensionne automatiquement et autorise l'auto-complétion de type _snippet_ avec ++tab++.

        {{IDE()}}

    === "IDE vertical"
        ```markdown
        {% raw %}
        {{ IDEv() }}
        {% endraw %}
        ```
        Cette commande crée un IDE vide, avec division verticale. L'engrenage permet de lancer le code tapé dans la zone de saisie (avec les numéros de ligne). La zone de saisie se redimensionne automatiquement et autorise l'auto-complétion de type snippet avec ++tab++.

        {{IDEv()}}

    === "IDE avec code"
        ```markdown
        {% raw %}
        {{ IDE('foo/bar/nom_de_fichier') }}
        {% endraw %}
        ```
        Cette commande charge le fichier `nom_de_fichier.py` dans un IDE. Le fichier doit être dans `docs/scripts/foo/bar/`. Ne pas oublier les guillemets.

        {{IDE('scripts/demo/demo1')}}

    === "IDE vertical avec code"
        ```markdown
        {% raw %}
        {{ IDEv('foo/bar/nom_de_fichier') }}
        {% endraw %}
        ```
        Cette commande charge le fichier `nom_de_fichier` dans un IDE avec division verticale. Le fichier doit être dans `docs/scripts/foo/bar/`.       

        {{IDEv('scripts/demo/demo1')}}
 

??? warning "Détails techniques"

    Tous les IDE et les terminaux partagent le même _namespace_. On peut donc accéder à n'importe quelle fonction créée dans n'importe quel IDE ou terminal. 
    
    **C'est un comportement voulu qui a des avantages et des inconvénients.**

!!! done "Amélioration notable"

    ~~Pour que les IDE fonctionnent, il faut absolument indiquer `{% raw %} {{ IDE('nom_de_fichier', -1) }} {% endraw %}` sur le dernier IDE de la page.~~

    Une solution plus élégante modifiant le template Jinja2 `my_theme_customizations/main.html` est maintenant utilisée. Plus besoin d'indiquer le dernier IDE !


### Exemples

L'exemple ci-dessous, obtenu avec `#!markdown {% raw %} {{ IDEv('scripts/exo2') }} {% endraw %}`. N'hésitez pas à modifier le code pour calculer la moyenne, l'écart-type, afficher cela dans le terminal etc.

{{IDEv('scripts/exo2')}}

L'exemple ci-dessous a été obtenu avec `#!markdown {% raw %} {{ IDE('scripts/algo_glouton') }} {% endraw %}`.

{{IDE('scripts/algo_glouton')}}


## Installation

L'installation demande

- de modifier :
  
    - le fichier YML `mkdocs.yml` ;
    - le fichier de macro `main.py` ;

- d'ajouter :

    - un dossier `#!bash my_theme_customizations/` à la racine du projet Mkdocs ;
    - un template HTML `#!bash my_theme_customizations/main.html` ;
    - un fichier CSS `#!bash docs/xtra/stylesheets/pyoditeur.css` ;
    - deux fichiers Javascript `#!bash docs/xtra/javascripts/interpreter.js` et `#!bash my_theme_customizations/js/ide.js` ;

### Fichier YML `mkdocs.yml`

Ajoutez les lignes surlignées dans votre fichier mkdocs.yml .

```yaml hl_lines="7 16 19"
    --8<--- "docs/scripts/mkdocs.yml"
```

### Fichier macro Python `main.py`

À votre fichier `main.py`, ajoutez les lignes du fichier [`main.py`](https://gitlab.com/bouillotvincent/pyodide-mkdocs/-/raw/main/docs/scripts/main.py "main.py sur Gitlab").


### Création du dossier `custom_dir`

**N'oubliez pas de créer le dossier `#!bash my_theme_customizations/` à la racine du projet Mkdocs**.

Dans ce dossier, ajoutez le template Jinja `#!bash main.html` :

```jinja
    --8<--- "my_theme_customizations/main.html"
```

### Fichier CSS `pyoditeur.css`

Afin de coller au thème du site, recopiez et ajoutez le fichier [`pyoditeur.css`](https://gitlab.com/bouillotvincent/pyodide-mkdocs/-/raw/main/docs/xtra/stylesheets/pyoditeur.css "Pyoditeur CSS sur Gitlab") au dossier `docs/xtra/stylesheets/`.

### Fichiers javascripts `interpreter.js` et `ide.js`

Deux fichiers Javascript [`interpreter.js`](https://gitlab.com/bouillotvincent/pyodide-mkdocs/-/raw/main/docs/xtra/javascripts/interpreter.js "interpreter JS sur Gitlab ") et [`ide.js`](https://gitlab.com/bouillotvincent/pyodide-mkdocs/-/raw/main/my_theme_customizations/js/ide.js "ide JS sur Gitlab ") sont nécessaires :

- `interpreter.js` doit être placé dans le dossier : `docs/xtra/javascripts/` ;
- `ide.js` doit être placé dans le dossier : `my_theme_customizations/js/ide.js`.

**Et c'est tout !**
