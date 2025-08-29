import nbformat

nb_path = 'nb/Sesame_CSM_(1B)_TTS.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

if 'widgets' in nb['metadata']:
    del nb['metadata']['widgets']

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Metadata cleaned successfully.")
