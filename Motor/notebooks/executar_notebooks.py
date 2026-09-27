"""Executa e salva os quatro notebooks com o Python que chamou este script.

Uso: python Motor/notebooks/executar_notebooks.py
Requer pandas, numpy, scipy, scikit-learn, matplotlib, nbformat, nbclient,
ipykernel e jupyter_client no mesmo ambiente.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    '01_orientacao_motorista.ipynb',
    '02_planejamento_energetico.ipynb',
    '03_analise_operacao.ipynb',
    '04_controle_automatico.ipynb',
]


def main():
    folder = Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='motor-jupyter-') as temporary:
        temporary = Path(temporary)
        kernel = temporary / 'kernels' / 'motor-local'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Motor local', 'language': 'python',
        }))
        os.environ['JUPYTER_PATH'] = str(temporary)
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temporary / 'runtime')
        os.environ['IPYTHONDIR'] = str(temporary / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(temporary / 'matplotlib')
        names = sys.argv[1:] or NOTEBOOKS
        if any(name not in NOTEBOOKS for name in names):
            raise SystemExit("Use apenas os nomes dos quatro notebooks desta pasta.")
        for name in names:
            path = folder / name
            print('EXECUTANDO', name, flush=True)
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(
                nb, timeout=180, kernel_name='motor-local', allow_errors=False,
                resources={'metadata': {'path': str(folder)}},
            )
            try:
                client.execute()
                nb.metadata['motor_executado_utc'] = datetime.now(timezone.utc).isoformat()
            finally:
                nbformat.write(nb, path)
            print('CONCLUÍDO', name, flush=True)


if __name__ == '__main__':
    main()
