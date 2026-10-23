"""Checks that every notebook-derived block in the backend is still IDENTICAL to the notebooks.

Each block in crowd_count.py / motion.py / features.py / risk_model.py / pipeline.py / visualization.py starts with a marker
        # --- notebook NN, cell M: <name> ----
The script re-extracts <name> from  ../notebooks/NN_*.ipynb  (top-level function / class / assignment, decorators included) and compares the source text.

    python verify_notebook_sync.py            # exit code 0 = everything identical
"""
import ast, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "..", "notebooks")
MODULES = ["crowd_count.py", "motion.py", "features.py", "risk_model.py", "pipeline.py", "visualization.py"]
MARK = re.compile(r"^# --- notebook (\d\d), cell (\d+): (\S+) -+$")


def notebook_defs(path):
    out = {}
    for c in json.load(open(path, encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code": continue
        src = "".join(c["source"])
        try: tree = ast.parse(src)
        except SyntaxError: continue
        lines = src.split("\n")
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                out.setdefault(n.name, "\n".join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1:n.end_lineno]))
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): out.setdefault(t.id, "\n".join(lines[n.lineno - 1:n.end_lineno]))
    return out


def main():
    nbs = {}
    for p in glob.glob(os.path.join(NB_DIR, "[0-9][0-9]_*.ipynb")): nbs[os.path.basename(p)[:2]] = notebook_defs(p)
    if not nbs: print(f"no notebooks found in {NB_DIR}"); return 2
    ok = bad = 0
    for mod in MODULES:
        text = open(os.path.join(HERE, mod), encoding="utf-8").read().split("\n"); i = 0
        while i < len(text):
            m = MARK.match(text[i])
            if not m: i += 1; continue
            nb, _cell, name = m.groups(); j = i + 1
            while j < len(text) and not text[j].startswith("# --- notebook ") and not text[j].startswith("# ====="): j += 1
            block = "\n".join(text[i + 1:j]).rstrip("\n")
            ref = nbs.get(nb, {}).get(name)
            if ref is None: print(f"  ?  {mod}: '{name}' not found in notebook {nb}"); bad += 1
            elif ref.rstrip("\n") != block: print(f"  X  {mod}: '{name}' differs from notebook {nb}"); bad += 1
            else: ok += 1
            i = j
    print(f"{ok} blocks identical, {bad} different / missing")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
