"""
Directive 10 - Part B.3: build the blind HITL annotation package (no ratings yet).

Regenerates attributions from the anchored XAI pipeline (same champions + params as the committed
run) for the 60 selected docs, forms the top-10 method sets + a seed-42 RANDOM set per (doc,
family), blind-shuffles them into Set A/B/C(/D), and writes results/hitl/:
  - key.json        the method key (per set_uid: doc_uid, corpus, family, method, label,
                    positions[], tokens[]) + docs (doc_uid -> corpus, hash, y_true, text).
                    The annotator must NOT open this until human_ratings.csv is committed.
  - annotate.html   single-file OFFLINE tool: email + blind sets, per-token checkboxes, one
                    0/1/2 holistic radio per set, progress counter, autosave, Export CSV ->
                    human_ratings.csv (doc_uid, set_uid, token_idx, token, is_cue, holistic).
  - blind_sheets.csv  the same blind content flat (no method), as a fallback.

Blinding (per the pre-registration): RANDOM positions and the A/B/C(/D) shuffle are drawn with
per-(doc,family) seed-42-anchored generators, so the package is deterministic and independent of
processing order. Asserts every doc_uid is inside the committed TEST explained_ids (surface S1).

Run: python -m src.experiments.d10_package
"""

from __future__ import annotations

import csv as _csv
import json
from pathlib import Path

import numpy as np

try:
    from src.xai import champions as C, explainers, textops
    from src.splits import load_splits
except ImportError:  # pragma: no cover
    from xai import champions as C, explainers, textops
    from splits import load_splits

_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _ROOT / "results" / "xai"
HITL_DIR = _ROOT / "results" / "hitl"
SEED = 42
TOPN = 10
FAMILY_CODE = {"tfidf": 0, "transformer": 1}
# HITL method sets per family (RANDOM appended after); IG is transformer-only.
FAMILY_METHODS = {"tfidf": ["shap", "lime"], "transformer": ["shap", "lime", "ig"]}
UNIT_OF = {("d1_sb", "tfidf"): "tfidf_d1_sb", ("d2", "tfidf"): "tfidf_d2",
           ("d3_en", "tfidf"): "tfidf_d3_en", ("d3_en", "transformer"): "transformer_d3_en"}
LABELS = ["A", "B", "C", "D", "E"]


def _seed(h: str, *extra: int) -> np.random.Generator:
    return np.random.default_rng([SEED, int(h[:8], 16), *extra])


def build() -> dict:
    sel = json.loads((HITL_DIR / "selected_ids.json").read_text(encoding="utf-8"))
    champ_cache: dict[str, object] = {}
    text_cache: dict[str, dict] = {}
    commit_hashes: dict[str, set] = {}   # unit -> committed explained_ids hashes (S1 check)

    key = {"directive": "D10", "seed": SEED, "sets": {}, "docs": {}}
    blind_rows: list[dict] = []
    data_docs: list[dict] = []           # for annotate.html (NO methods)
    n_sets = n_tokens = 0

    for corpus, cinfo in sel["corpora"].items():
        exc = cinfo["explained_corpus"]
        if exc not in text_cache:
            te = load_splits(exc)["test"].reset_index(drop=True)
            text_cache[exc] = {h: t for h, t in zip(te["_hash"].to_numpy(), te["text"].tolist())}
        for u in cinfo["source_units"]:
            commit_hashes.setdefault(u, set(
                json.loads((XAI_DIR / f"explained_ids_{u}.json").read_text(encoding="utf-8"))["hashes"]))

        for doc in cinfo["docs"]:
            h, doc_uid = doc["hash"], doc["doc_uid"]
            # S1: every doc is inside the committed TEST explained_ids of every source unit
            for u in cinfo["source_units"]:
                assert h in commit_hashes[u], f"{doc_uid} not in committed explained_ids {u}"
            text = text_cache[exc][h]
            words = textops.words(text)
            nw = len(words)
            key["docs"][doc_uid] = {"corpus": corpus, "hash": h, "y_true": doc["y_true"],
                                    "stratum": doc["stratum"], "n_word_tokens": nw, "text": text}
            groups = []
            for gi, family in enumerate(doc["families"]):
                unit = UNIT_OF[(corpus, family)]
                if unit not in champ_cache:
                    champ_cache[unit] = C.build_champion(unit)
                champ = champ_cache[unit]
                pred = int(float(champ.proba([text])[0]) >= 0.5)
                assert pred == 1, f"{doc_uid} {family}: champion pred != phishing"

                built = []  # (method, positions)
                for method in FAMILY_METHODS[family]:
                    attr = explainers.attribute(champ, family, method, text)
                    built.append((method, textops.top_k_positions(attr, 1, min(TOPN, nw))))
                rpos = _seed(h, FAMILY_CODE[family]).permutation(nw)[:min(TOPN, nw)].tolist()
                built.append(("random", rpos))

                order = _seed(h, FAMILY_CODE[family], 99).permutation(len(built)).tolist()
                ui_sets = []
                for li, si in enumerate(order):
                    method, pos = built[si]
                    pos = [int(p) for p in pos]
                    label = LABELS[li]
                    set_uid = f"{doc_uid}__{family}__{label}"
                    toks = [words[p] for p in pos]
                    key["sets"][set_uid] = {"doc_uid": doc_uid, "corpus": corpus, "family": family,
                                            "method": method, "label": label, "positions": pos,
                                            "tokens": toks}
                    for ti, (p, tk) in enumerate(zip(pos, toks)):
                        blind_rows.append({"doc_uid": doc_uid, "corpus": corpus, "set_uid": set_uid,
                                           "set_label": label, "token_idx": ti, "token": tk})
                    ui_sets.append({"set_uid": set_uid, "label": label, "tokens": toks})
                    n_sets += 1
                    n_tokens += len(toks)
                groups.append({"group_idx": gi, "sets": ui_sets})
            data_docs.append({"doc_uid": doc_uid, "corpus": corpus, "stratum": doc["stratum"],
                              "text": text, "groups": groups})

    HITL_DIR.mkdir(parents=True, exist_ok=True)
    (HITL_DIR / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    with open(HITL_DIR / "blind_sheets.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["doc_uid", "corpus", "set_uid", "set_label",
                                           "token_idx", "token", "is_cue", "holistic"])
        w.writeheader()
        for r in blind_rows:
            w.writerow({**r, "is_cue": "", "holistic": ""})

    data = {"n_docs": len(data_docs), "n_sets": n_sets, "n_tokens": n_tokens, "docs": data_docs}
    (HITL_DIR / "annotate.html").write_text(_render_html(data), encoding="utf-8")

    print(f"  docs={data['n_docs']}  sets={n_sets}  tokens={n_tokens}", flush=True)
    print(f"  wrote {HITL_DIR/'key.json'}, {HITL_DIR/'annotate.html'}, {HITL_DIR/'blind_sheets.csv'}",
          flush=True)
    return data


def _render_html(data: dict) -> str:
    # embed data JSON safely inside a non-executing <script type=application/json>
    payload = (json.dumps(data, ensure_ascii=False)
               .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    return _HTML_TEMPLATE.replace("/*__DATA__*/", payload)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HITL Plausibility Annotation (Directive 10)</title>
<style>
 :root{--bg:#f7f7f8;--card:#fff;--ink:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--accent:#2563eb;
       --tp:#065f46;--fp:#9a3412;--cue:#fef3c7;--done:#dcfce7;}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      background:var(--bg);color:var(--ink)}
 header{position:sticky;top:0;z-index:10;background:var(--card);border-bottom:1px solid var(--line);
        padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header h1{font-size:15px;margin:0;font-weight:700}
 .prog{font-variant-numeric:tabular-nums;color:var(--muted)}
 .bar{height:6px;background:var(--line);border-radius:3px;flex:1;min-width:120px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--accent);width:0}
 button{font:inherit;padding:7px 14px;border:1px solid var(--line);border-radius:7px;background:#fff;
        cursor:pointer}
 button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
 button:disabled{opacity:.5;cursor:not-allowed}
 main{max-width:960px;margin:0 auto;padding:16px}
 .doc{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:14px 0;padding:16px}
 .doc h2{font-size:14px;margin:0 0 8px}
 .badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;margin-right:6px}
 .b-corpus{background:#eef2ff;color:#3730a3}
 .b-tp{background:#ecfdf5;color:var(--tp)} .b-fp{background:#fff7ed;color:var(--fp)}
 .email{white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid var(--line);
        border-radius:8px;padding:10px;max-height:260px;overflow:auto;font-size:13.5px;color:#111}
 .group{margin-top:12px}
 .group>h3{font-size:12px;color:var(--muted);margin:8px 0 6px;text-transform:uppercase;letter-spacing:.04em}
 .set{border:1px solid var(--line);border-radius:9px;padding:10px;margin:8px 0;background:#fff}
 .set.done{background:var(--done);border-color:#86efac}
 .set h4{margin:0 0 8px;font-size:13px}
 .toks{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
 .tok{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--line);border-radius:7px;
      padding:3px 8px;background:#fafafa;cursor:pointer;user-select:none}
 .tok.on{background:var(--cue);border-color:#f59e0b}
 .tok input{margin:0}
 .hol{display:flex;gap:14px;align-items:center;font-size:13px;border-top:1px dashed var(--line);
      padding-top:8px}
 .hol .q{color:var(--muted)}
 .hol label{display:inline-flex;gap:4px;align-items:center;cursor:pointer}
 .hol.unset{background:#fff;outline:2px solid #fca5a5;border-radius:6px;padding:6px}
 footer{max-width:960px;margin:0 auto;padding:20px 16px 60px;color:var(--muted);font-size:13px}
 .note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;margin:10px 0}
</style></head>
<body>
<header>
 <h1>HITL Plausibility Annotation</h1>
 <div class="bar"><i id="pbar"></i></div>
 <span class="prog" id="prog">0 / 0 sets</span>
 <button id="save">Save</button>
 <button class="primary" id="export">Export CSV</button>
</header>
<main id="main"></main>
<footer>
 <div class="note"><b>Blind.</b> Sets are labelled A/B/C(/D) only; you do not know which procedure
 produced any set. Do <b>not</b> open <code>key.json</code>, <code>judge_ratings.csv</code>, or
 <code>cue_lexicon.py</code> while rating. Progress autosaves to this browser; use <b>Export CSV</b>
 to write <code>human_ratings.csv</code> when finished.</div>
 <div id="tip"></div>
</footer>
<script type="application/json" id="payload">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const KEY = 'hitl_d10_ratings_v1';
const store = JSON.parse(localStorage.getItem(KEY) || '{}'); // set_uid -> {cues:{idx:1}, holistic:n}
function totalSets(){let n=0;DATA.docs.forEach(d=>d.groups.forEach(g=>n+=g.sets.length));return n;}
const NSET = totalSets();

function setState(su){ return store[su] || (store[su] = {cues:{}, holistic:null}); }
function isDone(su){ const s=store[su]; return s && s.holistic!==null && s.holistic!==undefined; }
function persist(){ localStorage.setItem(KEY, JSON.stringify(store)); refreshProgress(); }
function refreshProgress(){
  let done=0; DATA.docs.forEach(d=>d.groups.forEach(g=>g.sets.forEach(s=>{if(isDone(s.set_uid))done++;})));
  document.getElementById('prog').textContent = done+' / '+NSET+' sets';
  document.getElementById('pbar').style.width = (100*done/NSET)+'%';
  document.getElementById('export').disabled = false;
}

function render(){
  const main=document.getElementById('main'); main.textContent='';
  DATA.docs.forEach((d,di)=>{
    const doc=document.createElement('section'); doc.className='doc';
    const h=document.createElement('h2');
    const bc=document.createElement('span'); bc.className='badge b-corpus'; bc.textContent=d.corpus;
    const bs=document.createElement('span'); bs.className='badge '+(d.stratum==='TP'?'b-tp':'b-fp');
    bs.textContent=(d.stratum==='TP'?'true positive':'false positive');
    h.appendChild(bc); h.appendChild(bs);
    h.appendChild(document.createTextNode('Email '+(di+1)+' of '+DATA.docs.length));
    doc.appendChild(h);
    const em=document.createElement('div'); em.className='email'; em.textContent=d.text; doc.appendChild(em);
    d.groups.forEach((g,gi)=>{
      const gd=document.createElement('div'); gd.className='group';
      if(d.groups.length>1){ const gh=document.createElement('h3');
        gh.textContent='Token sets — group '+(gi+1)+' of '+d.groups.length; gd.appendChild(gh); }
      else { const gh=document.createElement('h3'); gh.textContent='Token sets'; gd.appendChild(gh); }
      g.sets.forEach(s=>gd.appendChild(renderSet(s))); doc.appendChild(gd);
    });
    main.appendChild(doc);
  });
  refreshProgress();
}

function renderSet(s){
  const st=setState(s.set_uid);
  const card=document.createElement('div'); card.className='set'; card.id='card_'+s.set_uid;
  const title=document.createElement('h4'); title.textContent='Set '+s.label; card.appendChild(title);
  const toks=document.createElement('div'); toks.className='toks';
  s.tokens.forEach((t,idx)=>{
    const lab=document.createElement('label'); lab.className='tok'+(st.cues[idx]?' on':'');
    const cb=document.createElement('input'); cb.type='checkbox'; cb.checked=!!st.cues[idx];
    cb.onchange=()=>{ if(cb.checked)st.cues[idx]=1; else delete st.cues[idx];
                      lab.classList.toggle('on',cb.checked); persist(); };
    const sp=document.createElement('span'); sp.textContent=t;
    lab.appendChild(cb); lab.appendChild(sp); toks.appendChild(lab);
  });
  card.appendChild(toks);
  const hol=document.createElement('div'); hol.className='hol'+(isDone(s.set_uid)?'':' unset');
  const q=document.createElement('span'); q.className='q';
  q.textContent='Does this set explain why the email is phishing?'; hol.appendChild(q);
  [0,1,2].forEach(v=>{
    const lab=document.createElement('label'); const r=document.createElement('input');
    r.type='radio'; r.name='hol_'+s.set_uid; r.checked=(st.holistic===v);
    r.onchange=()=>{ st.holistic=v; hol.classList.remove('unset');
                     card.classList.add('done'); persist(); };
    lab.appendChild(r); lab.appendChild(document.createTextNode(' '+v)); hol.appendChild(lab);
  });
  if(isDone(s.set_uid)) card.classList.add('done');
  card.appendChild(hol); return card;
}

function exportCSV(){
  const rows=[['doc_uid','set_uid','token_idx','token','is_cue','holistic']];
  let missing=0;
  DATA.docs.forEach(d=>d.groups.forEach(g=>g.sets.forEach(s=>{
    const st=store[s.set_uid]||{cues:{},holistic:null};
    const hv=(st.holistic===null||st.holistic===undefined)?'':st.holistic;
    if(hv==='')missing++;
    s.tokens.forEach((t,idx)=>rows.push([d.doc_uid,s.set_uid,idx,t,st.cues[idx]?1:0,hv]));
  })));
  const csv=rows.map(r=>r.map(c=>{const s=String(c);
     return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}).join(',')).join('\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='human_ratings.csv'; a.click();
  document.getElementById('tip').textContent = missing
    ? ('Exported, but '+missing+' set(s) have no holistic score — finish them and re-export.')
    : 'Exported human_ratings.csv — all sets rated. Commit it into results/hitl/.';
}
document.getElementById('export').onclick=exportCSV;
document.getElementById('save').onclick=()=>{persist();document.getElementById('tip').textContent='Saved.';};
render();
</script>
</body></html>
"""


if __name__ == "__main__":
    build()
