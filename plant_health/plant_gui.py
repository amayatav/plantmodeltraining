"""
plant_gui.py - Plant Identifier GUI with health-concern advice.

Needs these files in the same folder (data_dir):
    advice.csv, species_groups.csv, symptoms.csv

Usage (from the notebook, after the model is trained/loaded):
    import plant_gui
    demo = plant_gui.build_demo(species_model, species_classes, predict, data_dir)
    demo.launch(share=True, debug=True)

`predict` is the notebook's predict(img, model, classes, k=3) function.
This module does not import torch, so it only depends on gradio, pandas, pillow.
"""
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Wording for the follow-up questions.
# advice.csv only stores codes (soil_moisture / wet ...). Edit the text here to
# change what the user sees. Anything missing falls back to a cleaned-up code.
# ---------------------------------------------------------------------------
CHECK_TEXT = {
    "soil_moisture": ("How does the soil feel?",
                      {"wet": "Wet or soggy", "moist": "Evenly moist", "dry": "Dry"}),
    "likely_situation": ("Which of these sounds most like your situation?",
                         {"dry_air": "The air is dry (heater, AC, or a vent nearby)",
                          "irregular_watering": "I water irregularly",
                          "heavy_fertilizing": "I fertilize heavily or often",
                          "tap_water": "I water with tap water",
                          "unsure": "I'm not sure"}),
    "season": ("What time of year is it?",
               {"fall": "Fall", "not_fall": "Not fall (spring, summer, or winter)"}),
    "sun_exposure": ("How much sun does it get?",
                     {"low": "Too little (dim spot)", "high": "Too much (strong direct sun)"}),
    "needle_location": ("Where are the yellow needles?",
                        {"inner_older": "Inner, older needles",
                         "tips_or_irregular": "Needle tips or irregular patches"}),
    "timing": ("When did the browning show up?",
               {"after_winter": "After winter", "other_season": "Another time of year"}),
    "appearance": ("What do the spots look like?",
                   {"white_powder": "White powdery coating", "dark_spots": "Dark spots"}),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def norm(s):
    """'Quercus_robur' / 'quercus  robur' -> 'quercus robur' (for matching)."""
    return " ".join(str(s).replace("_", " ").lower().split())


def pretty(s):
    return str(s).replace("_", " ")


def cap1(s):
    return s[:1].upper() + s[1:]


def check_question(key):
    return CHECK_TEXT.get(key, (cap1(pretty(key)) + "?", {}))[0]


def answer_label(key, ans):
    return CHECK_TEXT.get(key, ("", {}))[1].get(ans, cap1(pretty(ans)))


# ---------------------------------------------------------------------------
# Loading the CSVs
# ---------------------------------------------------------------------------
def expand_species(cell):
    """'Acer rubrum / saccharum / platanoides' -> three full species names."""
    parts = [p.strip() for p in cell.split("/")]
    genus = parts[0].split()[0]
    return [p if len(p.split()) >= 2 else f"{genus} {p}" for p in parts]


def expand_common(cell, n):
    """'Red / sugar / Norway maple' -> ['Red maple', 'Sugar maple', 'Norway maple']."""
    parts = [p.strip() for p in cell.split("/")]
    if n == 1 or len(parts) != n:
        return [cell.strip()] * n
    tail = parts[-1].split()[-1]
    return [cap1(p if len(p.split()) > 1 else f"{p} {tail}") for p in parts]


def read_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df.apply(lambda c: c.str.strip())


def load_data(data_dir):
    d = Path(data_dir)
    advice = read_csv(d / "advice.csv")
    groups = read_csv(d / "species_groups.csv")
    symptoms = read_csv(d / "symptoms.csv")

    plants = {}   # normalized species name -> {species, common, group}
    for _, r in groups.iterrows():
        names = expand_species(r["Species"])
        commons = expand_common(r["Common name"], len(names))
        for sp, cn in zip(names, commons):
            plants[norm(sp)] = {"species": sp, "common": cn, "group": r["Group"]}
    return advice, plants, symptoms


# ---------------------------------------------------------------------------
# Advice engine (pure pandas, no UI)
# ---------------------------------------------------------------------------
def matching_rows(advice, plant, symptom_id):
    """Advice rows for this plant + symptom. A row whose species_override names
    this exact species beats the shared group rows."""
    a = advice[advice.symptom_id == symptom_id]
    over = a.species_override.map(norm)
    mine = a[over == norm(plant["species"])]
    if len(mine):
        return mine
    return a[(a.group == plant["group"]) & (over == "")]


def symptoms_for(advice, symptoms, plant):
    """(label, id) for every symptom that has advice for this plant."""
    return [(r.label, r.symptom_id) for r in symptoms.itertuples()
            if len(matching_rows(advice, plant, r.symptom_id))]


def needs_question(rows):
    """Rows that depend on a follow-up answer (not 'none'/'any')."""
    return rows[(rows.check_key != "none") & (rows.check_answer != "any")]


def followup(rows):
    """(check_key, [answers]) if the user must answer a question, else None."""
    opts = needs_question(rows)
    if opts.check_answer.nunique() >= 2:
        return opts.check_key.iloc[0], list(dict.fromkeys(opts.check_answer))
    return None


def format_row(r):
    out = f"### Likely cause: {r.likely_cause}\n\n{r.fix}\n\n"
    if r.source_url:
        out += f"*Source: [{r.source_title or r.source_url}]({r.source_url})*"
    elif r.source_title:
        out += f"*Source: {r.source_title}*"
    return out


def advice_markdown(advice, plant, symptom_id, symptom_label, symptom_help, chosen=None):
    rows = matching_rows(advice, plant, symptom_id)
    header = f"**You selected:** {symptom_label}"
    if symptom_help:
        header += f"  \n*{symptom_help}*"
    if not len(rows):
        return header + "\n\nNo advice for this combination yet."

    asked = followup(rows)
    cond = needs_question(rows)
    if asked and chosen is None:
        return header + "\n\n*Answer the question above to see the likely cause.*"

    shown = rows[(rows.check_answer == "any") | (rows.check_key == "none")
                 | (rows.check_answer == chosen)] if asked else rows
    if not len(shown):
        return header + "\n\nNo advice matches that answer yet."

    parts = []
    for r in shown.itertuples():
        text = format_row(r)
        # a single condition-specific row: say what it assumes
        if not asked and r.check_key != "none" and r.check_answer != "any":
            text += (f"\n\n*Fits best when: {check_question(r.check_key)} "
                     f"-> {answer_label(r.check_key, r.check_answer)}*")
        parts.append(text)
    return header + "\n\n---\n\n" + "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Startup sanity checks (printed once so typos in the CSVs are easy to spot)
# ---------------------------------------------------------------------------
def sanity_report(advice, plants, symptoms, classes):
    known_groups = {p["group"] for p in plants.values()}
    bad_g = sorted(set(advice.group) - known_groups)
    bad_s = sorted(set(advice.symptom_id) - set(symptoms.symptom_id))
    no_row = [c for c in classes if norm(c) not in plants]
    no_advice = sorted({p["common"] for k, p in plants.items()
                        if any(norm(c) == k for c in classes)
                        and not (advice.group == p["group"]).any()})
    if bad_g:
        print("WARNING advice.csv groups not found in species_groups.csv:", bad_g)
    if bad_s:
        print("WARNING advice.csv symptom_ids not found in symptoms.csv:", bad_s)
    if no_row:
        print("WARNING model classes with no row in species_groups.csv:", no_row)
    if no_advice:
        print("NOTE no advice rows yet for:", no_advice)


# ---------------------------------------------------------------------------
# The GUI
# ---------------------------------------------------------------------------
def build_demo(model, classes, predict, data_dir="."):
    if hasattr(model, "eval"):
        model.eval()
    advice, plants, symptoms = load_data(data_dir)
    sym_label = dict(zip(symptoms.symptom_id, symptoms.label))
    sym_help = dict(zip(symptoms.symptom_id, symptoms.help_text))
    sanity_report(advice, plants, symptoms, classes)

    class_by_norm = {norm(c): c for c in classes}   # 'Monstera deliciosa' -> 'Monstera_deliciosa'

    def plant_of(cls):
        return plants.get(norm(cls)) if cls else None

    def display(cls):
        p = plant_of(cls)
        return f"{p['common']} ({p['species']})" if p else pretty(cls)

    plant_choices = sorted(((display(c), c) for c in classes), key=lambda t: t[0])

    # Start the symptom / follow-up widgets with EVERY possible choice so the
    # server always accepts the value; each plant then narrows the list.
    all_symptoms = [(r.label, r.symptom_id) for r in symptoms.itertuples()]
    all_answers = {}
    for r in advice.itertuples():
        if r.check_key != "none" and r.check_answer != "any":
            all_answers.setdefault(r.check_answer, answer_label(r.check_key, r.check_answer))
    all_answers = [(lbl, code) for code, lbl in all_answers.items()]

    # ---- Tab 1: single photo + health concerns ----
    def single(img):
        if img is None:
            return None, gr.update()
        top = predict(img, model, classes, k=3)["top_k"]
        return {display(n): p for n, p in top}, class_by_norm.get(norm(top[0][0]), top[0][0])

    def on_plant(cls):
        p = plant_of(cls)
        hide = gr.update(visible=False, value=None)
        if cls is None:
            return gr.update(choices=[], value=None), hide, ""
        if not p:
            return (gr.update(choices=[], value=None), hide,
                    f"No health information for **{pretty(cls)}** yet.")
        opts = symptoms_for(advice, symptoms, p)
        if not opts:
            return (gr.update(choices=[], value=None), hide,
                    f"No health advice for **{display(cls)}** yet.")
        return gr.update(choices=opts, value=None), hide, ""

    def on_symptom(cls, sym):
        p = plant_of(cls)
        if not p or not sym:
            return gr.update(visible=False, value=None), ""
        rows = matching_rows(advice, p, sym)
        asked = followup(rows)
        md = advice_markdown(advice, p, sym, sym_label[sym], sym_help.get(sym, ""))
        if asked:
            key, answers = asked
            radio = gr.update(visible=True, value=None, label=check_question(key),
                              choices=[(answer_label(key, a), a) for a in answers])
            return radio, md
        return gr.update(visible=False, value=None), md

    def on_check(cls, sym, ans):
        p = plant_of(cls)
        if not p or not sym:
            return ""
        return advice_markdown(advice, p, sym, sym_label[sym], sym_help.get(sym, ""), ans)

    # ---- Tab 2: many photos ----
    def batch(files):
        if not files:
            return [], pd.DataFrame(columns=["Photo", "#1", "#2", "#3"])
        gallery, rows = [], []
        for f in files:
            path = f if isinstance(f, str) else f.name
            img = Image.open(path)
            top = predict(img, model, classes, k=3)["top_k"]
            gallery.append((img.convert("RGB"), f"{display(top[0][0])} ({top[0][1]:.0%})"))
            rows.append({"Photo": Path(path).name,
                         **{f"#{i+1}": f"{display(n)} - {p:.1%}" for i, (n, p) in enumerate(top)}})
        return gallery, pd.DataFrame(rows)

    theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald")

    with gr.Blocks(theme=theme, title="Plant Identifier") as demo:
        gr.Markdown("# 🌿 Plant Identifier\nUpload a photo and the model shows its top 3 guesses with confidence levels. "
                    "Then tell us what looks wrong to get care advice.")

        with gr.Tabs():
            with gr.Tab("Single photo"):
                with gr.Row():
                    inp = gr.Image(type="pil", label="Your photo", height=380,
                                   sources=["upload", "webcam", "clipboard"])
                    out = gr.Label(num_top_classes=3, label="Top 3 predictions")

                gr.Markdown("### 🩺 Health concerns\nNot looking right? Confirm the plant, then pick what you're noticing.")
                with gr.Row():
                    plant = gr.Dropdown(choices=plant_choices, value=None, label="Which plant is this?",
                                        info="Filled in from the top guess. Change it if the model got it wrong.")
                    symptom = gr.Dropdown(choices=all_symptoms, value=None, label="What are you noticing?")
                check = gr.Radio(choices=all_answers, value=None, label="", visible=False)
                advice_out = gr.Markdown()

                inp.change(single, inp, [out, plant])
                plant.change(on_plant, plant, [symptom, check, advice_out])
                symptom.input(on_symptom, [plant, symptom], [check, advice_out])
                check.input(on_check, [plant, symptom, check], advice_out)
                gr.ClearButton([inp, out, plant, symptom, check, advice_out])

            with gr.Tab("Multiple photos"):
                files = gr.File(label="Upload photos", file_count="multiple", file_types=["image"])
                btn = gr.Button("Identify all", variant="primary")
                gal = gr.Gallery(label="Results", columns=4, height="auto")
                table = gr.Dataframe(label="Top 3 for each photo", wrap=True)
                btn.click(batch, files, [gal, table])

        gr.Markdown("*The model can only choose from the species it was trained on, so unusual plants may get "
                    "low-confidence guesses. Care advice is general guidance from the cited sources, not a diagnosis.*")
    return demo
