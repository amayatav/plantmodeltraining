# Plant Scan: Species Model Results (v2)

**Prepared by:** Zuri McKee (Modeling)
**Date:** November 20, 2026
**Status:** Species model is chosen and scored on the test split. The health/condition feature is **not built yet** and needs a team decision (section 8).

> **Corrections to the earlier version of this doc:** it said 35 species and 3,078 images. The real dataset is **34 species and 3,073 images**. It also described the ResNet50 fine-tune as the best model; it has been replaced by the DINOv2 model below.

---

## TL;DR

- **Final species model: DINOv2-base (frozen) + a logistic regression classifier.** It scores **86.3% top-1 accuracy, 0.870 macro F1, and 94.8% top-3 accuracy** on the held-out test split (461 images). That meets our 85% species-accuracy target, but with a thin margin (see the caveat below).
- It is a large improvement over the ResNet50 approach on the same test images: **86.3% vs 69.6%** for the fine-tuned ResNet50 and 52.1% for the frozen ResNet50 baseline.
- The big lever was the **backbone**, not more data or more training. A self-supervised DINOv2 backbone with a simple classifier beat a fully fine-tuned ResNet50 by about 17 points.
- **Open problems:** (1) real phone photos haven't been scored because the phone set is about 20 images with no labels; (2) GBIF has almost no photos of common houseplants, so several popular ones are missing; (3) the plant-health feature has no dataset or method yet.

---

## 1. Data

| Item | Value |
|---|---|
| Source | GBIF plant images via Hugging Face `juppy44/gbif-plants-raw` |
| Species kept | **34** (11 houseplants, 23 trees, shrubs and other garden plants) |
| Images | **3,073** (collection capped at 100 per species) |
| Split (stratified 70/15/15, seed 42) | train **2,151** / val **461** / test **461** |
| Cleaning | removed corrupt, tiny (<64 px) and exact-duplicate images; classes with fewer than 30 clean images are dropped |
| Split file | `species_split.csv` (one saved manifest, so every model sees identical test images) |

**Species kept:**
- *Houseplants (11):* Aloe vera, Chlorophytum comosum, Crassula ovata, Dieffenbachia seguine, Dracaena trifasciata, Epipremnum aureum, Ficus elastica, Hedera helix, Monstera deliciosa, Philodendron hederaceum, Tradescantia zebrina
- *Trees, shrubs and other (23):* Acer platanoides, A. rubrum, A. saccharum, Betula pendula, Buxus sempervirens, Cornus florida, Fraxinus americana, Ginkgo biloba, Hydrangea macrophylla, Ilex aquifolium, Liriodendron tulipifera, Magnolia grandiflora, Pinus strobus, P. sylvestris, Platanus occidentalis, Quercus alba, Q. robur, Rhododendron ponticum, Rosa canina, Salix babylonica, Syringa vulgaris, Taraxacum officinale, Taxus baccata

**Candidate species dropped for having too few images** (the image count found in the dataset):

| Species | Images found |
|---|---|
| Schlumbergera truncata (Christmas cactus) | 1 |
| Echeveria elegans | 1 |
| Zamioculcas zamiifolia (ZZ plant) | 1 |
| Spathiphyllum wallisii (peace lily) | 6 |
| Aglaonema commutatum | 10 |
| Ficus lyrata (fiddle-leaf fig) | 11 |
| Lavandula angustifolia | 21 |
| Forsythia suspensa | 23 |
| Nephrolepis exaltata (Boston fern) | 25 |

**The houseplant gap.** GBIF is mostly wild observations, so trees and shrubs are well covered and many indoor plants are not. Peace lily, ZZ plant, fiddle-leaf fig and Christmas cactus are exactly what home gardeners photograph, and our model cannot recognize them. See section 8.

---

## 2. What we trained and compared

| Model | Backbone | How trained |
|---|---|---|
| ResNet50, frozen (baseline) | ImageNet | new linear head only, 10 epochs |
| ResNet50, fine-tuned | ImageNet | all layers, augmentation, AdamW, cosine schedule, 10 epochs, lr 1e-4 |
| DINOv2-small, frozen head | DINOv2 | head only, 5 epochs |
| DINOv2-small, fine-tuned | DINOv2 | all layers, same recipe, 10 epochs, lr 1e-5 |
| DINOv2-small / base, frozen + logistic regression | DINOv2 | features extracted once from clean images, then a logistic regression (standardized features, C = 0.1) |

All models share the same split, image transforms and metrics. Checkpoints were chosen on **validation** macro F1, never on test.

**Hardware:** Apple M4 laptop (MPS), `num_workers=0`. About 100 to 145 s per ResNet50 epoch, 90 s per fine-tune epoch and 54 s per frozen-head epoch for DINOv2-small.

---

## 3. Results

### 3.1 Test split (final numbers, 461 images)

| Model | Top-1 accuracy | Macro F1 | Top-3 |
|---|---|---|---|
| ResNet50, frozen head (baseline) | 0.521 | 0.499 | not computed |
| ResNet50, fine-tuned | 0.696 | 0.693 | 0.844 |
| **DINOv2-base, frozen + logistic regression (final)** | **0.863** | **0.870** | **0.948** |

- The DINOv2 model was scored on test **once**. Every choice (backbone, C) was made on validation.
- The ResNet50 test split was scored several times (dry run, full run, error analysis), though no model choice was made from it.
- For scale: with 34 classes, random guessing gives about 3% top-1 and about 9% top-3.

### 3.2 Validation split (used for choosing a model, 461 images)

| Model | Val accuracy | Val macro F1 |
|---|---|---|
| ResNet50, frozen features + logistic regression | 0.599 | 0.603 |
| ResNet50, fine-tuned | about 0.72 | about 0.72 |
| DINOv2-small, frozen head (training loop, 5 epochs) | 0.813 | 0.817 |
| DINOv2-small, fine-tuned (lr 1e-5) | 0.846 | 0.846 |
| DINOv2-small, frozen + logistic regression | 0.866 | 0.875 |
| **DINOv2-base, frozen + logistic regression (C = 0.1)** | **0.889** | **0.895** (top-3: 0.963) |

**Sensitivity to C** (the classifier's regularization strength) for DINOv2-base: C = 0.01 gives 0.879 / 0.885; C = 0.03 and 0.1 give 0.889 / 0.895; C = 0.3 and 1.0 give 0.892 / 0.897 (accuracy / macro F1). The result is flat from 0.03 upward, so we kept C = 0.1 and stopped tuning.

### 3.3 Learning curves (validation macro F1, mean of 3 random subsets of the training data)

| Training images | ResNet50 features | DINOv2-small | DINOv2-base |
|---|---|---|---|
| 537 | 0.438 | 0.776 | 0.830 |
| 1,075 | 0.531 | 0.829 | 0.867 |
| 1,613 | 0.561 | 0.848 | 0.885 |
| 2,151 | 0.603 | 0.875 | 0.895 |

DINOv2 with only 25% of the data beats ResNet50 features with all of it. The DINOv2-base curve is flattening, so more images of the same kind would help only modestly.

---

## 4. Findings

1. **The backbone mattered most.** With the same split, augmentation, optimizer and fine-tune epochs, fine-tuned DINOv2-small beat fine-tuned ResNet50 by about 13 points on validation (0.846 vs 0.72). The head-stage length (5 vs 10 epochs) and fine-tune learning rate (1e-5 vs 1e-4) differed.
2. **Fine-tuning DINOv2 did not beat the simple probe.** Fine-tuned DINOv2-small (0.846) scored below the frozen logistic regression on the same backbone (0.866). Note the fine-tune number is slightly optimistic, since its best epoch was picked on validation. We shipped the cheaper, simpler option.
3. **ResNet50 overfit.** Train accuracy reached about 96% while validation stalled near 72% from epoch 7, so more epochs would not have helped.
4. **Remaining errors are lookalike pairs.** DINOv2-base's weakest validation classes are below.

**Weakest classes, DINOv2-base (validation, 12 weakest):**

| Class | Val images | Recall | Most confused with |
|---|---|---|---|
| Ginkgo_biloba | 6 | 0.67 | Liriodendron_tulipifera (2) |
| Pinus_sylvestris | 15 | 0.67 | Pinus_strobus (4), Fraxinus_americana (1) |
| Quercus_alba | 15 | 0.67 | Quercus_robur (2), Platanus_occidentalis (1) |
| Quercus_robur | 15 | 0.73 | Acer_saccharum (1), Liriodendron_tulipifera (1) |
| Liriodendron_tulipifera | 15 | 0.73 | Platanus_occidentalis (1), Acer_saccharum (1) |
| Acer_saccharum | 15 | 0.73 | Quercus_alba (1), Fraxinus_americana (1) |
| Platanus_occidentalis | 15 | 0.80 | Acer_saccharum (1), Acer_platanoides (1) |
| Pinus_strobus | 15 | 0.80 | Pinus_sylvestris (3) |
| Acer_rubrum | 15 | 0.80 | Taxus_baccata (1), Pinus_strobus (1) |
| Salix_babylonica | 10 | 0.80 | Buxus_sempervirens (1), Crassula_ovata (1) |
| Acer_platanoides | 15 | 0.80 | Acer_rubrum (1), Quercus_robur (1) |
| Buxus_sempervirens | 15 | 0.87 | Taxus_baccata (2) |

For comparison, the fine-tuned ResNet50's worst classes scored 0.20 to 0.47 recall on test. What remains for DINOv2 is the genuinely hard part of species ID: pine vs pine, oak vs oak, and similar trees. Ginkgo has very few images (about 25 in train), so its score is noisy.

---

## 5. Caveats

- **The margin over the target is thin.** The test score is 86.3% on 461 images. A rough 95% interval is about ±3 points (roughly 83% to 89%), so we can say "86.3% on our test split, which meets the 85% target," but not that the true accuracy is certainly above 85%.
- **Per-class numbers are rough.** Validation and test have 6 to 15 images per class, so one wrong image moves a class's recall by about 7 to 17 points. Use overall accuracy and macro F1 as the headline numbers.
- **All results are on GBIF photos.** Real phone photos will probably score lower, and that is our stated key risk. It is currently unmeasured (section 8).
- **The model always picks one of its 34 species,** even for a plant outside the list. The app should show a "not sure" message below a confidence cutoff, chosen on validation data.
- **Pretraining overlap.** DINOv2 was pretrained on a large web image collection. We cannot verify how much it overlaps with GBIF photos, so the report should state this.
- **Licensing.** GBIF images carry per-image licenses, some non-commercial. That is fine for a class project but should be noted in the report.
- **Confidence scores** from logistic regression are not guaranteed to be calibrated probabilities.

---

## 6. How the app should use the model

The final model is saved as `checkpoints/species_dinov2_base_probe.joblib`. It holds the standardizer, the classifier, the class names and the backbone name (`facebook/dinov2-base`). The notebook defines `SpeciesPredictor`, which takes an image and returns `prediction`, `confidence` and `top_k` (the top 3 species with scores).

- **Show the top 3.** It is 94.8% accurate on test and handles lookalikes much better than a single guess.
- **Add a confidence cutoff** such as "not sure, try another photo." The threshold has not been chosen yet.
- **Pin library versions.** The saved scaler and classifier depend on the scikit-learn version used to train them, so the app environment needs the same `scikit-learn`, `torch` and `transformers` versions (check with `pip freeze`).
- Inference runs the DINOv2-base backbone (about 86M parameters). Speed per image has not been measured on the app's hardware.

---

## 7. Plant health: where things stand

The Checkpoint 1 report promises healthy-vs-diseased classification at 80% or better, plus matching care advice. **None of that is built or measured.** The species data has no health labels.

**Datasets considered:**

| Dataset | What it is | Verdict |
|---|---|---|
| PlantVillage (named in the Launch doc) | About 14 food crops, lab-style leaf photos | Wrong plants for our goal (everyday plants, not crops) |
| Mendeley "Indoor Plant Disease Image Dataset" (7wbstnfpjy, v2) | 3 houseplants (money plant, snake plant, spider plant), 3 classes each (healthy plus two conditions), 787 original smartphone images, CC BY 4.0 | Team judged it will not work: only 3 species. Details are from the dataset page; we did not inspect the files |
| Zenodo "Indoor plants DataSet" (record 6984514) | A 38.8 kB table of care parameters (light, watering, humidity, soil, toxicity). No images, no health labels | Not usable for training, but could supply content for care advice. Its species coverage and license terms need checking |
| Team phone photos | About 20 images, no labels | Too small and unlabeled to use yet |

**Proposed alternative (not yet agreed):** replace the health model with a mostly deterministic approach. After the species is identified, the app asks the user "What do you see?" (yellow leaves, brown tips, spots, wilting, pests). A lookup table maps species plus symptom to likely causes and fixes, with sources cited and a "general guidance, not a professional diagnosis" note. This needs no new model. Two variants are possible: a color heuristic that pre-selects "possible yellowing," or a zero-shot vision-language model (unproven accuracy).

**What this would change:** the 80% healthy-vs-diseased measure could no longer be reported. The team should tell the instructor and rewrite that evaluation measure in the next report, for example: species model accuracy plus advice-table correctness reviewed against cited sources.

---

## 8. Decisions needed from the team

1. **Houseplant coverage (Dakota):** which species must the app cover, and where do the missing images come from (iNaturalist or GBIF APIs filtered to human observations, a Kaggle houseplant dataset, or team phone photos)? Adding classes changes the split, so all models would be retrained and new numbers would not be directly comparable.
2. **Phone-photo test set (Dakota):** label the existing ~20 photos with species names that match our class list, and take more (about 10 or more per species). Without labels we cannot measure real-world accuracy.
3. **Health feature:** choose between the deterministic symptom-picker, a labeled health dataset, or narrowing health to a few species. Inform the instructor of any scope change.
4. **Lookalike species:** keep separate, merge, or show at genus level ("Pine, *Pinus*") when confidence is split?
5. **Confidence cutoff** for the "not sure" message.
6. **Report wording:** the species target is met at 86.3% on test. State the margin honestly.

---

## 9. Suggested next steps (in order)

1. Label the phone photos and score them once with the final model.
2. Settle the houseplant list and collect missing images.
4. Hand `SpeciesPredictor` to the interface team and test it on real photos.
5. Update the Checkpoint 2 report: baseline vs fine-tuned ResNet50 (as originally planned), plus the DINOv2 result as the improvement.

---

## 10. Reproducing

- **Notebook:** run in VS Code on a local machine; everything lives under `~/plant_project` (`plants/`, `species_split.csv`, `species_urls.json`, `feats_*.npz`, `checkpoints/`, `results/`).
- **After a kernel restart, run in this order:** config cell, split cell, training-and-evaluation cell, then the learning-curve cells. No rescan or re-download is needed.
- **Feature caches** (`feats_*.npz`) make the DINOv2 experiments fast to rerun. They hold train and validation features only; test features are extracted once, in the final scoring cell.
- **Mac notes:** use `num_workers=0`; the device is picked automatically (`mps`).
- **Do not rerun the final test-scoring cell with changes.** That would turn the test split into a second validation set.
