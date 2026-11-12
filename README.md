# Verity — Local Fake/Real News Detection

## Files
- `app.py`: Streamlit dashboard. Paste your own news, get a prediction and confidence estimate.
- `train_model.py`: trains and evaluates the Decision Tree + TF-IDF pipeline from your notebook.
- `News.csv`: your dataset (you must copy this into this folder).
- `requirements.txt`: dependencies.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Accuracy vs confidence
- Held-out test accuracy is calculated by `train_model.py` on labeled test rows.
- For an individual pasted article, the app shows the model's predicted class and (if available) class probability.
- To calculate accuracy on your own pasted articles, select the known actual label and save the evaluation. The app calculates accuracy over the labeled articles entered during that app session.
- A prediction confidence is not proof that a news story is true.

## Label mapping
Your notebook's `predict_news` function assumes `1 = Real News` and `0 = Fake News`. Verify that the `class` column in your dataset follows this mapping.
