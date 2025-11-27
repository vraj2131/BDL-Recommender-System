# Bayesian Movie Recommender System

Interactive Bayesian recommender built on **MovieLens 100K** that compares four
models:

- **Baseline MF**
- **Baseline MF (VI)**
- **Core VI** – Hierarchical Bayesian MF with user/item side information
- **Core MCMC** – Same hierarchical model inferred with MCMC on a warm subset

---

## 🚀 Live Demo

Try the app on Streamlit Cloud:

👉 **https://bdl-recommender-system-q84z9jmkpmo5qul9rvlnv8.streamlit.app/**

---

## 🎯 Project Overview

This project started from a **plain matrix factorization (MF) baseline** that
can only:

- Predict ratings for users and items seen in training (warm-start)
- Use a simple dot–product between user and item latent vectors
- Provide no notion of uncertainty or side information

Our goal was to build **richer Bayesian recommenders** that:

1. Match or beat MF on **warm-start prediction**.
2. Provide **true cold-start recommendations** using user features.
3. Offer **uncertainty estimates** for each predicted rating.
4. Show these behaviors in an **interactive dashboard** that is easy to
   understand for non-experts.

---

## 🧪 What We Actually Implemented

### 1. Four Trained Models

1. **Baseline MF**
   - Classic low-rank MF trained with squared-error loss.
   - Uses only the user–item rating matrix.
   - Supports warm-start users/items.

2. **Baseline MF (VI)**
   - Bayesian version of MF using **variational inference**.
   - Learns distributions over user and item latent factors.
   - Still **no side information**, but can quantify parameter uncertainty.
   - Warm-start only.

3. **Core VI (HBMFSI VI)**
   - **Hierarchical Bayesian MF with Side Information (HBMFSI)**.
   - Uses:
     - User features: age, gender, occupation.
     - Item features: 19 movie genres.
   - Learns linear mappings:
     - user features → user latent space
     - item features → item latent space
   - Supports:
     - Warm-start users/items.
     - **Cold-start users** (only features known).
     - **Cold-start items** (only features known).
   - Inferred with variational Bayes.

4. **Core MCMC (HBMFSI MCMC)**
   - Same hierarchical structure as Core VI.
   - Uses **MCMC samples** on a warm
     subset of the data.
   - Provides better calibrated **posterior predictive uncertainty** by averaging
     over samples.
   - Evaluated for warm-start and cold-start on that subset.

All four models were trained and evaluated, and the learned
weights / samples are loaded in the Streamlit app.

---

## 💡 Why This Approach Is Interesting 

Compared to a standard MF homework, our project adds several layers:

1. **Side Information → True Cold-Start**
   - Instead of just imputing ratings from latent vectors, we **explicitly map
     user and item features into the latent space**.
   - This lets us generate recommendations for:
     - A brand-new user defined only by age, gender, occupation.
     - A new movie defined by its genre vector.
   - The dashboard makes this visible via the **“Cold-start: New User”**
     scenario.

2. **Bayesian View of MF**
   - We implement **both VI and MCMC** for Bayesian MF with side information.
   - This allows us to:
     - Compare approximate inference (VI) vs sampling-based inference (MCMC).
     - Inspect the **stability of predictions** via predictive standard
       deviations from MCMC.

3. **Unified Experimental Comparison**
   - All models are evaluated on **the same MovieLens 100K splits** with:
     - RMSE / MAE on warm-start test ratings.
     - RMSE / MAE on **cold-start users** and **cold-start items** (where
       applicable).
   - The dashboard shows a compact **metrics table** so you can see that:
     - Core VI / Core MCMC roughly **match or slightly beat Baseline MF** on
       warm-start RMSE/MAE.
     - Core models are the **only ones that handle feature-only cold-start**.

4. **Interactive Semi-Cold Behavior**
   - In the cold-start panel you can optionally specify **1–2 movies the new
     user likes**.
   - These ratings are folded into the user representation, producing a
     **semi-cold** recommendation (mix of side information + a few ratings).
   - This mimics real onboarding flows where you ask a new user for a handful of
     favorite movies.

5. **Teaching-Focused Dashboard**
   - The UI is structured to make the modeling story clear:
     - **Scenario selector**: warm vs cold.
     - **Model selector**: Baseline vs Bayesian / side-info models.
     - Explicit message when a model **cannot** handle a scenario
       (e.g., Baseline MF on feature-only cold-start).
   - This makes it a useful **pedagogical tool** to explain:
     - Why side information matters.
     - What Bayesian inference buys you beyond point estimates.
     - How different models behave on the *same* user.

---

## 🧑‍💻 How the Dashboard Works

### Scenarios

- **Warm-start: Existing User**
  - Input: MovieLens user ID (1–943).
  - The app:
    - Looks up this user’s latent vector for each model.
    - Scores all movies, sorts by predicted rating.
    - Shows the **Top-N recommendations** with movie IDs, titles, and predicted
      ratings.
  - For MCMC, the user slider is restricted to the subset for which samples were
    stored.

- **Cold-start: New User**
  - Input:
    - Age, gender, occupation.
    - Optional list of movies the user already likes.
  - The app:
    - Encodes the user features into a user latent vector via `A` (Core VI) or
      the corresponding mapping.
    - Optionally folds in explicit ratings from the selected liked movies.
    - Uses that latent vector to score all movies and return Top-N
      recommendations.
  - Baseline models are clearly marked as **not applicable** for this
    feature-only setting.

### Predicted Rating Interpretation

- `pred_rating` in the tables is the **model’s expected rating on a 1–5 scale**.
- Higher values indicate movies that the model thinks the user is more likely to
  rate highly.
- In the MCMC-based model, these ratings are **averaged over samples**, so they
  also implicitly reflect uncertainty.

---

### 📚 Dataset

We use MovieLens 100K:

~100,000 ratings by 943 users on 1,682 movies.

User attributes: age, gender, occupation, zip code.

Movie attributes: genres, release information.
