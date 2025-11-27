import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------
DATA_DIR = Path("ml-100k")
MF_PATH = "mf_warm_model.pt"
BAYES_MF_VI_PATH = "bayesian_mf_vi_warm.pt"
HBMFSI_VI_PATH = "hbmfsi_vi_warm.pt"
HBMFSI_MCMC_PATH = "hbmfsi_mcmc_warm_subset.pt"

TOP_N_DEFAULT = 10
RATING_SCALE = (1.0, 5.0)
DEFAULT_LIKE_RATING = 4.5  # assumed rating for "movies you like" in cold-start

# ---------------------------------------------------------
# Utility: cache loaders
# ---------------------------------------------------------
@st.cache_data
def load_movielens_metadata():
    """Load MovieLens 100K user + movie metadata."""
    users = pd.read_csv(
        DATA_DIR / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        names=[
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Childrens",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "FilmNoir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "SciFi",
            "Thriller",
            "War",
            "Western",
        ],
        encoding="latin-1",
    )

    return users, movies


@st.cache_data
def build_movie_id_to_index(movies_df: pd.DataFrame):
    """Map MovieLens movie_id -> 0-based index (aligned with factors)."""
    return {int(mid): i for i, mid in enumerate(movies_df["movie_id"].values)}


@st.cache_resource
def load_models():
    """
    Load the four trained models from disk.

    We explicitly set weights_only=False for PyTorch>=2.6 (e.g. Streamlit Cloud),
    because these .pt files contain full pickled objects, not just tensor weights.
    The try/except keeps it compatible with older local PyTorch versions that
    don't know the weights_only argument.
    """
    def _load_model(path):
        """
        Safe model loader for PyTorch 2.6 on Streamlit Cloud.
        First tries weights_only=True, then falls back.
        """
        try:
            # First try strict loading (PyTorch 2.6+)
            obj = torch.load(path, map_location="cpu", weights_only=True)
            return obj, None
        except:
            try:
                # Fallback: load full pickle (ONLY SAFE BECAUSE FILE IS YOURS)
                obj = torch.load(path, map_location="cpu", weights_only=False)
                return obj, None
            except Exception as e:
                st.error(f"Failed to load model {path}: {e}")
                raise e


    mf_model, mf_ckpt = _load_model(MF_PATH)
    bayes_mf_vi_model, bayes_ckpt = _load_model(BAYES_MF_VI_PATH)
    hbmfsi_vi_model, hb_vi_ckpt = _load_model(HBMFSI_VI_PATH)
    mcmc_obj, mcmc_ckpt = _load_model(HBMFSI_MCMC_PATH)

    return {
        "Baseline MF": {
            "model": mf_model,
            "meta": mf_ckpt,
        },
        "Baseline MF (VI)": {
            "model": bayes_mf_vi_model,
            "meta": bayes_ckpt,
        },
        "Core VI": {
            "model": hbmfsi_vi_model,
            "meta": hb_vi_ckpt,
        },
        "Core MCMC": {
            "model": mcmc_obj,
            "meta": mcmc_ckpt,
        },
    }



# ---------------------------------------------------------
# Feature builders
# ---------------------------------------------------------
def build_user_feature_vector(age, gender, occupation, users_df):
    """
    Very simple user feature encoder:
      - normalized age
      - one-hot gender
      - one-hot occupation

    This may produce fewer than P=31 dims; we pad later inside the
    cold-start helper so it matches the trained A_mu dimension.
    """
    age_norm = (age - 18) / (70 - 18)
    feat = [age_norm]

    genders = ["M", "F"]
    for g in genders:
        feat.append(1.0 if gender == g else 0.0)

    occs = sorted(users_df["occupation"].unique())
    for occ in occs:
        feat.append(1.0 if occ == occupation else 0.0)

    return np.array(feat, dtype=np.float32)


def build_item_feature_vector(movie_row):
    """(Not currently used in the app; kept for completeness)."""
    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Childrens",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western",
    ]
    feat = movie_row[genre_cols].astype(float).values
    return feat.astype(np.float32)


# ---------------------------------------------------------
# Small helpers to unwrap state_dicts for VI models
# ---------------------------------------------------------
def _unwrap_state_dict(model_obj):
    """
    For Bayesian MF VI / HBMFSI VI checkpoints, pull out the actual state_dict.
    """
    if isinstance(model_obj, dict) and "model_state_dict" in model_obj:
        return model_obj["model_state_dict"]
    # Plain state_dict (MF baseline)
    if isinstance(model_obj, dict):
        return model_obj
    # nn.Module case (not used here, but safe)
    return model_obj.state_dict()


def _extract_user_item_from_state_dict(sd):
    """
    Try to extract user/item latent matrices from a VI state_dict.
    """
    possible_user_keys = ["user_mu", "user_factors.weight"]
    possible_item_keys = ["item_mu", "item_factors.weight"]

    user_key = next((k for k in possible_user_keys if k in sd), None)
    item_key = next((k for k in possible_item_keys if k in sd), None)

    if user_key is None or item_key is None:
        raise KeyError(
            f"Could not infer user/item latent matrices from state_dict keys: {list(sd.keys())}"
        )

    U = sd[user_key]
    V = sd[item_key]
    return U, V


def _extract_A_from_state_dict(sd):
    """Extract A_mu (user feature mapping) from HBMFSI VI state_dict."""
    if "A_mu" not in sd:
        raise KeyError(
            f"Could not infer A_mu from state_dict keys: {list(sd.keys())}"
        )
    return sd["A_mu"]


def _extract_UV_from_mcmc(mcmc_obj):
    """
    For HBMFSI MCMC checkpoint, get sampled U, V tensors.

    Expected structure:
        mcmc_obj["samples"]["U"] : (S, U_sub, K)
        mcmc_obj["samples"]["V"] : (S, I_sub, K)
    """
    if isinstance(mcmc_obj, dict) and "samples" in mcmc_obj:
        samples = mcmc_obj["samples"]
        return samples["U"], samples["V"]
    # Backwards-compatible fallback
    return mcmc_obj["U"], mcmc_obj["V"]


def _pad_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate 1D numpy array to a target dimension."""
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    # pad with zeros
    pad_len = target_dim - vec.shape[0]
    return np.concatenate([vec, np.zeros(pad_len, dtype=vec.dtype)], axis=0)


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------
def top_n_for_existing_user_mf(model_obj, user_idx, all_item_indices, N=10):
    """
    Baseline MF warm-start prediction:
      r_hat = global_mean + U[user] · V[item]^T

    model_obj: plain state_dict with keys:
      - 'user_factors.weight'
      - 'item_factors.weight'
      - optional 'global_mean'
    """
    device = torch.device("cpu")
    if not isinstance(model_obj, dict):
        sd = model_obj.state_dict()
    else:
        sd = model_obj

    U = sd["user_factors.weight"].to(device)
    V = sd["item_factors.weight"].to(device)
    gmean = float(sd["global_mean"]) if "global_mean" in sd else 0.0

    u_vec = U[user_idx]  # (K,)
    scores = gmean + V @ u_vec
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_existing_user_bayes_mf_vi(model_obj, user_idx, all_item_indices, N=10):
    """
    Bayesian MF VI warm-start using posterior means user_mu, item_mu.
    """
    device = torch.device("cpu")
    sd = _unwrap_state_dict(model_obj)
    U, V = _extract_user_item_from_state_dict(sd)
    U = U.to(device)
    V = V.to(device)

    gmean = float(model_obj.get("global_mean", 0.0)) if isinstance(model_obj, dict) else 0.0

    u_vec = U[user_idx]
    scores = gmean + V @ u_vec
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_existing_user_core_vi(model_obj, user_idx, all_item_indices, N=10):
    """
    HBMFSI Core VI warm-start using user_mu, item_mu + global_mean.
    """
    device = torch.device("cpu")
    sd = _unwrap_state_dict(model_obj)
    U, V = _extract_user_item_from_state_dict(sd)
    U = U.to(device)
    V = V.to(device)

    gmean = float(model_obj.get("global_mean", 0.0)) if isinstance(model_obj, dict) else 0.0

    u_vec = U[user_idx]
    scores = gmean + V @ u_vec
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_existing_user_mcmc(mcmc_obj, user_idx, all_item_indices, N=10):
    """
    HBMFSI Core MCMC warm-start:
    use posterior mean over samples for U, V.
    """
    device = torch.device("cpu")
    U_s, V_s = _extract_UV_from_mcmc(mcmc_obj)  # (S, U_sub, K), (S, I_sub, K)
    U_mean = U_s.mean(dim=0).to(device)
    V_mean = V_s.mean(dim=0).to(device)
    gmean = float(mcmc_obj.get("global_mean", 0.0)) if isinstance(mcmc_obj, dict) else 0.0

    u_vec = U_mean[user_idx]
    scores = gmean + V_mean @ u_vec
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


# ---------- Cold / semi-cold helpers ----------

def _fold_in_user_from_likes_vi(V: torch.Tensor,
                                liked_item_idx,
                                like_rating=DEFAULT_LIKE_RATING,
                                num_steps=80,
                                lr=0.05,
                                l2=0.01):
    """
    Generic fold-in for a NEW user with only a few liked movies.

    Optimizes a new latent vector u_new so that V @ u_new gives
    high ratings for the liked items.
    """
    device = V.device
    K = V.shape[1]
    u_new = torch.zeros(K, device=device, requires_grad=True)

    item_idx_t = torch.tensor(liked_item_idx, dtype=torch.long, device=device)
    r_target = torch.full((len(liked_item_idx),), like_rating, device=device)

    opt = torch.optim.SGD([u_new], lr=lr)

    for _ in range(num_steps):
        opt.zero_grad()
        r_hat = (V[item_idx_t] * u_new).sum(dim=1)
        loss = ((r_hat - r_target) ** 2).mean() + l2 * (u_new**2).mean()
        loss.backward()
        opt.step()

    return u_new.detach()


def _fold_in_user_from_features_and_likes_core_vi(model_obj,
                                                  f_user_np: np.ndarray,
                                                  liked_item_idx,
                                                  like_rating=DEFAULT_LIKE_RATING,
                                                  num_steps=80,
                                                  lr=0.05,
                                                  l2=0.01):
    """
    Core VI semi-cold fold-in:
      - start from u0 = A^T f_user (features)
      - refine u using liked movies via gradient steps
    """
    device = torch.device("cpu")
    sd = _unwrap_state_dict(model_obj)
    U_mu, V_mu = _extract_user_item_from_state_dict(sd)
    A_mu = _extract_A_from_state_dict(sd)

    V = V_mu.to(device)          # (num_items, K)
    A_mu = A_mu.to(device)       # (P, K)
    P = A_mu.shape[0]

    f_user_np = _pad_to_dim(f_user_np, P)
    f = torch.tensor(f_user_np, dtype=torch.float32, device=device)

    u_new = (f @ A_mu).detach().clone().requires_grad_(True)

    if len(liked_item_idx) == 0:
        return u_new.detach(), V

    item_idx_t = torch.tensor(liked_item_idx, dtype=torch.long, device=device)
    r_target = torch.full((len(liked_item_idx),), like_rating, device=device)

    opt = torch.optim.SGD([u_new], lr=lr)

    for _ in range(num_steps):
        opt.zero_grad()
        r_hat = (V[item_idx_t] * u_new).sum(dim=1)
        loss = ((r_hat - r_target) ** 2).mean() + l2 * (u_new**2).mean()
        loss.backward()
        opt.step()

    return u_new.detach(), V


def top_n_for_cold_user_baseline_mf(model_obj,
                                    movie_id_to_index,
                                    liked_movie_ids,
                                    like_rating=DEFAULT_LIKE_RATING,
                                    N=10):
    """
    Cold/semi-cold user for Baseline MF:
    only works if we have at least one liked movie (no side-info).
    """
    if len(liked_movie_ids) == 0:
        raise ValueError("Baseline MF needs at least one liked movie for cold-start.")

    device = torch.device("cpu")
    sd = model_obj if isinstance(model_obj, dict) else model_obj.state_dict()
    V = sd["item_factors.weight"].to(device)
    gmean = float(sd["global_mean"]) if "global_mean" in sd else 0.0

    liked_idx = [movie_id_to_index[mid] for mid in liked_movie_ids]
    u_new = _fold_in_user_from_likes_vi(V, liked_idx, like_rating=like_rating)

    scores = gmean + V @ u_new
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_cold_user_baseline_mf_vi(model_obj,
                                       movie_id_to_index,
                                       liked_movie_ids,
                                       like_rating=DEFAULT_LIKE_RATING,
                                       N=10):
    """
    Cold/semi-cold user for Baseline MF (VI):
    uses item_mu and learns a new user vector from liked movies.
    """
    if len(liked_movie_ids) == 0:
        raise ValueError("Baseline MF (VI) needs at least one liked movie for cold-start.")

    device = torch.device("cpu")
    sd = _unwrap_state_dict(model_obj)
    _, V = _extract_user_item_from_state_dict(sd)
    V = V.to(device)

    liked_idx = [movie_id_to_index[mid] for mid in liked_movie_ids]
    u_new = _fold_in_user_from_likes_vi(V, liked_idx, like_rating=like_rating)

    gmean = float(model_obj.get("global_mean", 0.0)) if isinstance(model_obj, dict) else 0.0
    scores = gmean + V @ u_new
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_cold_user_core_vi(model_obj,
                                f_user_np: np.ndarray,
                                movie_id_to_index,
                                liked_movie_ids,
                                like_rating=DEFAULT_LIKE_RATING,
                                N=10):
    """
    Core VI cold/semi-cold user:
      - if no liked movies -> pure feature cold-start (A^T f)
      - if liked movies -> semi-cold fold-in
    """
    device = torch.device("cpu")
    sd = _unwrap_state_dict(model_obj)
    _, V_mu = _extract_user_item_from_state_dict(sd)
    A_mu = _extract_A_from_state_dict(sd)

    V = V_mu.to(device)
    A_mu = A_mu.to(device)
    P = A_mu.shape[0]

    f_user_np = _pad_to_dim(f_user_np, P)

    if len(liked_movie_ids) == 0:
        f = torch.tensor(f_user_np, dtype=torch.float32, device=device)
        u_new = f @ A_mu
    else:
        liked_idx = [movie_id_to_index[mid] for mid in liked_movie_ids]
        u_new, V = _fold_in_user_from_features_and_likes_core_vi(
            model_obj,
            f_user_np,
            liked_idx,
            like_rating=like_rating,
        )

    gmean = float(model_obj.get("global_mean", 0.0)) if isinstance(model_obj, dict) else 0.0
    scores = gmean + V @ u_new
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


def top_n_for_cold_user_core_mcmc(mcmc_obj,
                                  core_vi_obj,
                                  f_user_np: np.ndarray,
                                  movie_id_to_index,
                                  liked_movie_ids,
                                  like_rating=DEFAULT_LIKE_RATING,
                                  N=10):
    """
    Core MCMC cold/semi-cold user:
    reuse the Core VI fold-in user vector (so likes affect both),
    but score with MCMC posterior mean item factors.
    """
    device = torch.device("cpu")

    # Get u_new from Core VI logic
    u_new, _V_dummy = _fold_in_user_from_features_and_likes_core_vi(
        core_vi_obj,
        f_user_np,
        [movie_id_to_index[mid] for mid in liked_movie_ids]
        if len(liked_movie_ids) > 0
        else [],
        like_rating=like_rating,
    )

    U_s, V_s = _extract_UV_from_mcmc(mcmc_obj)
    V_mean = V_s.mean(dim=0).to(device)

    gmean = float(mcmc_obj.get("global_mean", 0.0)) if isinstance(mcmc_obj, dict) else 0.0
    scores = gmean + V_mean @ u_new.to(device)
    scores = scores.detach().cpu().numpy()
    idx_sorted = np.argsort(-scores)
    top_idx = idx_sorted[:N]
    return top_idx, scores[top_idx]


# ---------------------------------------------------------
# Small helper: show recommendations table
# ---------------------------------------------------------
def make_recs_table(top_idx, scores, movies_df):
    rec_movies = movies_df.iloc[top_idx][["movie_id", "title"]].copy()
    rec_movies["pred_rating"] = scores
    return rec_movies


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Bayesian Recommender Dashboard", layout="wide")

    st.title("🎬 Bayesian Movie Recommender — Warm & Cold Start")

    st.markdown(
        """
        This dashboard compares four models trained on MovieLens 100K:

        - **Baseline MF** — standard matrix factorization baseline  
        - **Baseline MF (VI)** — Bayesian MF (no side-information)  
        - **Core VI** — Hierarchical Bayesian MF with side-information (VI)  
        - **Core MCMC** — Hierarchical Bayesian MF with full MCMC on a warm subset  

        Use the controls in the sidebar to choose a **scenario** and a **model**,
        then click **Recommend** to see that model's top-\(N\) movies.
        """
    )

    users_df, movies_df = load_movielens_metadata()
    movie_id_to_index = build_movie_id_to_index(movies_df)
    models = load_models()

    # -----------------------------------------------------
    # Sidebar controls
    # -----------------------------------------------------
    st.sidebar.header("Controls")

    scenario = st.sidebar.radio(
        "Scenario",
        ["Warm existing user", "Cold-start user"],
        index=0,
    )

    model_name = st.sidebar.selectbox(
        "Model",
        ["Baseline MF", "Baseline MF (VI)", "Core VI", "Core MCMC"],
        index=2,  # default Core VI
    )

    top_n = st.sidebar.slider("Top-N recommendations", 3, 20, TOP_N_DEFAULT)

    # -----------------------------------------------------
    # Metrics / capabilities panel
    # -----------------------------------------------------
    st.subheader("Model Capabilities & Test Performance (Summary)")

    metrics = pd.DataFrame(
        [
            [
                "Baseline MF",
                "Warm + Cold (partial)",
                "No",
                "No",
                0.9395,
                0.7403,
                1.0944,
                0.9219,
                1.1070,
                0.9286,
            ],
            [
                "Baseline MF (VI)",
                "Warm only",
                "No",
                "Yes",
                1.0465,
                0.8557,
                None,
                None,
                None,
                None,
            ],
            [
                "Core VI",
                "Warm + Cold",
                "Yes",
                "Yes",
                0.9336,
                0.7418,
                0.9891,
                0.7951,
                1.0196,
                0.8352,
            ],
            [
                "Core MCMC",
                "Warm + Cold (subset)",
                "Yes",
                "Yes",
                0.9331,
                0.7367,
                0.9299,
                0.7469,
                1.0792,
                0.8686,
            ],
        ],
        columns=[
            "Model",
            "Coverage",
            "Uses side info",
            "Uncertainty",
            "RMSE (warm)",
            "MAE (warm)",
            "RMSE (cold-users)",
            "MAE (cold-users)",
            "RMSE (cold-items)",
            "MAE (cold-items)",
        ],
    )

    st.dataframe(metrics, use_container_width=True)
    st.caption(
        "Predicted rating (`pred_rating`) shown below is the model's **expected rating on the 1–5 scale**."
    )

    # -----------------------------------------------------
    # Scenario 1: Warm existing user
    # -----------------------------------------------------
    if scenario == "Warm existing user":
        st.subheader(f"Warm-start: Existing User — {model_name}")

        # MCMC subset has only 250 users:
        max_uid = 250 if model_name == "Core MCMC" else int(users_df["user_id"].max())

        user_id = st.number_input(
            f"MovieLens User ID (1–{max_uid})",
            min_value=1,
            max_value=max_uid,
            value=min(10, max_uid),
        )
        user_idx = user_id - 1  # 0-based

        if st.button("Recommend for this user"):
            with st.spinner("Computing recommendations..."):
                all_item_indices = np.arange(len(movies_df))

                if model_name == "Baseline MF":
                    mf_model = models["Baseline MF"]["model"]
                    top_idx, scores = top_n_for_existing_user_mf(
                        mf_model, user_idx, all_item_indices, N=top_n
                    )

                elif model_name == "Baseline MF (VI)":
                    bayes_model = models["Baseline MF (VI)"]["model"]
                    top_idx, scores = top_n_for_existing_user_bayes_mf_vi(
                        bayes_model, user_idx, all_item_indices, N=top_n
                    )

                elif model_name == "Core VI":
                    core_vi = models["Core VI"]["model"]
                    top_idx, scores = top_n_for_existing_user_core_vi(
                        core_vi, user_idx, all_item_indices, N=top_n
                    )

                else:  # Core MCMC
                    mcmc_obj = models["Core MCMC"]["model"]
                    top_idx, scores = top_n_for_existing_user_mcmc(
                        mcmc_obj, user_idx, all_item_indices, N=top_n
                    )

                rec_table = make_recs_table(top_idx, scores, movies_df)
                st.dataframe(rec_table, use_container_width=True)

    # -----------------------------------------------------
    # Scenario 2: Cold-start user (Age/Gender/Occupation)
    # -----------------------------------------------------
    else:
        st.subheader(f"Cold-start: New User — {model_name}")

        colu, colv = st.columns(2)
        with colu:
            age = st.number_input("Age", min_value=10, max_value=80, value=25)
            gender = st.selectbox("Gender", ["M", "F"])
        with colv:
            occupation = st.selectbox(
                "Occupation", sorted(users_df["occupation"].unique())
            )

        st.markdown("*(Optional)* Provide 1–2 movies you already like:")

        movie_choices = movies_df[["movie_id", "title"]].copy()
        movie_choices["label"] = (
            movie_choices["movie_id"].astype(str) + " — " + movie_choices["title"]
        )

        selected_labels = st.multiselect(
            "Pick movies you like (for semi-cold update):",
            options=list(movie_choices["label"]),
            default=[],
        )

        liked_movie_ids = []
        for lbl in selected_labels:
            mid = int(lbl.split(" — ")[0])
            liked_movie_ids.append(mid)

        f_user = build_user_feature_vector(age, gender, occupation, users_df)

        if st.button("Recommend for this profile"):
            with st.spinner("Computing cold-start recommendations..."):
                try:
                    if model_name == "Baseline MF":
                        if len(liked_movie_ids) == 0:
                            st.warning(
                                "Baseline MF cannot use only features for cold-start. "
                                "Please select at least one movie you like, or switch to a Core model."
                            )
                            return
                        mf_model = models["Baseline MF"]["model"]
                        top_idx, scores = top_n_for_cold_user_baseline_mf(
                            mf_model,
                            movie_id_to_index,
                            liked_movie_ids,
                            like_rating=DEFAULT_LIKE_RATING,
                            N=top_n,
                        )

                    elif model_name == "Baseline MF (VI)":
                        if len(liked_movie_ids) == 0:
                            st.warning(
                                "Baseline MF (VI) cannot use only features for cold-start. "
                                "Please select at least one movie you like, or switch to a Core model."
                            )
                            return
                        bayes_model = models["Baseline MF (VI)"]["model"]
                        top_idx, scores = top_n_for_cold_user_baseline_mf_vi(
                            bayes_model,
                            movie_id_to_index,
                            liked_movie_ids,
                            like_rating=DEFAULT_LIKE_RATING,
                            N=top_n,
                        )

                    elif model_name == "Core VI":
                        core_vi = models["Core VI"]["model"]
                        top_idx, scores = top_n_for_cold_user_core_vi(
                            core_vi,
                            f_user,
                            movie_id_to_index,
                            liked_movie_ids,
                            like_rating=DEFAULT_LIKE_RATING,
                            N=top_n,
                        )

                    else:  # Core MCMC
                        core_vi = models["Core VI"]["model"]
                        mcmc_obj = models["Core MCMC"]["model"]
                        top_idx, scores = top_n_for_cold_user_core_mcmc(
                            mcmc_obj,
                            core_vi,
                            f_user,
                            movie_id_to_index,
                            liked_movie_ids,
                            like_rating=DEFAULT_LIKE_RATING,
                            N=top_n,
                        )

                    rec_table = make_recs_table(top_idx, scores, movies_df)
                    st.dataframe(rec_table, use_container_width=True)

                except ValueError as e:
                    st.warning(str(e))

        st.markdown(
            """
            🔎 **Notes**

            - If you do **not** select any movies, Core models use only your
              age/gender/occupation features (pure cold-start).  
            - If you **do** pick 1–2 movies you like, the model does a small
              **semi-cold fold-in** so those likes directly influence the
              recommended list.
            - Baseline MF models cannot use feature-only cold-start; they need at
              least one liked movie.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        **Summary**

        - *Baseline MF* and *Baseline MF (VI)* are strong warm-start baselines.  
        - *Core VI* and *Core MCMC* use side information to handle **true cold-start**,
          while also giving competitive or better RMSE/MAE.  
        - *Core MCMC* additionally exposes full posterior uncertainty over ratings.
        """
    )


if __name__ == "__main__":
    main()
