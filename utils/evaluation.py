import sklearn
import numpy as np
import spacy


""" Helper functions """


def continuous_mutual_info(mus, ys):
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in range(num_attributes):
        m[:, i] = sklearn.feature_selection.mutual_info_regression(
            mus, ys[:, i])
    return m


def continuous_entropy(ys):
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in tqdm(range(num_factors)):
        h[j] = mutual_info_regression(
            ys[:, j].reshape(-1, 1), ys[:, j]
        )
    return h


def _compute_score_matrix(mus, ys):
    "Score matrix given by linear regression"
    # from evaluation
    num_latent_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes])
    for i in range(num_latent_codes):
        for j in range(num_attributes):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            # Attributes are considered continuous.
            cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
            cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
            var_mu = cov_mu_i_y_j[0, 0]
            var_y = cov_mu_i_y_j[1, 1]
            if var_mu > 1e-12:
                score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
            else:
                score_matrix[i, j] = 0.
    return score_matrix


def _compute_avg_diff_top_two(matrix):
    # from evaluation
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def _compute_correlation_matrix(mus, ys):
    # from evaluation
    num_latent_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes])
    for i in range(num_latent_codes):
        for j in range(num_attributes):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            rho, p = scipy.stats.spearmanr(mu_i, y_j)
            if p <= 0.05:
                score_matrix[i, j] = np.abs(rho)
            else:
                score_matrix[i, j] = 0.
    return score_matrix


""" Actual Metrics """

# TODO correct?
def compute_interpretability_metric(latent_codes, attributes, attr_list):
    # from evaluation
    interpretability_metrics = {}
    total = 0
    for i, attr_name in tqdm(enumerate(attr_list), desc="Interpretability"):
        attr_labels = attributes[:, i]
        mutual_info = sklearn.feature_selection.mutual_info_regression(
            latent_codes, attr_labels)
        dim = np.argmax(mutual_info)

        reg = sklearn.linear_model.LinearRegression().fit(
            latent_codes[:, dim:dim+1], attr_labels)
        score = reg.score(latent_codes[:, dim:dim+1], attr_labels)
        interpretability_metrics[attr_name] = (int(dim), float(score))
        total += float(score)
    interpretability_metrics["mean"] = (-1, total/len(attr_list))
    return interpretability_metric


def compute_mig(latent_codes, attributes):
    # from evaluation
    score_dict = {}
    m = continuous_mutual_info(latent_codes, attributes)
    entropy = continuous_entropy(attributes)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    )
    return score_dict


def compute_sap_score():
    # from evaluation
    score_matrix = _compute_score_matrix(latent_codes, attributes)
    # TODO necessary?
    # Score matrix should have shape [num_codes, num_attributes].
    assert score_matrix.shape[0] == latent_codes.shape[1]
    assert score_matrix.shape[1] == attributes.shape[1]

    scores = {
        "SAP_score": _compute_avg_diff_top_two(score_matrix)
    }


def compute_correlation_score():
    # from evaluation
    corr_matrix = _compute_correlation_matrix(latent_codes, attributes)
    scores = {
        "Corr_score": np.mean(np.max(corr_matrix, axis=0))
    }
    return scores
