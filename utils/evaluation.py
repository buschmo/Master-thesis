# Calcuations
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
import numpy as np
from torch import Tensor
import scipy

# Typing
from typing import Tuple
from multiprocessing import Queue

""" Code primarily taken from ar-vae evaluation.py """

""" Helper functions """


def continuous_mutual_info(mus: Tensor, ys: Tensor) -> np.array:
    """ Estimates the empirical mutual information on continuous attributes.

    Needed for MIG.

    Args:
        mus (Tensor): latent code
        ys (Tensor): attribute

    Returns:
        np.array: values of [code, attribute] pairing
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    # calculate MI for each attribute with every latent z
    for j in range(num_attributes):
        m[:, j] = mutual_info_regression(
            mus, ys[:, j])
    return m


def discrete_mutual_info(mus: Tensor, ys: Tensor) -> np.array:
    """ Estimates the empirical mutual information on discrete attributes.

    Needed for MIG.

    Args:
        mus (Tensor): latent code
        ys (Tensor): attribute

    Returns:
        np.array: values of [code, attribute] pairing
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in range(num_codes):
        for j in range(num_attributes):
            m[i, j] = mutual_info_score(ys[:, j], mus[:, i])
    return m


def continuous_entropy(ys: Tensor) -> np.array:
    """ Computes entropy for continuous attribute values

    Needed for MIG.

    Args:
        ys (Tensor): attributes

    Returns:
        np.array: entropy for each attribute dimension
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    # calculate MI for each attribute
    for j in range(num_factors):
        h[j] = mutual_info_regression(
            ys[:, j].reshape(-1, 1), ys[:, j]
        )
    return h


def discrete_entropy(ys: Tensor) -> np.array:
    """ Computes entropy for discrete attribute values

    Needed for MIG.

    Args:
        ys (Tensor): attributes

    Returns:
        np.array: entropy for each attribute dimension
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    # calculate MI for each attribute
    for j in range(num_factors):
        # H(Y) = I(Y|Y) in discrete case
        h[j] = mutual_info_score(ys[:, j], ys[:, j])


def _compute_score_matrix(mus: Tensor, ys: Tensor) -> np.array:
    """ Compute score matrix given for continuous attributes by linear regression

    Needed for SAP score.
    Takes the R**2 score obtained with fitting a line that minimizes the linear regression error.

    Args:
        mus (Tensor): latent code
        ys (Tensor): attributes

    Returns:
        np.array: score matrix
    """
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


def _compute_discrete_score_matrix(mus: Tensor, ys: Tensor) -> np.array:
    """ Compute score matrix S given for discrete attributes by linear regression

    Needed for SAP score.
    Fits one or more thresholds directly on the i-th latent variable minimizing the balanced classification errors.
    S_{i,j} is the balanced classification accuracy for the j-th attribute.

    Balanced classification accuracy: sum(#correct / #all in class) / #classes

    Args:
        mus (Tensor): latent code
        ys (Tensor): attributes

    Returns:
        np.array: score matrix
    """
    # TODO implement
    pass


def _compute_avg_diff_top_two(matrix: Tensor) -> float:
    """ Computes the difference between the two highest values in matrix

    Needed for SAP score.

    Args:
        matrix (Tensor): matrix with values

    Returns:
        float: difference between 1st and 2nd highest value
    """
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def _compute_correlation_matrix(mus: Tensor, ys: Tensor) -> np.array:
    """ Computes the correlation matrix of two tensors

    Needed for spearman's rank correlation.

    Args:
        mus (Tensor): latent code z
        ys (Tensor): attributes a

    Returns:
        np.array: correlation matrix
    """
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


def compute_interpretability_metric(latent_codes: Tensor, attributes: Tensor, attr_list: list[str], queue: Queue) -> dict[str, dict[str, float]]:
    """ Computes the interpretability score

    Based on Adel et al (2018) - Discovering Interpretable Representations for Both Deep Generative and Discriminative Models
    Chapter 4

    Beware, this does not focus on the disentangled dimension, but considers all dimensions.

    Args:
        latent_codes (Tensor): latent code z
        attributes (Tensor): attributes to be interpreted
        attr_list (list[str]): names for the attributes
        queue (Queue): to return the result in a parallelized call

    Returns:
        dict[str, Tuple[int, float]]: mapping attribute names to tuple of dimension and value. Also containt key "mean" with dimension -1
    """
    interpretability_metrics = {}
    total = 0
    for i, attr_name in enumerate(attr_list):
        # get i-th attribute value
        attr_values = attributes[:, i]
        # (i) get maximal informative dimension of latent
        mutual_info = mutual_info_regression(
            latent_codes, attr_values)
        dim = np.argmax(mutual_info)

        # (ii) measure interpretability by using a simple probabilistic relationship
        reg = LinearRegression().fit(
            latent_codes[:, dim:dim+1], attr_values)
        score = reg.score(latent_codes[:, dim:dim+1], attr_values)
        interpretability_metrics[attr_name] = (int(dim), float(score))
        total += float(score)
    interpretability_metrics["Mean"] = (-1, total/len(attr_list))
    scores = {"Interpretability": interpretability_metrics}
    if queue:
        queue.put(scores)
    else:
        return scores


def compute_mig(latent_codes: Tensor, attributes: Tensor, queue: Queue) -> dict[str, float]:
    """ Computes the mutual information gap for continuous attributes

    Based on Chen et al (2018) - Isolating Sources of Disentanglement in Variational Autoencoders
    Chapter 4.1

    Args:
        latent_codes (Tensor): latent code z
        attributes (Tensor): attributes a
        queue (Queue): to return the result in a parallelized call

    Returns:
        dict[str, float]: key "mig" with score
    """
    scores = {}
    # Equation (5)
    m = continuous_mutual_info(latent_codes, attributes)
    entropy = continuous_entropy(attributes)
    sorted_m = np.sort(m, axis=0)[::-1]
    # Equation (6)
    scores = {
        "Mutual Information Gap": np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    }
    if queue:
        queue.put(scores)
    else:
        return scores


def compute_discrete_mig(latent_codes: Tensor, attributes: Tensor) -> dict[str, float]:
    """ Computes the mutual information gap for discrete attributes

    Based on Chen et al (2018) - Isolating Sources of Disentanglement in Variational Autoencoders
    Chapter 4.1

    Args:
        latent_codes (Tensor): latent code z
        attributes (Tensor): attributes a

    Returns:
        dict[str, float]: key "mig" with score
    """
    # TODO implement
    # this should work similar to compute_mig, but swap for discrete mi and entropy
    pass


def compute_sap_score(latent_codes: Tensor, attributes: Tensor, queue: Queue) -> dict[str, float]:
    """ Computes the SAP score

    Based on Kumar et al (2018) - Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    Chapter 3

    Args:
        latent_codes (Tensor): latent code z
        attributes (Tensor): attributes a
        queue (Queue): to return the result in a parallelized call

    Returns:
        dict[str, float]: key "SAP_score" with value
    """
    # (i) Matrix with linear regression score
    score_matrix = _compute_score_matrix(latent_codes, attributes)
    # Score matrix should have shape [num_codes, num_attributes].
    assert score_matrix.shape[0] == latent_codes.shape[1]
    assert score_matrix.shape[1] == attributes.shape[1]

    # (ii) for each attribute, take the difference of the top two entries
    scores = {
        "Separated Attribute Predictability": _compute_avg_diff_top_two(score_matrix)
    }
    if queue:
        queue.put(scores)
    else:
        return scores


def compute_correlation_score(latent_codes: Tensor, attributes: Tensor, queue: Queue) -> dict[str, float]:
    """ Calculate spearman's rank correlation

    Based on Spearman (1904) - The Proof and Measurement of Association between Two Things

    Args:
        latent_codes (Tensor): latent codes z
        attributes (Tensor): attributes a
        queue (Queue): to return the result in a parallelized call

    Returns:
        dict[str, float]: key "Corr_score" giving the spearman score
    """
    corr_matrix = _compute_correlation_matrix(latent_codes, attributes)
    scores = {
        "Spearman's Rank Correlation": np.mean(np.max(corr_matrix, axis=0))
    }
    if queue:
        queue.put(scores)
    else:
        return scores
