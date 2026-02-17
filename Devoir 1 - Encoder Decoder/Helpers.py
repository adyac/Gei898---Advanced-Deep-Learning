def normalize(data):
    """
    normalization par capteur.
    Returns:
        normalized_data
        mu
        sigma
    """
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)

    normalized_data = (data - mu) / (sigma)
    return normalized_data, mu, sigma