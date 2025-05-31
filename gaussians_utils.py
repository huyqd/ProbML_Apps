import jax
from jax import numpy as jnp
from gaussians import Gaussian

# Import necessary libraries
import jax.random as jrandom

jax.config.update("jax_enable_x64", True)


def _select_data(X_all, Y_all, sigma_noise, selected_indices):
    """Selects data points and prepares noise covariance for observed data."""
    if len(selected_indices) == 0:
        return None, None, None

    X_select = X_all[list(selected_indices)]
    Y_select = Y_all[list(selected_indices)]
    # Noise covariance matrix for selected data (assuming iid noise)
    Lambda_select_sq = sigma_noise**2 * jnp.eye(len(selected_indices))

    return X_select, Y_select, Lambda_select_sq


def phi(x):
    """
    Feature function for simple linear regression: [1, x]
    Accepts a single scalar x or a JAX array of shape (N, 1).
    Returns a JAX array of shape (N, num_features) or (num_features,).
    """
    if jnp.ndim(x) == 0:  # Handle scalar input
        return jnp.array([1.0, x])
    else:  # Handle array input (N, 1)
        return jnp.hstack([jnp.ones_like(x), x])


def create_prior_distribution(mu0_prior, mu1_prior, s11_prior, s22_prior, rho_prior):
    """Constructs the prior Gaussian distribution from scalar parameters."""
    mu_prior = jnp.asarray([mu0_prior, mu1_prior])
    # Ensure s11 and s22 are positive for sqrt
    s11_safe = jnp.maximum(s11_prior, 1e-6)
    s22_safe = jnp.maximum(s22_prior, 1e-6)
    # Ensure rho is within valid range [-1, 1]
    rho_safe = jnp.clip(rho_prior, -0.999, 0.999)

    S12_prior = rho_safe * jnp.sqrt(s11_safe * s22_safe)
    Sigma_prior = jnp.asarray([[s11_safe, S12_prior], [S12_prior, s22_safe]])

    return Gaussian(mu=mu_prior, Sigma=Sigma_prior)


def compute_posterior(prior_dist: Gaussian, X_select, Y_select, Lambda_select_sq):
    """Computes the posterior distribution given prior and selected data."""
    if X_select is not None and Y_select is not None and Lambda_select_sq is not None:
        # The .condition() method of the Gaussian class computes the posterior
        # Assumes phi is accessible (e.g., globally defined or imported)
        posterior_dist = prior_dist.condition(phi(X_select), Y_select, Lambda_select_sq)
    else:
        # If no data is selected, the posterior is the same as the prior
        posterior_dist = prior_dist
    return posterior_dist


def generate_parameter_space_data(prior_dist, posterior_dist, X_select, Y_select, key):
    """Generates data points for the parameter space plot (contours, samples, MLE)."""
    n_contour_levels = 3  # Plot contours at 1, 2, 3 standard deviations
    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    circle_pts = jnp.stack(
        [jnp.cos(theta), jnp.sin(theta)], axis=1
    )  # Unit circle points (100, 2)

    # Prior contour points
    prior_contour_pts = []
    if prior_dist.L is not None:
        for i in range(1, n_contour_levels + 1):
            pts = prior_dist.mu + i * jnp.dot(circle_pts, prior_dist.L.T)
            prior_contour_pts.append(pts)

    # Posterior contour points
    posterior_contour_pts = []
    if posterior_dist.L is not None:
        for i in range(1, n_contour_levels + 1):
            pts = posterior_dist.mu + i * jnp.dot(circle_pts, posterior_dist.L.T)
            posterior_contour_pts.append(pts)

    # Sample from prior and posterior parameter space
    num_samples_param_space = 10  # Plot a few samples
    key, subkey = jrandom.split(key)
    prior_samples_param_space = (
        prior_dist.sample(subkey, num_samples_param_space)
        if prior_dist.L is not None
        else None
    )
    key, subkey = jrandom.split(subkey)  # Use new key for posterior samples
    posterior_samples_param_space = (
        posterior_dist.sample(subkey, num_samples_param_space)
        if posterior_dist.L is not None
        else None
    )

    # Calculate MLE point if data is selected
    w_mle = None
    if X_select is not None and Y_select is not None:
        try:
            # Assumes phi is accessible
            Phi_select = phi(X_select)
            # Check if Phi_select.T @ Phi_select is invertible
            if jnp.linalg.det(jnp.dot(Phi_select.T, Phi_select)) > 1e-6:
                w_mle = jnp.linalg.solve(
                    jnp.dot(Phi_select.T, Phi_select), jnp.dot(Phi_select.T, Y_select)
                )
            # Else: not enough distinct points, MLE is not unique, don't calculate point
        except Exception as e:
            print(
                f"Warning: Error calculating MLE point: {e}"
            )  # Use print for visibility in notebook/console

    return {
        "prior_contour_pts": prior_contour_pts,
        "posterior_contour_pts": posterior_contour_pts,
        "prior_samples": prior_samples_param_space,
        "posterior_samples": posterior_samples_param_space,
        "w_mle": w_mle,
        "key": key,  # Return the updated key
    }


def generate_function_space_data(
    prior_dist,
    posterior_dist,
    X_all,
    Y_all,
    sigma_noise,
    X_select,
    Y_select,
    w_mle,
    key,
):
    """Generates data points for the function space plot (data, means, uncertainty, samples)."""
    x_plot = jnp.linspace(-5, 5, 100)[:, None]  # X values for plotting functions
    # Assumes phi is accessible
    phi_plot = phi(x_plot)  # Feature matrix for plotting

    # Prior mean function and uncertainty band (+/- 2 std dev)
    prior_mean_f = jnp.dot(phi_plot, prior_dist.mu)
    prior_var_f = jnp.sum(phi_plot * jnp.dot(phi_plot, prior_dist.Sigma), axis=1)
    prior_std_f = jnp.sqrt(prior_var_f)
    prior_upper_f = prior_mean_f + 2 * prior_std_f
    prior_lower_f = prior_mean_f - 2 * prior_std_f

    # Posterior mean function and uncertainty band (+/- 2 std dev)
    posterior_mean_f = jnp.dot(phi_plot, posterior_dist.mu)
    posterior_var_f = jnp.sum(
        phi_plot * jnp.dot(phi_plot, posterior_dist.Sigma), axis=1
    )
    posterior_std_f = jnp.sqrt(posterior_var_f)
    posterior_upper_f = posterior_mean_f + 2 * posterior_std_f
    posterior_lower_f = posterior_mean_f - 2 * posterior_std_f

    # Sample functions from prior and posterior
    num_samples_func_space = 5  # Plot a few sample functions
    key, subkey = jrandom.split(key)  # Use new key for function samples
    prior_samples_param_space_func = (
        prior_dist.sample(subkey, num_samples_func_space)
        if prior_dist.L is not None
        else None
    )
    prior_func_samples = (
        jnp.dot(phi_plot, prior_samples_param_space_func.T)
        if prior_samples_param_space_func is not None
        else None
    )

    key, subkey = jrandom.split(subkey)  # Use new key for posterior function samples
    posterior_samples_param_space_func = (
        posterior_dist.sample(subkey, num_samples_func_space)
        if posterior_dist.L is not None
        else None
    )
    posterior_func_samples = (
        jnp.dot(phi_plot, posterior_samples_param_space_func.T)
        if posterior_samples_param_space_func is not None
        else None
    )

    # Calculate MLE Function line if w_mle is available
    mle_func = jnp.dot(phi_plot, w_mle) if w_mle is not None else None

    return {
        "x_plot": x_plot,
        "all_data_x": X_all[:, 0],  # Assuming X_all is (N, 1)
        "all_data_y": Y_all,
        "sigma_noise": sigma_noise,
        "selected_data_x": X_select[:, 0] if X_select is not None else jnp.array([]),
        "selected_data_y": Y_select if Y_select is not None else jnp.array([]),
        "prior_mean_f": prior_mean_f,
        "prior_upper_f": prior_upper_f,
        "prior_lower_f": prior_lower_f,
        "prior_func_samples": prior_func_samples,
        "posterior_mean_f": posterior_mean_f,
        "posterior_upper_f": posterior_upper_f,
        "posterior_lower_f": posterior_lower_f,
        "posterior_func_samples": posterior_func_samples,
        "mle_func": mle_func,
        "key": key,  # Return the updated key
    }
