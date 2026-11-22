from __future__ import annotations

from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        if n == 0:
            raise ValueError("Input array X must contain at least one sample.")

        # Step 1: Pre-compute centroid, distances to centroid, and pairwise distances
        centroid = X.mean(axis=0)
        d = np.linalg.norm(X - centroid, axis=1)

        diff = X[:, None, :] - X[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dist_sq, 0.0))

        if self.sigma is None:
            if n > 1:
                upper_idx = np.triu_indices(n, k=1)
                sigma = float(0.2 * np.median(dists[upper_idx]))
            else:
                sigma = 1.0
        else:
            sigma = float(self.sigma)

        sigma = max(sigma, 1e-12)
        tau = 0.1 * sigma

        # k-NN graph stored per node: nbrs[i] are i's k_eff nearest neighbours
        # (closest first) and w_ij[i, m] is the weight of edge i -> nbrs[i, m].
        k_eff = max(min(self.k, n - 1), 0)
        order = np.argsort(dists, axis=1, kind="stable")
        nbrs = order[:, 1 : k_eff + 1]
        rows = np.arange(n)[:, None]
        w_ij = np.exp(-dist_sq[rows, nbrs] / (sigma ** 2))

        # Step 2: Initialize flux
        flux = np.ones(n, dtype=np.float64)

        # Step 3: Simulate flow for T iterations
        downhill_mask = d[nbrs] <= d[:, None] + tau
        last_delta = np.zeros((n, k_eff), dtype=np.float64)

        for t in range(self.T):
            if t == 0 or self.beta == 0:
                phi = np.ones(n, dtype=np.float64)
            elif self.beta is None:
                phi = np.log1p(np.maximum(flux, 0.0))
            else:
                phi = np.power(np.maximum(flux, 0.0), self.beta)
            w_tilde = np.where(downhill_mask, w_ij * phi[nbrs], 0.0)

            denom = w_tilde.sum(axis=1, keepdims=True)
            safe_denom = np.where(denom > 0.0, denom, 1.0)
            delta = np.where(
                denom > 0.0, self.alpha * w_tilde * (flux[:, None] / safe_denom), 0.0
            )

            outgoing = delta.sum(axis=1)
            incoming = np.zeros(n, dtype=np.float64)
            np.add.at(incoming, nbrs, delta)
            flux = flux - outgoing + incoming
            last_delta = delta

        # Step 4: Detect sinks
        final_outgoing = last_delta.sum(axis=1) if self.T > 0 else np.zeros(n, dtype=np.float64)
        sink_indices = np.flatnonzero(final_outgoing < self.epsilon)
        if sink_indices.size == 0:
            sink_indices = np.array([int(np.argmin(d))], dtype=int)

        sink_list = [int(s) for s in sink_indices]

        # Step 5: Assign clusters via steepest downhill paths with cycle detection
        labels = np.full(n, -1, dtype=int)
        next_label = 0
        for s in sink_list:
            labels[s] = next_label
            next_label += 1

        for i in range(n):
            if labels[i] != -1:
                continue

            curr = i
            path: list[int] = []
            visited_in_path: set[int] = set()

            while labels[curr] == -1:
                if curr in visited_in_path:
                    # Cycle detected: designate current node as a new sink
                    labels[curr] = next_label
                    sink_list.append(curr)
                    next_label += 1
                    break

                visited_in_path.add(curr)
                path.append(curr)

                valid = downhill_mask[curr]
                if not np.any(valid):
                    labels[curr] = next_label
                    sink_list.append(curr)
                    next_label += 1
                    break

                cand_delta = np.where(valid, last_delta[curr], -np.inf)
                curr = int(nbrs[curr, int(np.argmax(cand_delta))])

            target_label = int(labels[curr])
            for node in path:
                labels[node] = target_label

        # Step 6: Post-processing - merge small clusters into nearest larger clusters
        s_min = max(2, n // 20)
        while True:
            unique_lbls, counts = np.unique(labels, return_counts=True)
            small_lbls = [int(lbl) for lbl, cnt in zip(unique_lbls, counts) if cnt < s_min]
            large_lbls = [int(lbl) for lbl, cnt in zip(unique_lbls, counts) if cnt >= s_min]

            if not small_lbls or not large_lbls:
                break

            small_lbl = small_lbls[0]
            mu_small = X[labels == small_lbl].mean(axis=0)

            best_large = None
            best_dist = np.inf
            for large_lbl in large_lbls:
                mu_large = X[labels == large_lbl].mean(axis=0)
                dist = float(np.linalg.norm(mu_small - mu_large))
                if dist < best_dist:
                    best_dist = dist
                    best_large = large_lbl

            if best_large is not None:
                labels[labels == small_lbl] = best_large
            else:
                break

        # Boolean mask over nodes: True where the node is a sink
        sinks = np.zeros(n, dtype=bool)
        sinks[sink_list] = True

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"