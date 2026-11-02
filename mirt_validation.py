"""R mirt reference fits and comparison tables for the MHRM dashboard.

Calibration uses identical complete responses, Q zeros, N(0,I) latent traits,
2PL slope/intercept parameterization and slope bounds. This is an estimator
cross-check, not train/test predictive cross-validation or an equivalence proof.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from mhrm_engine import marginal_loglikelihood, validate_inputs


def r_environment():
    env = os.environ.copy()
    local = Path.home() / '.cache/ai_psychometrics2/R-library'
    if not env.get('MIRT_R_LIBRARY') and local.is_dir():
        env['MIRT_R_LIBRARY'] = str(local)
    env.setdefault('OMP_NUM_THREADS', '1')
    return env


def check_mirt():
    executable = shutil.which('Rscript')
    if not executable:
        return False, 'Rscript is not installed on this server.'
    code = "p<-Sys.getenv('MIRT_R_LIBRARY');if(nzchar(p)).libPaths(c(p,.libPaths()));if(!requireNamespace('mirt',quietly=TRUE))quit(status=2);cat(as.character(packageVersion('mirt')))"
    try:
        result = subprocess.run([executable, '--vanilla', '-e', code], env=r_environment(),
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f'R mirt could not be loaded: {exc}'
    if result.returncode:
        return False, 'The R package mirt is not available on this server.'
    return True, result.stdout.strip()


def fit_mirt(data, q_matrix, *, cycles=500, burn_in=150, sem_cycles=100,
             seed=42, gain_scale=.1, decay=.75, mh_steps=5, tolerance=.001,
             timeout=300):
    q = np.asarray(q_matrix)
    if q.ndim != 2:
        raise ValueError('Q must be a J by D matrix.')
    y, q = validate_inputs(data, q.shape[1], q)
    for name, value in [('cycles', cycles), ('burn_in', burn_in), ('sem_cycles', sem_cycles), ('mh_steps', mh_steps)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > 2**31-1:
        raise ValueError('The R seed must be an integer in [0, 2^31-1].')
    if not .5 < decay <= 1 or not 0 < gain_scale <= 1:
        raise ValueError('Use decay in (0.5,1] and gain_scale in (0,1].')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be positive and finite.')
    executable = shutil.which('Rscript')
    if not executable:
        raise RuntimeError('Rscript is not installed. The Python result is still available.')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='mhrm_mirt_validation_') as folder:
        root = Path(folder)
        pd.DataFrame(y.astype(int)).to_csv(root/'responses.csv', index=False)
        pd.DataFrame(q.astype(int)).to_csv(root/'q.csv', index=False)
        command = [executable, '--vanilla', str(Path(__file__).with_suffix('.R')), str(root),
                   str(cycles), str(burn_in), str(sem_cycles), str(seed),
                   str(gain_scale), str(decay), str(mh_steps), str(tolerance)]
        try:
            process = subprocess.run(command, env=r_environment(), capture_output=True,
                                     text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'R mirt exceeded the {timeout}-second limit; no reference fit is available.') from exc
        if process.returncode:
            raise RuntimeError('R mirt calibration failed: ' + (process.stderr.strip() or process.stdout.strip()))
        try:
            estimates = pd.read_csv(root/'estimates.csv')
            metadata = pd.read_csv(root/'metadata.csv', dtype=str, keep_default_na=False).iloc[0]
            means = pd.read_csv(root/'means.csv')['mean'].to_numpy()
            covariance = pd.read_csv(root/'covariance.csv').to_numpy()
            a = estimates[[f'a{k+1}' for k in range(q.shape[1])]].to_numpy()
            d = estimates['d'].to_numpy()
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise RuntimeError('R mirt returned incomplete reference output.') from exc
        if a.shape != q.shape or d.shape != (q.shape[0],) or not np.isfinite(a).all() or not np.isfinite(d).all():
            raise RuntimeError('R mirt returned invalid item coefficients.')
        if np.any(a[~q] != 0) or np.any(a < -1e-8) or np.any(a > 10 + 1e-8):
            raise RuntimeError('The reference coefficients do not satisfy the requested loading constraints.')
        if means.shape != (q.shape[1],) or covariance.shape != (q.shape[1], q.shape[1]):
            raise RuntimeError('The R reference latent-distribution dimensions do not match.')
        if not np.allclose(means, 0, atol=1e-12) or not np.allclose(covariance, np.eye(q.shape[1]), atol=1e-12):
            raise RuntimeError('The R reference does not have the required fixed N(0,I) distribution.')
        if metadata['converged'] not in ('TRUE', 'FALSE'):
            raise RuntimeError('The R convergence flag is invalid.')
        native_ll = pd.to_numeric(metadata['native_loglik'], errors='coerce')
        return {'a': a, 'd': d, 'converged': metadata['converged'] == 'TRUE',
                'rm_cycles': int(metadata['stage3_iterations']),
                'mirt_version': metadata['mirt_version'], 'r_version': metadata['r_version'],
                'native_loglik': float(native_ll), 'seconds': time.perf_counter()-started,
                'model_specification': (root/'model.txt').read_text(),
                'messages': '\n'.join(s for s in (process.stdout.strip(), process.stderr.strip()) if s),
                'settings': dict(cycles=cycles, burn_in=burn_in, sem_cycles=sem_cycles,
                                 seed=seed, gain_scale=gain_scale, decay=decay,
                                 mh_steps=mh_steps, tolerance=tolerance)}


def compare_estimates(data, q_matrix, python_a, python_d, reference,
                      true_a=None, true_d=None, python_marginal=None, factor_labels=None):
    """Compare active slopes only; evaluate both fits with common integration nodes.

    factor_labels (e.g. ['G', 'S_1', ...]) renames slopes from a1, a2, ... to a(G), a(S_1), ...
    """
    q = np.asarray(q_matrix, dtype=bool)
    py_a, py_d = np.asarray(python_a), np.asarray(python_d)
    r_a, r_d = np.asarray(reference['a']), np.asarray(reference['d'])
    if py_a.shape != q.shape or r_a.shape != q.shape or py_d.shape != (len(q),) or r_d.shape != (len(q),):
        raise ValueError('Coefficients must have the same item and factor ordering.')
    if factor_labels is not None and len(factor_labels) != q.shape[1]:
        raise ValueError('Provide one factor label per dimension.')
    slope = (lambda k: f'a{k+1}') if factor_labels is None else (lambda k: f'a({factor_labels[k]})')
    rows = []
    for j in range(len(q)):
        for k in range(q.shape[1]):
            row = dict(Item=j+1, Parameter=slope(k), Active=bool(q[j, k]),
                       Python=float(py_a[j, k]), R_mirt=float(r_a[j, k]),
                       Python_minus_R=float(py_a[j, k]-r_a[j, k]))
            if true_a is not None: row['True'] = float(true_a[j, k])
            rows.append(row)
        row = dict(Item=j+1, Parameter='d', Active=True, Python=float(py_d[j]),
                   R_mirt=float(r_d[j]), Python_minus_R=float(py_d[j]-r_d[j]))
        if true_d is not None: row['True'] = float(true_d[j])
        rows.append(row)
    per_item = pd.DataFrame(rows)
    summary, recovery = [], []
    groups = [('Active slopes (all factors)', py_a[q], r_a[q], None if true_a is None else np.asarray(true_a)[q]),
              ('Intercepts', py_d, r_d, true_d)]
    groups.extend((f'Active {slope(k)}', py_a[q[:, k], k], r_a[q[:, k], k],
                   None if true_a is None else np.asarray(true_a)[q[:, k], k]) for k in range(q.shape[1]))
    for label, py, r, truth in groups:
        difference = py-r
        corr = float(np.corrcoef(py, r)[0, 1]) if len(py)>1 and np.std(py)>1e-12 and np.std(r)>1e-12 else np.nan
        summary.append(dict(Parameter=label, Count=len(py),
                            Mean_difference=float(np.mean(difference)),
                            MAE=float(np.mean(np.abs(difference))),
                            RMSE=float(np.sqrt(np.mean(difference**2))),
                            Max_absolute_difference=float(np.max(np.abs(difference))), Correlation=corr))
        if truth is not None:
            for estimator, fitted in [('Python', py), ('R mirt', r)]:
                error = fitted-np.asarray(truth)
                recovery.append(dict(Estimator=estimator, Parameter=label,
                                     Bias=float(np.mean(error)), MAE=float(np.mean(np.abs(error))),
                                     RMSE=float(np.sqrt(np.mean(error**2)))))
    py_ll, method = python_marginal if python_marginal is not None else marginal_loglikelihood(data, py_a, py_d)
    r_ll, method_r = marginal_loglikelihood(data, r_a, r_d)
    if method != method_r:
        raise ValueError('Both estimates must be evaluated using the same integration method.')
    return dict(per_item=per_item, summary=pd.DataFrame(summary), recovery=pd.DataFrame(recovery),
                likelihood=pd.DataFrame({'Estimator': ['Python', 'R mirt'],
                                         'Common_integrated_loglik': [py_ll, r_ll]}),
                loglik_difference=float(py_ll-r_ll), integration_method=method)
