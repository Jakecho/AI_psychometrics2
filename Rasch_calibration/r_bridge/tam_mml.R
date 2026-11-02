# Reference MML-EM: TAM::tam.mml, Rasch, normal ability with mean 0 (constraint = "cases").
# Optional: settings$groups (one normal distribution per group; first group's mean 0 unless anchored)
# and anchors via xsi.fixed (settings$fixed_index / fixed_value; then every group mean is free).
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(TAM))
t0 <- Sys.time()
args_mml <- list(resp = resp, constraint = "cases", verbose = FALSE,
                 control = list(nodes = seq(-settings$extent, settings$extent, length.out = settings$n_nodes),
                                conv = settings$tol, maxiter = settings$max_iter))
if (!isTRUE(settings$estimate_variance)) {
  args_mml$variance.fixed <- matrix(c(1, 1, settings$fixed_sd^2), nrow = 1)
}
fx <- fixed_matrix(settings)
if (!is.null(fx)) args_mml$xsi.fixed <- fx
groups <- settings$groups
if (!is.null(groups)) args_mml$group <- as.integer(groups)
fit <- do.call(tam.mml, args_mml)
wle <- tam.wle(fit, progress = FALSE)
first <- if (is.null(groups)) 1 else match(sort(unique(groups)), groups)
write_out(list(package = "TAM", version = as.character(packageVersion("TAM")), call = "tam.mml",
               b = num(fit$xsi$xsi), se_b = num(fit$xsi$se.xsi), variance = as.numeric(fit$variance)[1],
               group_means = as.numeric(fit$beta[, 1]), group_variances = as.numeric(rep_len(fit$variance, nrow(resp))[first]),
               loglik = as.numeric(fit$ic$loglike), iterations = fit$iter,
               wle = num(wle$theta), wle_se = num(wle$error), seconds = as.numeric(Sys.time() - t0, units = "secs")))
