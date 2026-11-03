# mirt supplementary reference: single-group and multiple-group calibration for PCM ('Rasch'),
# GPCM ('gpcm') and GRM ('graded') on an evenly spaced grid (technical theta_lim, quadpts).
# Coefficients are converted back to the lab's canonical parameters:
#   gpcm/Rasch: P_k ∝ exp(k a θ + d_k) → δ_k = −(d_k − d_{k−1}) / a ;  graded: b_k = −d_k / a.
# Estimated empirical priors use mirt's dentype = "empiricalhist_Woods".
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(mirt))
if (s$model == "RSM") fail("Use TAM for the RSM reference.")
if (s$design %in% c("fipc", "mgfipc")) fail("Fixed-item calibration is compared with irtQ (and TAM), not mirt.")
if (s$prior == "fixed_empirical") fail("mirt has no simple fixed-histogram prior interface; not compared.")
itype <- switch(s$model, PCM = "Rasch", GPCM = "gpcm", GRM = "graded")
tech <- list(theta_lim = c(-s$bound, s$bound), NCYCLES = s$MaxE)
dentype <- if (s$prior == "estimated_empirical") "empiricalhist_Woods" else "Gaussian"
df <- as.data.frame(resp)
t0 <- Sys.time()
res <- tryCatch({
  if (G == 1) {
    mod <- mirt(df, 1, itemtype = itype, quadpts = s$n_nodes, TOL = s$Etol, technical = tech, verbose = FALSE,
                dentype = dentype)
    cf <- coef(mod, IRTpars = FALSE, simplify = TRUE)
    gp <- list(cf$means, cf$cov)
    groups <- list(list(name = gnames[1], mean = as.numeric(cf$means), sd = sqrt(as.numeric(cf$cov))))
    call <- sprintf("mirt(data, 1, itemtype = '%s', quadpts = %d, technical = list(theta_lim = c(-%g, %g)), dentype = '%s')",
                    itype, s$n_nodes, s$bound, s$bound, dentype)
  } else {
    mod <- multipleGroup(df, 1, group = factor(group, levels = gnames), itemtype = itype, quadpts = s$n_nodes,
                         TOL = s$Etol, technical = tech, verbose = FALSE, dentype = dentype,
                         invariance = c(ids, "free_means", "free_var"))
    cfa <- coef(mod, IRTpars = FALSE, simplify = TRUE)
    cf <- cfa[[1]]
    groups <- lapply(seq_len(G), function(g) list(name = gnames[g], mean = as.numeric(cfa[[g]]$means),
                                                 sd = sqrt(as.numeric(cfa[[g]]$cov))))
    call <- sprintf("multipleGroup(data, 1, group, itemtype = '%s', invariance = c(items, 'free_means', 'free_var'))", itype)
  }
  it <- cf$items
  items <- lapply(seq_len(J), function(j) {
    id <- ids[j]; K <- ncat[j] - 1; row <- it[id, ]
    a <- row[["a1"]]
    if (itype == "graded") {
      d <- -row[paste0("d", seq_len(K))] / a
    } else {
      dk <- row[paste0("d", 0:K)]
      d <- -diff(dk) / a
    }
    item_out(id, a, d)
  })
  list(items = items, groups = groups, loglik = extract.mirt(mod, "logLik"), niter = extract.mirt(mod, "iterations"),
       call = call)
}, error = function(e) list(error = conditionMessage(e)))
res$version <- as.character(packageVersion("mirt"))
res$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write_out(res)
