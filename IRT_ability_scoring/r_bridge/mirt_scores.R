# mirt::fscores() on a model whose parameters are all fixed (no estimation: TOL = NaN).
# Dichotomous -> "4PL" class (g, u fixed as given), GRM -> "graded", PCM/GPCM/RSM -> "gpcm".
# Slope-intercept form: a1 = D a; dichotomous d = -D a b; graded d_k = -D a b_k; gpcm d_k = -D a sum_{v<=k} delta_v.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(mirt))
J <- length(items)
ncat <- sapply(items, function(it) if (it$model %in% c("Rasch", "2PL", "3PL", "4PL")) 2L else 1L + length(it$deltas))
types <- sapply(items, function(it) if (it$model %in% c("Rasch", "2PL", "3PL", "4PL")) "4PL" else if (it$model == "GRM") "graded" else "gpcm")
dummy <- sapply(ncat, function(k) pmin(0:(max(ncat) - 1), k - 1))
dummy <- matrix(dummy, ncol = J); colnames(dummy) <- colnames(resp)
sv <- mirt(dummy, 1, itemtype = types, pars = "values")
sv$est <- FALSE
setv <- function(item, name, v) sv$value[sv$item == item & sv$name == name] <<- v
for (j in seq_len(J)) {
  it <- items[[j]]; id <- colnames(resp)[j]; Da <- it$D * it$a
  setv(id, "a1", Da)
  if (types[j] == "4PL") {
    setv(id, "d", -Da * it$b); setv(id, "g", it$c); setv(id, "u", it$d)
  } else if (types[j] == "graded") {
    th <- unlist(it$steps); for (k in seq_along(th)) setv(id, paste0("d", k), -Da * th[k])
  } else {
    cum <- cumsum(unlist(it$deltas)); for (k in seq_along(cum)) setv(id, paste0("d", k), -Da * cum[k])
  }
}
# A bounded Gaussian density on the rectangular grid reproduces the prior N(mean, sd^2).
mod <- mirt(dummy, 1, itemtype = types, pars = sv, TOL = NaN, verbose = FALSE)
out <- list(version = as.character(packageVersion("mirt")), results = list())
for (m in settings$methods) {
  r <- tryCatch({
    e <- fscores(mod, method = m, response.pattern = resp, quadpts = settings$n_nodes,
                 theta_lim = c(-settings$theta_lim, settings$theta_lim),
                 mean = settings$prior_mean, cov = matrix(settings$prior_sd^2), verbose = FALSE)
    list(theta = enc(e[, "F1"]), se = enc(e[, "SE_F1"]))
  }, error = function(e) list(error = conditionMessage(e)))
  out$results[[m]] <- r
}
write_out(out)
