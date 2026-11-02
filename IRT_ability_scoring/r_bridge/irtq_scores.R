# irtQ::est_score() with fixed item parameters (primary reference).
# Models: Rasch -> 1PLM (a = 1), 2PL -> 2PLM, 3PL -> 3PLM, PCM/GPCM/RSM -> GPCM (steps = deltas), GRM -> GRM.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(irtQ))
model_map <- c(Rasch = "1PLM", "2PL" = "2PLM", "3PL" = "3PLM", PCM = "GPCM", GPCM = "GPCM", RSM = "GPCM", GRM = "GRM")
maxpar <- max(sapply(items, function(it) if (it$model %in% c("Rasch", "2PL", "3PL")) 3 else 1 + length(it$deltas)))
x <- data.frame(id = sapply(items, `[[`, "id"), cats = NA_integer_, model = NA_character_, stringsAsFactors = FALSE)
P <- matrix(NA_real_, length(items), maxpar)
for (j in seq_along(items)) {
  it <- items[[j]]
  x$model[j] <- model_map[[it$model]]
  if (it$model %in% c("Rasch", "2PL", "3PL")) {
    x$cats[j] <- 2L
    P[j, 1:3] <- c(it$a, it$b, if (it$model == "3PL") it$c else 0)
  } else {
    d <- unlist(it$deltas)
    x$cats[j] <- length(d) + 1L
    P[j, 1:(1 + length(d))] <- c(it$a, d)
  }
}
for (k in seq_len(maxpar)) x[[paste0("par.", k)]] <- P[, k]
D <- settings$D
rng <- c(-settings$bound, settings$bound)
prior <- c(settings$prior_mean, settings$prior_sd)
weights <- NULL
if (settings$quad_rule == "rectangular") weights <- data.frame(theta = unlist(settings$nodes), weight = unlist(settings$weights))
out <- list(version = as.character(packageVersion("irtQ")), results = list(), notes = list())
for (m in settings$methods) {
  r <- tryCatch({
    if (m == "EAP.SUM" || m == "INV.TCC") {
      if (anyNA(resp)) stop("irtQ replaces missing responses with 0 for ", m, "; not compared on data with missing responses")
      e <- est_score(x, resp, D = D, method = m, norm.prior = prior, nquad = settings$n_nodes, weights = weights,
                     tol = settings$tol, range.tcc = c(-7, 7))
      list(theta = enc(e$est.par$est.theta), se = enc(e$est.par$se.theta))
    } else {
      e <- est_score(x, resp, D = D, method = m, range = rng, norm.prior = prior, nquad = settings$n_nodes,
                     weights = weights, tol = settings$tol, max.iter = settings$max_iter)
      list(theta = enc(e$est.theta), se = enc(e$se.theta))
    }
  }, error = function(e) list(error = conditionMessage(e)))
  out$results[[m]] <- r
}
write_out(out)
