# TAM reference calibration for the Rasch family: PCM and RSM (irtmodel = "PCM" / "RSM").
# Identification follows TAM: reference mean fixed at 0, variance estimated (Rasch slopes fix the unit).
# single → tam.mml();  concurrent → tam.mml(group = , Y = focal dummies)  (group means and variances)
# fipc / mgfipc → xsi.fixed for the anchors' parameters and beta.fixed = FALSE (means estimated).
# Priors: estimated normal (TAM default) or fixed normal (variance.fixed). TAM's empirical-prior interfaces
# differ from irtQ's, so empirical priors are reported as outside this script's scope.
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(TAM))
if (s$model == "GPCM") {
  # Supplementary GPCM check: tam.mml.2pl(irtmodel = "GPCM") fixes N(0, 1); item_irt gives a = alpha, δ_k = beta + tau_k
  if (s$design != "single" || s$prior %in% c("fixed_empirical", "estimated_empirical"))
    fail("TAM's GPCM check here covers the single-group normal design only.")
  t0 <- Sys.time()
  res <- tryCatch({
    mod <- tam.mml.2pl(resp, irtmodel = "GPCM", verbose = FALSE,
                       control = list(nodes = nodes, progress = FALSE, conv = s$Etol, convD = s$Etol * 1e-2,
                                      maxiter = s$MaxE))
    ii <- mod$item_irt
    items <- lapply(seq_len(J), function(j) {
      K <- ncat[j] - 1
      item_out(ids[j], ii$alpha[j], ii$beta[j] + unlist(ii[j, paste0("tau.Cat", seq_len(K))]))
    })
    list(items = items, groups = list(list(name = gnames[1], mean = as.numeric(mod$beta)[1],
                                           sd = sqrt(as.numeric(mod$variance)[1]))),
         loglik = -mod$deviance / 2, niter = mod$iter,
         call = sprintf("tam.mml.2pl(resp, irtmodel = 'GPCM', control = list(nodes = seq(-%g, %g, len = %d)))",
                        s$bound, s$bound, s$n_nodes))
  }, error = function(e) list(error = conditionMessage(e)))
  res$version <- as.character(packageVersion("TAM"))
  res$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  write_out(res)
  quit(save = "no", status = 0)
}
if (!(s$model %in% c("PCM", "RSM"))) fail("TAM's tam.mml reference here covers PCM, RSM and single-group GPCM.")
if (s$prior %in% c("fixed_empirical", "estimated_empirical"))
  fail("Empirical ability priors are not compared with TAM in this lab (TAM's interface differs from irtQ's).")
ctrl <- list(nodes = nodes, progress = FALSE, conv = s$Etol, convD = s$Etol * 1e-2, maxiter = s$MaxE)
gidx <- match(group, gnames)
args <- list(resp = resp, irtmodel = s$model, control = ctrl, verbose = FALSE)
if (G > 1) {
  args$group <- gidx
  Y <- sapply(2:G, function(g) as.numeric(gidx == g))
  args$Y <- matrix(Y, ncol = G - 1)
}
anchored <- s$design %in% c("fipc", "mgfipc")
fixed_note <- ""
t0 <- Sys.time()
res <- tryCatch({
  if (anchored || s$prior == "fixed_normal") {
    pargs <- args; pargs$control <- list(nodes = nodes, progress = FALSE, maxiter = 1)
    probe <- do.call(tam.mml, pargs)
    xn <- rownames(probe$xsi)
  }
  if (anchored) {
    fx <- NULL
    for (k in seq_along(anchor_ids)) {
      id <- anchor_ids[k]; d <- anchor_vec("d", k)
      if (s$model == "PCM") {
        rows <- match(paste0(id, "_Cat", seq_along(d)), xn); fx <- rbind(fx, cbind(rows, d))
      } else {
        fx <- rbind(fx, cbind(match(id, xn), anchors$b[k]))
        tau <- anchor_vec("tau", k)
        fx <- rbind(fx, cbind(match(paste0("Cat", seq_len(length(tau) - 1)), xn), tau[-length(tau)]))
      }
    }
    fx <- unique(fx)
    if (anyNA(fx)) stop("could not map anchor parameters to TAM xsi rows")
    args$xsi.fixed <- fx
    args$beta.fixed <- FALSE
    if (G == 1) args$Y <- NULL
  }
  if (s$prior == "fixed_normal") {
    if (G > 1) stop("fixed-normal multigroup comparison is not set up for TAM")
    args$variance.fixed <- matrix(c(1, 1, s$sd^2), 1, 3)
    if (anchored) { args$beta.fixed <- matrix(c(1, 1, s$mean), 1, 3) }
  }
  mod <- do.call(tam.mml, args)
  xsi <- mod$xsi$xsi; names(xsi) <- rownames(mod$xsi)
  items <- lapply(seq_len(J), function(j) {
    id <- ids[j]; K <- ncat[j] - 1
    if (s$model == "PCM") {
      d <- xsi[paste0(id, "_Cat", seq_len(K))]
    } else {
      tau <- xsi[paste0("Cat", seq_len(K - 1))]; tau <- c(tau, -sum(tau))
      d <- xsi[id] + tau
    }
    item_out(id, 1, d)
  })
  beta <- as.numeric(mod$beta)
  var <- as.numeric(mod$variance)
  means <- if (G > 1) c(beta[1], beta[1] + beta[-1]) else beta[1]
  # TAM returns the variance per examinee when group-specific variances are estimated
  vars <- if (length(var) == nrow(resp)) sapply(seq_len(G), function(g) var[which(gidx == g)[1]])
          else if (length(var) >= G) var[seq_len(G)] else rep(var[1], G)
  groups <- lapply(seq_len(G), function(g) list(name = gnames[g], mean = means[g], sd = sqrt(vars[g])))
  list(items = items, groups = groups, loglik = -mod$deviance / 2, niter = mod$iter,
       call = sprintf("tam.mml(resp, irtmodel = '%s'%s%s, control = list(nodes = seq(-%g, %g, len = %d)))",
                      s$model, if (G > 1) ", group, Y = focal dummies" else "",
                      if (anchored) ", xsi.fixed = <anchors>, beta.fixed = FALSE" else "",
                      s$bound, s$bound, s$n_nodes))
}, error = function(e) list(error = conditionMessage(e)))
res$version <- as.character(packageVersion("TAM"))
res$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write_out(res)
