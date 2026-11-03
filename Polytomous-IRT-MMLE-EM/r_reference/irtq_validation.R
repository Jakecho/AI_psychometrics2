# irtQ reference calibration: GPCM, PCM (= GPCM with fix.a.gpcm = TRUE) and GRM.
# single → est_irt();  fipc → est_irt(fipc = TRUE, fipc.method, fix.id)
# concurrent → est_mg(free.group = 2..G);  mgfipc → est_mg(fipc = TRUE, fix.id, free.group = 1..G)
# Priors: fixed normal (group.mean/group.var), fixed empirical (weights = data.frame(theta, weight)),
# estimated empirical (EmpHist = TRUE, Woods 2007). irtQ estimates normal moments only for free groups.
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(irtQ))
if (s$model == "RSM") fail("irtQ has no rating scale model; use TAM for RSM.")
mod_name <- if (s$model == "GRM") "GRM" else "GPCM"
fixa <- s$model == "PCM"
emp <- s$prior == "estimated_empirical"
W <- NULL
if (s$prior == "fixed_empirical") W <- data.frame(theta = nodes, weight = as.numeric(s$fixed_weights))
quad <- c(s$n_nodes, s$bound)

meta_for <- function(cols) {
  # item metadata: anchors carry bank values, free items neutral starting values
  a <- numeric(0); d <- list()
  for (j in cols) {
    id <- ids[j]; K <- ncat[j] - 1
    k <- match(id, anchor_ids)
    if (!is.na(k)) { a <- c(a, anchors$a[k]); d[[length(d) + 1]] <- anchor_vec("d", k) }
    else { a <- c(a, 1); d[[length(d) + 1]] <- seq(-1, 1, length.out = K) }
  }
  shape_df(par.prm = list(a = a, d = d), item.id = ids[cols], cats = ncat[cols], model = mod_name)
}

t0 <- Sys.time()
res <- tryCatch({
  if (s$design %in% c("single", "fipc")) {
    if (s$design == "fipc") {
      mod <- est_irt(x = meta_for(seq_len(J)), data = resp, D = s$D, fix.a.gpcm = fixa, Quadrature = quad,
                     weights = W, group.mean = s$mean, group.var = s$sd^2, EmpHist = emp, Etol = s$Etol,
                     MaxE = s$MaxE, fipc = TRUE, fipc.method = s$fipc_method, fix.id = anchor_ids,
                     se = FALSE, verbose = FALSE)
      call <- sprintf("est_irt(x, data, D = %g, fipc = TRUE, fipc.method = '%s', fix.id = <%d anchors>, EmpHist = %s, Quadrature = c(%d, %g))",
                      s$D, s$fipc_method, length(anchor_ids), emp, s$n_nodes, s$bound)
    } else {
      mod <- est_irt(data = resp, D = s$D, model = rep(mod_name, J), cats = ncat, item.id = ids,
                     fix.a.gpcm = fixa, Quadrature = quad, weights = W, group.mean = s$mean, group.var = s$sd^2,
                     EmpHist = emp, Etol = s$Etol, MaxE = s$MaxE, se = FALSE, verbose = FALSE)
      call <- sprintf("est_irt(data, D = %g, model = '%s', fix.a.gpcm = %s, EmpHist = %s, Quadrature = c(%d, %g))",
                      s$D, mod_name, fixa, emp, s$n_nodes, s$bound)
    }
    par <- getirt(mod, what = "par.est")
    gp <- getirt(mod, what = "group.par")
    w <- getirt(mod, what = "weights")
    groups <- list(list(name = gnames[1], mean = gp["estimates", "mu"], sd = gp["estimates", "sigma"]))
    weights <- list(w[, 2])
  } else {
    data_l <- list(); x_l <- list(); model_l <- list(); cats_l <- list(); id_l <- list()
    for (g in seq_len(G)) {
      rows <- group == gnames[g]
      cols <- which(colSums(!is.na(resp[rows, , drop = FALSE])) > 0)
      data_l[[g]] <- resp[rows, cols, drop = FALSE]
      x_l[[g]] <- meta_for(cols)
      model_l[[g]] <- rep(mod_name, length(cols)); cats_l[[g]] <- ncat[cols]; id_l[[g]] <- ids[cols]
    }
    if (s$design == "mgfipc") {
      mod <- est_mg(x = x_l, data = data_l, group.name = gnames, D = s$D, fix.a.gpcm = fixa,
                    free.group = seq_len(G), Quadrature = quad, EmpHist = emp, Etol = s$Etol, MaxE = s$MaxE,
                    fipc = TRUE, fipc.method = s$fipc_method, fix.id = anchor_ids, se = FALSE, verbose = FALSE)
      call <- sprintf("est_mg(x, data, group.name, fipc = TRUE, fipc.method = '%s', fix.id = <%d anchors>, free.group = 1:%d, EmpHist = %s)",
                      s$fipc_method, length(anchor_ids), G, emp)
    } else {
      mod <- est_mg(data = data_l, group.name = gnames, D = s$D, model = model_l, cats = cats_l, item.id = id_l,
                    fix.a.gpcm = fixa, free.group = if (G > 1) 2:G else NULL, Quadrature = quad,
                    weights = W, group.mean = s$mean, group.var = s$sd^2, EmpHist = emp, Etol = s$Etol,
                    MaxE = s$MaxE, se = FALSE, verbose = FALSE)
      call <- sprintf("est_mg(data, group.name, model, cats, item.id, free.group = 2:%d, EmpHist = %s, Quadrature = c(%d, %g))",
                      G, emp, s$n_nodes, s$bound)
    }
    par <- getirt(mod, what = "par.est")$overall
    gp <- getirt(mod, what = "group.par")
    w <- getirt(mod, what = "weights")
    groups <- lapply(seq_len(G), function(g) list(name = gnames[g], mean = gp[[g]]["estimates", "mu"],
                                                 sd = gp[[g]]["estimates", "sigma"]))
    weights <- lapply(seq_len(G), function(g) w[[g]][, 2])
  }
  items <- lapply(seq_len(nrow(par)), function(r) {
    K <- par$cats[r] - 1
    item_out(as.character(par$id[r]), par$par.1[r], unlist(par[r, paste0("par.", 2:(K + 1))]))
  })
  ll <- mod$loglikelihood
  if (is.list(ll)) ll <- ll$overall
  list(items = items, groups = groups, weights = weights, loglik = as.numeric(ll)[1], niter = mod$niter,
       call = call)
}, error = function(e) list(error = conditionMessage(e)))
res$version <- as.character(packageVersion("irtQ"))
res$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write_out(res)
