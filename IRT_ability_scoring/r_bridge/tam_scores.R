# TAM::tam.wle() for Rasch-family items (Rasch, PCM, RSM with a = 1, D = 1): all xsi fixed.
# TAM's PCM xsi are the step difficulties delta_v (P_k proportional to exp(k theta - sum_{v<=k} delta_v)).
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(TAM))
xsi <- unlist(lapply(items, function(it) unlist(it$deltas)))
# Every category must occur for TAM to build its design; pad with one row per category (removed afterwards).
ncat <- sapply(items, function(it) 1L + length(it$deltas))
pad <- sapply(ncat, function(k) pmin(0:(max(ncat) - 1), k - 1))
pad <- matrix(pad, ncol = length(items)); colnames(pad) <- colnames(resp)
dat <- rbind(resp, pad)
fit <- tam.mml(dat, irtmodel = "PCM", xsi.fixed = cbind(seq_along(xsi), xsi), verbose = FALSE,
               control = list(maxiter = 5))
keep <- seq_len(nrow(resp))
w <- tam.wle(fit, WLE = TRUE, progress = FALSE)
m <- tam.wle(fit, WLE = FALSE, progress = FALSE)
write_out(list(version = as.character(packageVersion("TAM")),
               results = list(WLE = list(theta = enc(w$theta[keep]), se = enc(w$error[keep])),
                              MLE = list(theta = enc(m$theta[keep]), se = enc(m$error[keep])))))
