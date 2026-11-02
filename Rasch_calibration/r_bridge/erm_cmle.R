# Reference CMLE: eRm::RM (sum0 = TRUE). eRm reports easiness beta; difficulty b = -beta.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(eRm))
t0 <- Sys.time()
fit <- RM(resp, sum0 = TRUE, se = TRUE)
write_out(list(package = "eRm", version = as.character(packageVersion("eRm")), call = "RM",
               b = num(-fit$betapar), se_b = num(fit$se.beta), cond_loglik = as.numeric(fit$loglik),
               iterations = fit$iter, convergence = fit$convergence,
               seconds = as.numeric(Sys.time() - t0, units = "secs")))
