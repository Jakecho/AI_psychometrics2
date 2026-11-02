# Optional development reference for test_mhrm_engine.py. Not used by the app.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 5L)
local_library <- Sys.getenv('MIRT_R_LIBRARY')
if (nzchar(local_library)) .libPaths(c(local_library, .libPaths()))
suppressPackageStartupMessages(library(mirt))
folder <- args[1L]
cycles <- as.integer(args[2L])
burn_in <- as.integer(args[3L])
sem_cycles <- as.integer(args[4L])
seed <- as.integer(args[5L])
responses <- as.matrix(read.csv(file.path(folder, 'responses.csv')))
q <- as.matrix(read.csv(file.path(folder, 'q.csv')))
colnames(responses) <- paste0('Item_', seq_len(ncol(responses)))
model_lines <- vapply(seq_len(ncol(q)), function(k) {
    paste0('F', k, ' = ', paste(which(q[, k] == 1), collapse = ','))
}, character(1))
model <- mirt.model(paste(model_lines, collapse = '\n'))
# No COV declarations: latent means are fixed to 0 and covariance to I.
pars <- mirt(responses, model, itemtype = '2PL', pars = 'values', verbose = FALSE)
slopes <- grepl('^a[0-9]+$', pars$name) & pars$est
pars$lbound[slopes] <- 0
pars$ubound[slopes] <- 10
set.seed(seed)
fit <- mirt(responses, model, itemtype = '2PL', method = 'MHRM', pars = pars,
            verbose = FALSE, SE = FALSE,
            technical = list(NCYCLES = cycles, BURNIN = burn_in,
                             SEMCYCLES = sem_cycles, parallel = FALSE,
                             set.seed = seed))
coefs <- coef(fit, simplify = TRUE, IRTpars = FALSE)
write.csv(coefs$items, file.path(folder, 'estimates.csv'), row.names = FALSE)
metadata <- data.frame(
    converged = extract.mirt(fit, 'converged'),
    stage3_iterations = extract.mirt(fit, 'iterations'),
    loglik = as.numeric(logLik(fit)),
    mirt_version = as.character(packageVersion('mirt')),
    r_version = as.character(getRversion())
)
write.csv(metadata, file.path(folder, 'metadata.csv'), row.names = FALSE)
