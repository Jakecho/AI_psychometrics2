# R mirt reference for the dashboard. All inputs are local temporary files.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 9L)
local_library <- Sys.getenv('MIRT_R_LIBRARY')
if (nzchar(local_library)) .libPaths(c(local_library, .libPaths()))
suppressPackageStartupMessages(library(mirt))
folder <- args[1L]
cycles <- as.integer(args[2L]); burn_in <- as.integer(args[3L])
sem_cycles <- as.integer(args[4L]); seed <- as.integer(args[5L])
gain_scale <- as.numeric(args[6L]); decay <- as.numeric(args[7L])
mh_steps <- as.integer(args[8L]); tolerance <- as.numeric(args[9L])
responses <- as.matrix(read.csv(file.path(folder, 'responses.csv')))
q <- as.matrix(read.csv(file.path(folder, 'q.csv')))
colnames(responses) <- paste0('Item_', seq_len(ncol(responses)))
model_lines <- vapply(seq_len(ncol(q)), function(k) {
    paste0('F', k, ' = ', paste(which(q[, k] == 1), collapse=','))
}, character(1))
writeLines(model_lines, file.path(folder, 'model.txt'))
model <- mirt.model(paste(model_lines, collapse='\n'))
# No free latent mean/variance/covariance declarations: N(0,I).
pars <- mirt(responses, model, itemtype='2PL', pars='values', verbose=FALSE)
slopes <- grepl('^a[0-9]+$', pars$name) & pars$est
pars$lbound[slopes] <- 0; pars$ubound[slopes] <- 10
set.seed(seed)
fit <- mirt(responses, model, itemtype='2PL', method='MHRM', optimizer='NR1',
            pars=pars, verbose=FALSE, SE=FALSE, TOL=tolerance,
            technical=list(NCYCLES=cycles, BURNIN=burn_in, SEMCYCLES=sem_cycles,
                           MHDRAWS=mh_steps, gain=c(gain_scale, decay),
                           parallel=FALSE, set.seed=seed))
coefs <- coef(fit, simplify=TRUE, IRTpars=FALSE)
write.csv(coefs$items, file.path(folder, 'estimates.csv'), row.names=FALSE)
write.csv(data.frame(mean=as.numeric(coefs$means)), file.path(folder, 'means.csv'), row.names=FALSE)
write.csv(coefs$cov, file.path(folder, 'covariance.csv'), row.names=FALSE)
write.csv(data.frame(converged=extract.mirt(fit, 'converged'),
                     stage3_iterations=extract.mirt(fit, 'iterations'),
                     native_loglik=as.numeric(logLik(fit)),
                     mirt_version=as.character(packageVersion('mirt')),
                     r_version=as.character(getRversion())),
          file.path(folder, 'metadata.csv'), row.names=FALSE)
