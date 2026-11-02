# Reference JMLE: TAM::tam.jml. xsi is the item difficulty (P = exp(theta - xsi)).
# Optional anchors via xsi.fixed (settings$fixed_index / fixed_value).
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(TAM))
t0 <- Sys.time()
args_jml <- list(resp = resp, adj = settings$adj, bias = settings$bias, constraint = settings$constraint,
                 control = list(conv = settings$tol, maxiter = settings$max_iter), verbose = FALSE)
fx <- fixed_matrix(settings)
if (!is.null(fx)) args_jml$xsi.fixed <- fx
fit <- do.call(tam.jml, args_jml)
write_out(list(package = "TAM", version = as.character(packageVersion("TAM")), call = "tam.jml",
               b = num(fit$xsi), se_b = num(fit$errorP), theta = num(fit$theta), wle = num(fit$WLE),
               iterations = fit$iter, deviance = fit$deviance, seconds = as.numeric(Sys.time() - t0, units = "secs")))
