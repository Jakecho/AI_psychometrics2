# Reference WLE with supplied item difficulties: TAM::tam.mml with all xsi fixed, then tam.wle.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(TAM))
b <- as.numeric(settings$b)
fit <- tam.mml(resp, xsi.fixed = cbind(seq_along(b), b), verbose = FALSE)
wle <- tam.wle(fit, progress = FALSE)
mle <- tam.wle(fit, WLE = FALSE, progress = FALSE)
write_out(list(package = "TAM", version = as.character(packageVersion("TAM")), call = "tam.wle",
               wle = num(wle$theta), wle_se = num(wle$error), mle = num(mle$theta), mle_se = num(mle$error),
               person_scores = num(wle$PersonScores), person_max = num(wle$PersonMax)))
