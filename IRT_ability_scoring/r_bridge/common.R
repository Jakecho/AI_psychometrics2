# Shared helpers for the reference scripts. Argument: <work folder> holding
#   items.json (list of items: id, model, a, b, c, d, steps, D, deltas),
#   responses.csv (blank = missing), settings.json.
local_lib <- Sys.getenv("SCORING_R_LIBRARY")
if (nzchar(local_lib)) .libPaths(c(local_lib, .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
folder <- args[1]
settings <- fromJSON(file.path(folder, "settings.json"))
items <- fromJSON(file.path(folder, "items.json"), simplifyVector = FALSE)
resp <- as.matrix(read.csv(file.path(folder, "responses.csv"), check.names = FALSE, na.strings = c("", "NA")))
write_out <- function(x) write_json(x, file.path(folder, "result.json"), auto_unbox = TRUE, digits = NA, na = "null")
num <- function(x) { x <- as.numeric(x); x[is.nan(x)] <- NA; x }
# Infinite values cannot be written to JSON: send them as +/-1e308 and decode in Python.
enc <- function(x) { x <- num(x); x[is.infinite(x) & x > 0] <- 1e308; x[is.infinite(x) & x < 0] <- -1e308; x }
