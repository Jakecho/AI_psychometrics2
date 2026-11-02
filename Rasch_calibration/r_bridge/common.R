# Shared helpers for the reference scripts. Arguments: <work folder>.
local_lib <- Sys.getenv("RASCH_R_LIBRARY")
if (nzchar(local_lib)) .libPaths(c(local_lib, .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
folder <- args[1]
settings <- fromJSON(file.path(folder, "settings.json"))
resp <- as.matrix(read.csv(file.path(folder, "responses.csv"), check.names = FALSE, na.strings = c("", "NA")))
write_out <- function(x) write_json(x, file.path(folder, "result.json"), auto_unbox = TRUE, digits = NA, na = "null")
num <- function(x) { x <- as.numeric(x); x[!is.finite(x)] <- NA; x }
# Optional anchors: settings$fixed_index (1-based) and settings$fixed_value -> TAM xsi.fixed matrix.
fixed_matrix <- function(settings) {
  if (is.null(settings$fixed_index) || length(settings$fixed_index) == 0) return(NULL)
  cbind(as.integer(settings$fixed_index), as.numeric(settings$fixed_value))
}
