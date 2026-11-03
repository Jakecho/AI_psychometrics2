# Shared helpers for the reference scripts.
# Argument: <work folder> holding responses.csv (column "group" + item columns, blank = missing)
# and settings.json (model, design, prior, grid, anchors, ...). Each script writes result.json.
local_lib <- Sys.getenv("POLY_R_LIBRARY")
if (nzchar(local_lib)) .libPaths(c(local_lib, .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
folder <- args[1]
s <- fromJSON(file.path(folder, "settings.json"), simplifyVector = TRUE)
raw <- read.csv(file.path(folder, "responses.csv"), check.names = FALSE, na.strings = c("", "NA"),
                colClasses = c(group = "character"))
group <- raw$group
resp <- as.matrix(raw[, setdiff(names(raw), "group"), drop = FALSE])
ids <- colnames(resp)
J <- ncol(resp)
gnames <- s$group_names
G <- length(gnames)
ncat <- as.integer(s$ncat)
nodes <- seq(-s$bound, s$bound, length.out = s$n_nodes)
write_out <- function(x) write_json(x, file.path(folder, "result.json"), auto_unbox = TRUE, digits = NA,
                                    na = "null", null = "null")
fail <- function(msg) { write_out(list(error = msg)); quit(save = "no", status = 0) }
anchors <- if (is.null(s$anchors) || length(s$anchors) == 0) list() else s$anchors
anchor_ids <- if (length(anchors)) anchors$id else character(0)
item_out <- function(id, a, d) list(id = id, a = a, d = as.numeric(d))
# Anchor parameter vectors: jsonlite returns a matrix when all anchors have equal length, else a list.
anchor_vec <- function(field, k) {
  v <- anchors[[field]]
  if (is.matrix(v)) as.numeric(v[k, ]) else as.numeric(v[[k]])
}
