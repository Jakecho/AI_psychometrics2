local_lib <- Sys.getenv("RASCH_R_LIBRARY")
if (nzchar(local_lib)) .libPaths(c(local_lib, .libPaths()))
v <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else NA
cat(jsonlite::toJSON(list(R = R.version.string, TAM = v("TAM"), eRm = v("eRm")), auto_unbox = TRUE, na = "null"))
