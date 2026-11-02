# Installs the reference packages. Use SCORING_R_LIBRARY to install into a project library.
lib <- Sys.getenv("SCORING_R_LIBRARY")
if (nzchar(lib)) { dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths())) }
pkgs <- c("jsonlite", "irtQ", "mirt", "TAM")
need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) install.packages(need, lib = if (nzchar(lib)) lib else .libPaths()[1], repos = "https://cloud.r-project.org")
for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")
