# Reference packages for page 12 and tests/test_reference.py.
# Installs into $POLY_R_LIBRARY when set, else the default user library.
lib <- Sys.getenv("POLY_R_LIBRARY")
if (nzchar(lib)) { dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths())) }
pkgs <- c("jsonlite", "TAM", "irtQ", "mirt")
need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) install.packages(need, lib = if (nzchar(lib)) lib else .libPaths()[1], repos = "https://cloud.r-project.org")
for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")
