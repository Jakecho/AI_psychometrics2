# Install / check the R reference packages for the Rasch Calibration Learning Lab.
# Usage:  Rscript install_packages.R            (installs into your default R library)
#         RASCH_R_LIBRARY=/path Rscript install_packages.R   (installs into a project library)
lib <- Sys.getenv("RASCH_R_LIBRARY")
if (nzchar(lib)) { dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths())) }
options(repos = "https://cloud.r-project.org")
needed <- c("TAM", "eRm", "jsonlite")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, lib = if (nzchar(lib)) lib else .libPaths()[1])
}
for (p in needed) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("%-9s %s\n", p, if (ok) as.character(packageVersion(p)) else "NOT AVAILABLE"))
}
cat(R.version.string, "\n")
