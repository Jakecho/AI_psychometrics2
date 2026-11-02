# Run only an installed irtQ package. Never substitute sourced or emulated code.
suppressPackageStartupMessages(library(irtQ))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
z <- fromJSON(args[1], simplifyVector=FALSE)
cfg <- z$config
pr <- cfg$priors
base <- dirname(args[1])
read_matrix <- function(name) as.matrix(read.csv(file.path(base,name),check.names=FALSE))
meta <- read.csv(file.path(base,"items.csv"),check.names=FALSE)
# Accept 1/0 or TRUE/True: a text flag would otherwise unfix every bank anchor.
meta$fixed <- as.character(meta$fixed) %in% c("1","TRUE","True","true")
y <- read_matrix("responses.csv")
admin <- read_matrix("admin.csv") > 0
group <- as.integer(unlist(z$group))
names_g <- unlist(z$group_names)
starts <- read_matrix("starts.csv")
x <- shape_df(par.drm=list(a=starts[,1],b=starts[,2],g=starts[,3]),
              item.id=meta$id,cats=rep(2,nrow(meta)),model=rep(paste0(cfg$model,"M"),nrow(meta)))
weights <- read.csv(file.path(base,"initial_weights.csv"))
common <- list(D=cfg$D,use.aprior=pr$a,use.bprior=pr$b,use.gprior=pr$c,
               aprior=list(dist="lnorm",params=c(pr$a_meanlog,pr$a_sdlog)),
               bprior=list(dist="norm",params=c(pr$b_mean,pr$b_sd)),
               gprior=list(dist="beta",params=c(pr$c_alpha,pr$c_beta)),
               Quadrature=c(cfg$nodes,cfg$extent),group.mean=cfg$mean,group.var=cfg$sd^2,
               weights=if(is.null(cfg$initial_weights))NULL else weights,EmpHist=(cfg$density=="Histogram"),use.startval=TRUE,
               Etol=cfg$tolerance,MaxE=cfg$max_cycles,
               control=list(eval.max=1000,iter.max=500,x.tol=1e-8,rel.tol=1e-10),
               fipc=isTRUE(z$anchored),fipc.method=cfg$schedule,se=FALSE,verbose=FALSE)
if (isTRUE(z$anchored)) common$fix.id <- meta$id[meta$fixed]
warnings <- character()
if (length(names_g)==1) {
  common$x <- x
  common$data <- y
  f <- withCallingHandlers(do.call(est_irt,common),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
  par <- f$par.est
  ws <- list(f$weights)
} else {
  present <- lapply(seq_along(names_g),function(g) which(colSums(admin[group==(g-1),,drop=FALSE])>0))
  common$x <- lapply(present,function(ix) x[ix,,drop=FALSE])
  common$data <- lapply(seq_along(names_g),function(g) y[group==(g-1),present[[g]],drop=FALSE])
  common$group.name <- names_g
  common$free.group <- if (isTRUE(z$anchored)) names_g else names_g[-1]
  f <- withCallingHandlers(do.call(est_mg,common),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
  par <- f$par.est$overall
  ws <- f$weights
}
par <- par[match(meta$id,par$id),,drop=FALSE]
out <- data.frame(id=par$id,a=par$par.1,b=par$par.2,c=ifelse(is.na(par$par.3),0,par$par.3))
write.csv(out,file.path(base,"reference_items.csv"),row.names=FALSE)
for (g in seq_along(ws)) write.csv(ws[[g]],file.path(base,paste0("reference_weights_",g,".csv")),row.names=FALSE)
write_json(list(version=as.character(packageVersion("irtQ")),R=R.version.string,
                iterations=f$niter,maxpar.diff=f$maxpar.diff,convergence=f$test.1,
                warnings=warnings,optimizer="R nlminb; direct coefficients; tighter optimizer controls"),
           file.path(base,"reference_report.json"),auto_unbox=TRUE,pretty=TRUE)
