% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/foo.R
\name{linear.model.MLE}
\alias{linear.model.MLE}
\title{Maximum Likelihood estimation for the geostatistical linear Gaussian model}
\usage{
linear.model.MLE(
  formula,
  coords = NULL,
  data,
  ID.coords = NULL,
  kappa,
  fixed.rel.nugget = NULL,
  start.cov.pars,
  method = "BFGS",
  low.rank = FALSE,
  knots = NULL,
  messages = TRUE,
  profile.llik = FALSE,
  SPDE = FALSE,
  mesh = NULL,
  SPDE.analytic.hessian = FALSE
)
}
\arguments{
\item{formula}{an object of class "\code{\link{formula}}" (or one that can be coerced to that class): a symbolic description of the model to be fitted.}

\item{coords}{an object of class \code{\link{formula}} indicating the geographic coordinates.}

\item{data}{a data frame containing the variables in the model.}

\item{ID.coords}{vector of ID values for the unique set of spatial coordinates obtained from \code{\link{create.ID.coords}}. These must be provided in order to define a geostatistical model where locations have multiple observations. Default is \code{ID.coords=NULL}. See the \bold{Details} section for more information.}

\item{kappa}{shape parameter of the Matern covariance function.}

\item{fixed.rel.nugget}{fixed value for the relative variance of the nugget effect; default is \code{fixed.rel.nugget=NULL} if this should be included in the estimation.}

\item{start.cov.pars}{if \code{ID.coords=NULL}, a vector of length two with elements corresponding to the starting values of \code{phi} and the relative variance of the nugget effect \code{nu2}, respectively, that are used in the optimization algorithm; if \code{ID.coords} is provided, a third starting value for the relative variance of the individual unexplained variation \code{nu2.star = omega2/sigma2} must be provided. If \code{nu2} is fixed through \code{fixed.rel.nugget}, then start.cov.pars represents the starting value for \code{phi} only, if \code{ID.coords=NULL}, or for \code{phi} and \code{nu2.star}, otherwise.}

\item{method}{method of optimization. If \code{method="BFGS"} then the \code{\link{maxBFGS}} function is used; otherwise \code{method="nlminb"} to use the \code{\link{nlminb}} function. Default is \code{method="BFGS"}.}

\item{low.rank}{logical; if \code{low.rank=TRUE} a low-rank approximation of the Gaussian spatial process is used when fitting the model. Default is \code{low.rank=FALSE}.}

\item{knots}{if \code{low.rank=TRUE}, \code{knots} is a matrix of spatial knots that are used in the low-rank approximation. Default is \code{knots=NULL}.}

\item{messages}{logical; if \code{messages=TRUE} then status messages are printed on the screen (or output device) while the function is running. Default is \code{messages=TRUE}.}

\item{profile.llik}{logical; if \code{profile.llik=TRUE} the maximization of the profile likelihood is carried out. If \code{profile.llik=FALSE} the full-likelihood is used. Default is \code{profile.llik=FALSE}.}

\item{SPDE}{logical; if \code{SPDE=TRUE} the SPDE approximation for the Gaussian spatial model is used. Default is \code{SPDE=FALSE}.}

\item{mesh}{an object obtained as result of a call to the function \code{inla.mesh.2d}.}

\item{SPDE.analytic.hessian}{logical; if \code{SPDE.analytic.hessian=TRUE} computation of the hessian matrix using the SPDE approximation is carried out using analytical expressions, otherwise a numerical approximation is used. Defauls is \code{SPDE.analytic.hessian=FALSE}.}
}
\value{
An object of class "PrevMap".
The function \code{\link{summary.PrevMap}} is used to print a summary of the fitted model.
The object is a list with the following components:

\code{estimate}: estimates of the model parameters; use the function \code{\link{coef.PrevMap}} to obtain estimates of covariance parameters on the original scale.

\code{covariance}: covariance matrix of the ML estimates.

\code{log.lik}: maximum value of the log-likelihood.

\code{y}: response variable.

\code{D}: matrix of covariates.

\code{coords}: matrix of the observed sampling locations.

\code{ID.coords}: set of ID values defined through the argument \code{ID.coords}.

\code{method}: method of optimization used.

\code{kappa}: fixed value of the shape parameter of the Matern function.

\code{knots}: matrix of the spatial knots used in the low-rank approximation.

\code{const.sigma2}: adjustment factor for \code{sigma2} in the low-rank approximation.

\code{fixed.rel.nugget}: fixed value for the relative variance of the nugget effect.

\code{mesh}: the mesh used in the SPDE approximation.

\code{call}: the matched call.
}
\description{
This function performs maximum likelihood estimation for the geostatistical linear Gaussian Model.
}
\details{
This function estimates the parameters of a geostatistical linear Gaussian model, specified as
\deqn{Y = d'\beta + S(x) + Z,}
where \eqn{Y} is the measured outcome, \eqn{d} is a vector of coavariates, \eqn{\beta} is a vector of regression coefficients, \eqn{S(x)} is a stationary Gaussian spatial process and \eqn{Z} are independent zero-mean Gaussian variables with variance \code{tau2}. More specifically, \eqn{S(x)} has an isotropic Matern covariance function with variance \code{sigma2}, scale parameter \code{phi} and shape parameter \code{kappa}. In the estimation, the shape parameter \code{kappa} is treated as fixed. The relative variance of the nugget effect, \code{nu2=tau2/sigma2}, can be fixed though the argument \code{fixed.rel.nugget}; if \code{fixed.rel.nugget=NULL}, then the variance of the nugget effect is also included in the estimation.

\bold{Locations with multiple observations.}
If multiple observations are available at any of the sampled locations the above model is modified as follows. Let \eqn{Y_{ij}} denote the random variable associated to the measured outcome for the j-th individual at location \eqn{x_{i}}. The linear geostatistical model assumes the form \deqn{Y_{ij} = d_{ij}'\beta + S(x_{i}) + Z{i} + U_{ij},} where \eqn{S(x_{i})} and \eqn{Z_{i}} are specified as mentioned above, and \eqn{U_{ij}} are i.i.d. zer0-mean Gaussian variable with variance \eqn{\omega^2}. his model can be fitted by specifing a vector of ID for the unique set locations thourgh the argument \code{ID.coords} (see also \code{\link{create.ID.coords}}).

\bold{Low-rank approximation.}
In the case of very large spatial data-sets, a low-rank approximation of the Gaussian spatial process \eqn{S(x)} can be computationally beneficial. Let \eqn{(x_{1},\dots,x_{m})} and \eqn{(t_{1},\dots,t_{m})} denote the set of sampling locations and a grid of spatial knots covering the area of interest, respectively. Then \eqn{S(x)} is approximated as \eqn{\sum_{i=1}^m K(\|x-t_{i}\|; \phi, \kappa)U_{i}}, where \eqn{U_{i}} are zero-mean mutually independent Gaussian variables with variance \code{sigma2} and \eqn{K(.;\phi, \kappa)} is the isotropic Matern kernel (see \code{\link{matern.kernel}}). Since the resulting approximation is no longer a stationary process, the parameter \code{sigma2} is adjusted by a factor\code{constant.sigma2}. See \code{\link{adjust.sigma2}} for more details on the the computation of the adjustment factor \code{constant.sigma2} in the low-rank approximation.
}
\references{
Diggle, P.J., Giorgi, E. (2019). \emph{Model-based Geostatistics for Global Public Health.} CRC/Chapman & Hall.

Giorgi, E., Diggle, P.J. (2017). \emph{PrevMap: an R package for prevalence mapping.} Journal of Statistical Software. 78(8), 1-29. doi: 10.18637/jss.v078.i08

Higdon, D. (1998). \emph{A process-convolution approach to modeling temperatures in the North Atlantic Ocean.} Environmental and Ecological Statistics 5, 173-190.
}
\seealso{
\code{\link{shape.matern}}, \code{\link{summary.PrevMap}}, \code{\link{coef.PrevMap}}, \code{matern}, \code{\link{matern.kernel}}, \code{\link{maxBFGS}}, \code{\link{nlminb}}.
}
\author{
Emanuele Giorgi \email{e.giorgi@lancaster.ac.uk}

Peter J. Diggle \email{p.diggle@lancaster.ac.uk}
}
