% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/foo.R
\name{plot.pred.PrevMap}
\alias{plot.pred.PrevMap}
\title{Plot of a predicted surface}
\usage{
\method{plot}{pred.PrevMap}(x, type = NULL, summary = "predictions", ...)
}
\arguments{
\item{x}{an object of class "PrevMap".}

\item{type}{a character indicating the type of prediction to display: 'prevalence','odds', 'logit' or 'probit' for binomial models; "log" or "exponential" for Poisson models. Default is \code{NULL}.}

\item{summary}{character indicating which summary to display: 'predictions','quantiles', 'standard.errors' or 'exceedance.prob'; default is 'predictions'. If \code{summary="exceedance.prob"}, the argument \code{type} is ignored.}

\item{...}{further arguments passed to \code{\link{plot}} of the 'raster' package.}
}
\description{
\code{plot.pred.PrevMap} displays predictions obtained from \code{\link{spatial.pred.linear.MLE}}, \code{\link{spatial.pred.linear.Bayes}},\code{\link{spatial.pred.binomial.MCML}}, \code{\link{spatial.pred.binomial.Bayes}} and \code{\link{spatial.pred.poisson.MCML}}.
}
\author{
Emanuele Giorgi \email{e.giorgi@lancaster.ac.uk}

Peter J. Diggle \email{p.diggle@lancaster.ac.uk}
}
