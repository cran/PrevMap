% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data_sim.R
\docType{data}
\name{data_sim}
\alias{data_sim}
\title{Simulated binomial data-set over the unit square}
\format{
A data frame with 900 rows and 5 variables
}
\usage{
data(data_sim)
}
\description{
This binomial data-set was simulated by generating a zero-mean Gaussian process over a 30 by 30 grid covering the unit square. The parameters used in the simulation are \code{sigma2=1}, \code{phi=0.15} and \code{kappa=2}. The nugget effect was not included, hence \code{tau2=0}.
The variables are as follows:

\itemize{
  \item y binomial observations.
  \item units.m binomial denominators. 
  \item x1 horizontal coordinates.
  \item x2 vertical coordinates.
  \item S simulated values of the Gaussian process.
}
}
\keyword{datasets}
