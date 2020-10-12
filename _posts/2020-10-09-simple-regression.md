---
title: "Simple linear regression"
author: "Jae Kwan Koo"
date: 2020-10-10 10:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
pin: true
---  

  - [Library](#library)
  - [Week 1](#week-1)
      - [R Markdown](#r-markdown)
      - [Including Plots](#including-plots)
      - [Refer](#refer)

## Library

``` r
library(data.table)
library(tidyverse)

library(plotly)
library(gridExtra)
```

# Week 1

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

<img src="2020-10-09-simple-regression_files/figure-gfm/pressure-1.png" style="display: block; margin: auto;" />

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.

## Refer

[선형회귀](https://www.datacamp.com/community/tutorials/linear-regression-R#coefficients)

[원점을 지나는 회귀 in
ggplot](https://stackoverflow.com/questions/26705554/extend-geom-smooth-in-a-single-direction)