# Data Exploration

## Load Packages and Data

    library(tidyverse)

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.4
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.4.4     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.0
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

    library(coefplot)

    library(glmnet)

    ## Loading required package: Matrix
    ## 
    ## Attaching package: 'Matrix'
    ## 
    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack
    ## 
    ## Loaded glmnet 4.1-8

    library(ggplot2)



    library(boot)

    library(caret)

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'lattice'
    ## 
    ## The following object is masked from 'package:boot':
    ## 
    ##     melanoma
    ## 
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

    library(AppliedPredictiveModeling)

    library(reshape2)

    ## 
    ## Attaching package: 'reshape2'
    ## 
    ## The following object is masked from 'package:tidyr':
    ## 
    ##     smiths

    library(Rcpp)

    library(rstanarm)

    ## This is rstanarm version 2.32.1
    ## - See https://mc-stan.org/rstanarm/articles/priors for changes to default priors!
    ## - Default priors may change, so it's safest to specify priors, even if equivalent to the defaults.
    ## - For execution on a local, multicore CPU with excess RAM we recommend calling
    ##   options(mc.cores = parallel::detectCores())
    ## 
    ## Attaching package: 'rstanarm'
    ## 
    ## The following objects are masked from 'package:caret':
    ## 
    ##     compare_models, R2
    ## 
    ## The following object is masked from 'package:boot':
    ## 
    ##     logit
    ## 
    ## The following object is masked from 'package:coefplot':
    ## 
    ##     invlogit

    library(magrittr) 

    ## 
    ## Attaching package: 'magrittr'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     set_names
    ## 
    ## The following object is masked from 'package:tidyr':
    ## 
    ##     extract

    library(dplyr)

    df_train <- readr::read_csv("paint_project_train_data.csv", col_names = TRUE)

    ## Rows: 835 Columns: 8
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): Lightness, Saturation
    ## dbl (6): R, G, B, Hue, response, outcome
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

    df_holdout <- readr::read_csv("paint_project_holdout_data.csv", col_names = TRUE)

    ## Rows: 844 Columns: 6
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): Lightness, Saturation
    ## dbl (4): R, G, B, Hue
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

    df_bonus <- readr::read_csv("paint_project_bonus_data.csv", col_names = TRUE)

    ## Rows: 1764 Columns: 9
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): Lightness, Saturation
    ## dbl (7): R, G, B, Hue, response, outcome, challenge_outcome
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

    df_train %>% glimpse()

    ## Rows: 835
    ## Columns: 8
    ## $ R          <dbl> 172, 26, 172, 28, 170, 175, 90, 194, 171, 122, 0, 88, 144, …
    ## $ G          <dbl> 58, 88, 94, 87, 66, 89, 78, 106, 68, 151, 121, 140, 82, 163…
    ## $ B          <dbl> 62, 151, 58, 152, 58, 65, 136, 53, 107, 59, 88, 58, 132, 50…
    ## $ Lightness  <chr> "dark", "dark", "dark", "dark", "dark", "dark", "dark", "da…
    ## $ Saturation <chr> "bright", "bright", "bright", "bright", "bright", "bright",…
    ## $ Hue        <dbl> 4, 31, 8, 32, 5, 6, 34, 10, 1, 21, 24, 22, 36, 16, 26, 12, …
    ## $ response   <dbl> 12, 10, 16, 10, 11, 16, 10, 19, 14, 25, 14, 19, 14, 38, 15,…
    ## $ outcome    <dbl> 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,…

## Data Processing

Use boot::logit for response

    #min_value <- min(df_train$response, na.rm = TRUE) 
    #max_value <- max(df_train$response, na.rm = TRUE) 

    min_value <- 0
    max_value <- 100

    # Normalize to the (0, 1) range
    train_response_normalized <- (df_train$response - min_value) / (max_value - min_value)


    df_train$response_logit <- boot::logit(train_response_normalized)

    #df_train$response_logit 



    # Applying the inverse logit
    #df_train$response_back <- boot::inv.logit(df_train$response_logit)

    # Denormalize if necessary
    #df_train$response_original_scale <- df_train$response_back * (max_value - min_value) + min_value

    df_train %>% head()

    ## # A tibble: 6 × 9
    ##       R     G     B Lightness Saturation   Hue response outcome response_logit
    ##   <dbl> <dbl> <dbl> <chr>     <chr>      <dbl>    <dbl>   <dbl>          <dbl>
    ## 1   172    58    62 dark      bright         4       12       1          -1.99
    ## 2    26    88   151 dark      bright        31       10       1          -2.20
    ## 3   172    94    58 dark      bright         8       16       1          -1.66
    ## 4    28    87   152 dark      bright        32       10       0          -2.20
    ## 5   170    66    58 dark      bright         5       11       0          -2.09
    ## 6   175    89    65 dark      bright         6       16       0          -1.66

## Visualization

Counts for categorical variables.

    # levels
    df_train %>%  tail()

    ## # A tibble: 6 × 9
    ##       R     G     B Lightness Saturation   Hue response outcome response_logit
    ##   <dbl> <dbl> <dbl> <chr>     <chr>      <dbl>    <dbl>   <dbl>          <dbl>
    ## 1   199   204   216 soft      subdued       33       60       0         0.405 
    ## 2   214   184   189 soft      subdued        2       52       0         0.0800
    ## 3   226   199   182 soft      subdued        9       60       0         0.405 
    ## 4   241   211   202 soft      subdued        7       69       0         0.800 
    ## 5   227   208   213 soft      subdued        1       66       0         0.663 
    ## 6   233   207   200 soft      subdued        6       65       0         0.619

    df_train$Lightness %>% unique()

    ## [1] "dark"      "deep"      "light"     "midtone"   "pale"      "saturated"
    ## [7] "soft"

    df_train$Saturation %>% unique()

    ## [1] "bright"  "gray"    "muted"   "neutral" "pure"    "shaded"  "subdued"

We start by plotting discrete variables:

    ggplot(df_train, aes(x = factor(Lightness))) + 
      geom_bar(fill = "red", color = "black") +
      ggtitle("Counts")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-3-1.png)

    ggplot(df_train, aes(x = factor(Saturation))) + 
      geom_bar(fill = "red", color = "black") +
      ggtitle("Counts")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-3-2.png)

    ggplot(df_train, aes(x = factor(outcome))) + 
      geom_bar(fill = "red", color = "black") +
      ggtitle("Counts")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-3-3.png) We
note that Lightness and Saturation have fairly uniform distribution. The
loss of “gray” in Saturation is likely due to aesthetic or practical
reasons.

Then continuous variables, including RGB, Hue and response values.

    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")
    desired_bins <- 30
    for(var in continuous_vars) {
      var_range <- range(df_train[[var]], na.rm = TRUE)

      binwidth_var <- (var_range[2] - var_range[1]) / desired_bins

      print(
        ggplot(df_train, aes_string(x = var)) +
        geom_histogram(binwidth = binwidth_var, fill = "skyblue", color = "black") +
        ggtitle(paste("Distribution of", var))
      )
    }

    ## Warning: `aes_string()` was deprecated in ggplot2 3.0.0.
    ## ℹ Please use tidy evaluation idioms with `aes()`.
    ## ℹ See also `vignette("ggplot2-in-packages")` for more information.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

![](customer_data_files/figure-markdown_strict/unnamed-chunk-4-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-4-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-4-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-4-4.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-4-5.png)
None of the continuous variables follows Gaussian-like distribution.

## Continuous Inputs conditioned on the Discrete Inputs

    # Lightness
    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")

    for(var in continuous_vars) {

      var_range <- range(df_train[[var]], na.rm = TRUE)
      desired_bins <- 30
      binwidth_var <- (var_range[2] - var_range[1]) / desired_bins
      
      # Plot with faceting by 'Lightness'
    print(
        ggplot(df_train, aes_string(x = var)) +
        geom_histogram(aes(y = ..density..), binwidth = binwidth_var, fill = "skyblue", color = "black") +
        geom_density(alpha = 0.2, fill = "red") +
        facet_wrap(~ Lightness) +
        theme_minimal() +
        ggtitle(paste( var, "vs. Levels of Lightness"))
      )
    }

    ## Warning: The dot-dot notation (`..density..`) was deprecated in ggplot2 3.4.0.
    ## ℹ Please use `after_stat(density)` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

![](customer_data_files/figure-markdown_strict/unnamed-chunk-5-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-5-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-5-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-5-4.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-5-5.png)
Given different levels of light, especially lighter shades, continuous
*GRB* and *logit response* variable follows obvious distribution.

It is almost uniform for Hue, however, when levels of Lightness applied.
This is reasonable, since in this metric, Hue and Lightness are
suppposed to be independent with each other.

    # Saturation
    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")

    for(var in continuous_vars) {
      # Dynamically adjust binwidth 
      var_range <- range(df_train[[var]], na.rm = TRUE)
      desired_bins <- 30
      binwidth_var <- (var_range[2] - var_range[1]) / desired_bins
      
      # Plot with faceting by 'Saturation'
      print(
        ggplot(df_train, aes_string(x = var)) +
        geom_histogram(aes(y = ..density..), binwidth = binwidth_var, fill = "skyblue", color = "black") +
        geom_density(alpha = 0.2, fill = "red") +
        facet_wrap(~ Saturation) +
        theme_minimal() +
        ggtitle(paste( var, "vs. Levels of Saturation"))
      )
    }

![](customer_data_files/figure-markdown_strict/unnamed-chunk-6-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-6-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-6-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-6-4.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-6-5.png)
On the other hand, Saturation has less influence on continuous variables
compared to Lightness. We observe some trends on RGB, but more
diffusive.

    # Outcome
    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")

    for(var in continuous_vars) {
      # Dynamically adjust binwidth 
      var_range <- range(df_train[[var]], na.rm = TRUE)
      desired_bins <- 30
      binwidth_var <- (var_range[2] - var_range[1]) / desired_bins
      
      # Plot with faceting by 'Outcome'
      print(
        ggplot(df_train, aes_string(x = var)) +
        geom_histogram(aes(y = ..density..), binwidth = binwidth_var, fill = "skyblue", color = "black") +
    #     geom_boxplot(fill = "red", color = "black") +
          geom_density(alpha = 0.2, fill = "red") +
        facet_wrap(~ outcome) +
        theme_minimal() +
        ggtitle(paste( var, "vs. Levels of Outcome"))
      )
    }

![](customer_data_files/figure-markdown_strict/unnamed-chunk-7-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-7-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-7-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-7-4.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-7-5.png)
The difference between levels on outcome is not strong.

## Continuous Inputs vs. Continuous Inputs

We use `caret::featurePlot` to visualize the relation between coninuous
inputs.

    #continuous_vars <- c("R", "G", "B", "Hue", "response_logit")
    continuous_data <- df_train[ c("R", "G", "B", "Hue")]

    # Ensure 'outcome' is a factor

    #df_train$outcome <- as.factor(df_train$outcome)


    # Including 'outcome' for coloring in the plot
    featurePlot(x = continuous_data,
                y = as.factor(df_train$outcome),
                plot = "pairs", alpha = 0.7,
                auto.key = list(columns = 2))

![](customer_data_files/figure-markdown_strict/unnamed-chunk-8-1.png)

To study the correlation among the continuous inputs RBG and Hue, we
need to compute the covariance matrix.

    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")


    df_train_cor <-  df_train %>%dplyr::select(continuous_vars) %>% cor()

    ## Warning: Using an external vector in selections was deprecated in tidyselect 1.1.0.
    ## ℹ Please use `all_of()` or `any_of()` instead.
    ##   # Was:
    ##   data %>% select(continuous_vars)
    ## 
    ##   # Now:
    ##   data %>% select(all_of(continuous_vars))
    ## 
    ## See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

    df_train_cor 

    ##                         R         G         B         Hue response_logit
    ## R               1.0000000 0.7536893 0.5337811 -0.35907605     0.82576132
    ## G               0.7536893 1.0000000 0.8131972  0.05555940     0.98212316
    ## B               0.5337811 0.8131972 1.0000000  0.21223987     0.82914546
    ## Hue            -0.3590761 0.0555594 0.2122399  1.00000000    -0.01543818
    ## response_logit  0.8257613 0.9821232 0.8291455 -0.01543818     1.00000000

    #highCorr <- sum(abs(df_train_cor[upper.tri(df_train_cor)]) > .7)
    #highCorr


    #Convert the correlation_matrix into a long format suitable for ggplot
    melted_correlation_matrix <- melt(df_train_cor)

    ggplot(data = melted_correlation_matrix, aes(x = Var1, y = Var2, fill = value)) + 
      geom_tile() + 
      scale_fill_gradient2(low = "black", high = "red", mid = "white", midpoint = 0) +
      ggtitle("Correlation Matrix of Continuous Variables")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-9-1.png)
Note that G and response\_logit have pretty high correlations. Other
pairs are less so.

## Continuous Inputs vs. Continuous Outputs

    continuous_vars <- c("R", "G", "B", "Hue", "response_logit")
    continuous_data <- df_train[ c("R", "G", "B", "Hue", "response_logit")]

    # Ensure 'outcome' is a factor
    df_train$outcome <- as.factor(df_train$outcome)


    for(var in  c("R", "G", "B", "Hue")){
      print(
        ggplot(df_train, aes_string(x = var, y = 'response_logit', color = 'outcome')) +
        geom_point(alpha = 0.8, size = 3) +
        geom_smooth() +
        theme_minimal() +
        labs(title =  "Continuous Variable vs. response_logit with Outcome levels")
        
      )
    }

    ## `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](customer_data_files/figure-markdown_strict/unnamed-chunk-10-1.png)

    ## `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](customer_data_files/figure-markdown_strict/unnamed-chunk-10-2.png)

    ## `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](customer_data_files/figure-markdown_strict/unnamed-chunk-10-3.png)

    ## `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](customer_data_files/figure-markdown_strict/unnamed-chunk-10-4.png)
We see a clear correlation between RGB and `response_logit`, among which
`G` is the strongest. It is less so from `Hue`. From the pairs with
correlations, he trends depend on the categorical outputs, `outcome`.

To further investigate the influence of categorical inputs, we repeat
the code above.

    # Saturation
    for(var in  c("R", "G", "B", "Hue")){
      print(
        ggplot(df_train, aes_string(x = var, y = 'response_logit')) +
          geom_point(alpha = 0.8, size = 1, color = "red") +
          facet_wrap(~ Saturation) +
          theme_minimal() +
          labs(title =  "Continuous Variable vs. response_logit with Saturation Levels")
      )
    }

![](customer_data_files/figure-markdown_strict/unnamed-chunk-11-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-11-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-11-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-11-4.png)
Again, `G` is the most dominant variable while `RGB` all contributes.
Trends do depend on the levels of Saturation; for some inputs like
`grey`, `neutral` and `shaded`, it is more significant. For `Hue`,
however, it showed no obvious trend.

    # Lightness
    for(var in  c("R", "G", "B", "Hue")){
      print(
        ggplot(df_train, aes_string(x = var, y = 'response_logit')) +
          geom_point(alpha = 0.8, size = 1, color = "red") +
          facet_wrap(~ Lightness) +
          theme_minimal() +
          labs(title =  "Continuous Variable vs. response_logit with Lightness Levels")
      )
    }

![](customer_data_files/figure-markdown_strict/unnamed-chunk-12-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-12-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-12-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-12-4.png)
The results for levels of `Lightness` agrees with previous findings
mostly. Only one difference is `Hue` actually started to show trends
with respect to levels of `Lightness`.

## Continuous/Discrete Inputs vs Discrete Outputs

To visualize the behavior of the binary outcome with respect to the
continuous inputs and to visualize the behavior of the binary outcome
with respect to the categorical inputs, we make the plot of continuous
variables vs `outcome` and facet wrap them with respect to the discrete
variables.

Note that we used `geom_jitter()` to avoid overlapping and to improve
visualization.

    # Saturation

    df_train <- readr::read_csv("paint_project_train_data.csv", col_names = TRUE)

    ## Rows: 835 Columns: 8
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): Lightness, Saturation
    ## dbl (6): R, G, B, Hue, response, outcome
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

    min_value <- 0
    max_value <- 100

    # Normalize to the (0, 1) range
    train_response_normalized <- (df_train$response - min_value) / (max_value - min_value)
    df_train$response_logit <- boot::logit(train_response_normalized)
    df_train_cla <-dplyr::select(df_train, -response_logit, -response)




    var <- "R"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Saturation) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-13-1.png)

    var <- "G"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Saturation) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-13-2.png)

    var <- "B"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color  = 'blue') +
        #facet_wrap(~Saturation) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-13-3.png)

    var <- "Hue"
    deg <- 2
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Saturation) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-13-4.png)

    # Lightness


    #for (var in c("R", "G", "B", "Hue")) {}
    var <- "R"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Lightness) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-14-1.png)

    var <- "G"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Lightness) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-14-2.png)

    var <- "B"
    deg <- 1
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Lightness) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-14-3.png)

    var <- "Hue"
    deg <- 2
    df_train %>%
        ggplot(aes_string(x = var, y = "outcome")) +  
        geom_jitter(height = 0.02, width = 0, color = "red") +
        geom_smooth(formula = y ~ poly(x, deg), method = "glm", 
                    method.args = list(family = "binomial"), color = "blue") +
        facet_wrap(~Lightness) +
        labs(title = paste("Logistic Regression Fit for", var, "vs. Outcome")) +
        theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-14-4.png)

Given levels of discrete variable, the all continuous variable including
`Hue` showed relatively strong trends with respect to `outcome`. For
`glm` model, our suggested degree for R, G, B and Hue are 1, 1, 1, 2

# Regression

## Linear Models

We plan to train *10 different models* using `lm()`:

1.  Intercept-only model – no INPUTS!

2.  Categorical variables only – linear additive

3.  Continuous variables only – linear additive

4.  All categorical and continuous variables – linear additive

5.  Interaction of the categorical inputs with all continuous inputs
    main effects

6.  Add categorical inputs to all main effect and all pairwise
    interactions of continuous inputs

7.  Interaction of the categorical inputs with all main effect and all
    pairwise interactions of continuous inputs

8.  Try non-linear basis functions based on your EDA. Guess: (R^2 +
    G + B) \* (Lightness + 1)

9.  Can consider interactions of basis functions with other basis
    functions!

10. Can consider interactions of basis functions with the categorical
    inputs!

First, we’ll fit several linear models. Starting from simple (intercept
only) to more complex models with interactions and possibly non-linear
transformations (basis functions).

Note that we start by splitting the training data to training and
testing. This reduce the overfitting while applying our metric.

    set.seed(123)
    inTraining <- createDataPartition(df_train$response_logit, p=.95, list = FALSE)
    training_reg <- df_train[inTraining,]
    testing_reg  <- df_train[-inTraining,]


    # Intercept-only model – no INPUTS!
    mod_reg_1 <- lm(response_logit ~ 1, data = training_reg)
    #mod_reg_1 <- lm(response_logit ~ (Lightness + Saturation + Hue) * (R + G + B), data = training_reg)

    # Categorical variables only – linear additive
    mod_reg_2 <- lm(response_logit ~ Lightness + Saturation, data = training_reg)

    # Continuous variables only – linear additive
    mod_reg_3 <- lm(response_logit ~ R + G + B + Hue, data = training_reg)

    # All categorical and continuous variables – linear additive
    mod_reg_4 <- lm(response_logit ~ Lightness + Saturation + R + G + B + Hue, data = training_reg)

    # Interaction of the categorical inputs with all continuous inputs main effects
    mod_reg_5 <- lm(response_logit ~ (Lightness + Saturation) * (R + G + B + Hue), data = training_reg)

    # Add categorical inputs to all main effect and all pairwise interactions of continuous inputs
    mod_reg_6 <- lm(response_logit ~ Lightness + Saturation + R + G + B + Hue 
                    + R:G + R:B + R:Hue + G:B + G:Hue + B:Hue, data = training_reg)

    # Interaction of the categorical inputs with all 
    # main effect and all pairwise interactions of continuous inputs
    mod_reg_7 <- lm(response_logit ~ (Lightness + Saturation) *
                      (R + G + B + Hue + R:G + R:B + R:Hue + G:B + G:Hue + B:Hue), data = training_reg)

    #  Try non-linear basis functions based on your EDA
    #mod_reg_8 <- lm(response_logit ~  (R + I(R^2) + G + B) * (Lightness * Hue + Hue * Saturation), data = training_reg)
    mod_reg_8 <- lm(response_logit ~  (poly(R, 2) + poly(G, 2) + poly(B, 2) + poly(Hue, 2)) * (Lightness + Saturation), data = training_reg)


    # Can consider interactions of basis functions with other basis functions!
    mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
                    , data = training_reg)

    # Can consider interactions of basis functions with the categorical inputs!
    mod_reg_10 <- lm(response_logit ~  (Lightness + Saturation) * (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2) + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
                    , data = training_reg)

## Performance metric

Now we visualize the coefficient summaries for your top 3 models by root
mean square error(RMSE) metric and R-squared metric.

    library(modelr)

    # test

    # RMSE
    e1 <-rmse(mod_reg_1, testing_reg )
    e2 <-rmse(mod_reg_2, testing_reg )
    e3 <-rmse(mod_reg_3, testing_reg )
    e4 <-rmse(mod_reg_4, testing_reg )
    e5 <-rmse(mod_reg_5, testing_reg )
    e6 <-rmse(mod_reg_6, testing_reg )
    e7 <-rmse(mod_reg_7, testing_reg )
    e8 <-rmse(mod_reg_8, testing_reg )
    e9 <-rmse(mod_reg_9, testing_reg )
    e10 <-rmse(mod_reg_10, testing_reg )
    RMSE_errors <- data.frame(x = seq(1, 10), 
                         y = c(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10))

    RMSE_errors %>% ggplot( aes(x = x, y = y)) + 
      geom_point() + 
      geom_line(color = "red", size = 1) + 
      labs(title = "RMSE errors", x = "Model", y = "Error")

    ## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
    ## ℹ Please use `linewidth` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

![](customer_data_files/figure-markdown_strict/unnamed-chunk-16-1.png)

    # R-squared

    r_squared_1 <-rsquare(mod_reg_1, testing_reg )
    r_squared_2 <-rsquare(mod_reg_2, testing_reg )
    r_squared_3 <-rsquare(mod_reg_3, testing_reg )
    r_squared_4 <-rsquare(mod_reg_4, testing_reg )
    r_squared_5 <-rsquare(mod_reg_5, testing_reg )
    r_squared_6 <-rsquare(mod_reg_6, testing_reg )
    r_squared_7 <-rsquare(mod_reg_7, testing_reg )
    r_squared_8 <-rsquare(mod_reg_8, testing_reg )
    r_squared_9 <-rsquare(mod_reg_9, testing_reg )
    r_squared_10 <-rsquare(mod_reg_10, testing_reg )

    rsquare_errors <- data.frame(x = seq(1, 10), 
                         y = c(r_squared_1, r_squared_2, r_squared_3, r_squared_4, r_squared_5, r_squared_6, r_squared_7, r_squared_8, r_squared_9, r_squared_10))

    rsquare_errors %>% ggplot(aes(x = x, y = y)) + 
      geom_point() + 
      geom_line(color = "red", size = 1) + 
      labs(title = "R-squared", x = "Model", y = "Error")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-17-1.png)

The best 3 models according both R-squared and RMSE metric are model 9,
8 and 7. We have the visualization as the following.

    # determine the best models
    RMSE_errors[order(RMSE_errors$y),]

    ##     x          y
    ## 8   8 0.03394169
    ## 9   9 0.03882933
    ## 5   5 0.04532254
    ## 7   7 0.04550357
    ## 6   6 0.06641551
    ## 4   4 0.07500576
    ## 10 10 0.08217807
    ## 3   3 0.12746781
    ## 2   2 0.34015586
    ## 1   1 1.18594520

    rsquare_errors[order(rsquare_errors$y),]

    ##     x         y
    ## 1   1 0.0000000
    ## 2   2 0.9179053
    ## 3   3 0.9891416
    ## 10 10 0.9952042
    ## 4   4 0.9962834
    ## 6   6 0.9971826
    ## 7   7 0.9985762
    ## 5   5 0.9987477
    ## 9   9 0.9989715
    ## 8   8 0.9992521

    # coefplot
    coefplot(mod_reg_9) + labs(title = 'Coefficient Plot for Model 9')

![](customer_data_files/figure-markdown_strict/unnamed-chunk-18-1.png)

    coefplot(mod_reg_8) + labs(title = 'Coefficient Plot for Model 8')

![](customer_data_files/figure-markdown_strict/unnamed-chunk-18-2.png)

    coefplot(mod_reg_7) + labs(title = 'Coefficient Plot for Model 7')

![](customer_data_files/figure-markdown_strict/unnamed-chunk-18-3.png)

    #mod_reg_7
    #coefplot::multiplot(mod_reg_7, mod_reg_8, mod_reg_9) +
    #  labs(title = 'Coefficient Plot for Model 7, 8 and 9')

Note that the naive plotting function give messy results, then we find
the significant coefficients by selecting the ones whose confidence
interval does not intersect 0. To avoid noises, we designed a cutoff to
eliminate significant values that contribute little.

    #coef_summary <- summary(mod_reg_10)$coefficients
    #coef_summary[coef_summary[, "Pr(>|t|)"] < 0.05, ] %>% print()
    cutoff <- 0.01
    ci_9 <- confint(mod_reg_9)
    sig_9 <- ci_9[!(ci_9[,1] <= 0 & ci_9[,2] >= 0) & (abs(ci_9[,1] + ci_9[,2])/2 > cutoff), ]
    #print("Model 9")
    #print(sig_9)

    sig_9t <-  sig_9 %>% as.data.frame() %>%
      mutate(Estimate = (`2.5 %` + `97.5 %`) / 2,
             lower_bound = `2.5 %` * 1,
             upper_bound =  `97.5 %`
             )  %>%  rownames_to_column()

    sig_9t %>%  ggplot( aes(x = rowname, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 9", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-19-1.png)

    cutoff <- 1
    ci_8 <- confint(mod_reg_8)
    sig_8 <- ci_8[!(ci_8[,1] <= 0 & ci_8[,2] >= 0)  & (abs(ci_8[,1] + ci_8[,2])/2 > cutoff), ]
    #print("Model 8")
    #print(sig_8)


    sig_8t <-  sig_8 %>% as.data.frame() %>%
      mutate(Estimate = (`2.5 %` + `97.5 %`) / 2,
             lower_bound = `2.5 %` * 1,
             upper_bound =  `97.5 %`
             )  %>%  rownames_to_column()

    sig_8t %>%  ggplot( aes(x = rowname, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 8", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-19-2.png)

    cutoff <- 0.01
    ci_7 <- confint(mod_reg_7)
    sig_7 <- ci_7[!(ci_7[,1] <= 0 & ci_7[,2] >= 0) & (abs(ci_7[,1] + ci_7[,2])/2 > cutoff), ]
    #print("Model 7")
    #print(sig_7)



    sig_7t <- sig_7 %>% as.data.frame() %>%
      mutate(Estimate = (`2.5 %` + `97.5 %`) / 2,
             lower_bound = `2.5 %` * 1,
             upper_bound =  `97.5 %`
             )  %>%  rownames_to_column()

    sig_7t %>%  ggplot( aes(x = rowname, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 7", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-19-3.png)
`Lightness` is a key factor. Especially in model 7, where Lightness, or
the interaction with it, dominated every significant coefficient. Second
to lightness, `Saturation` also dominated in model 8 and 9.

## Bayesian Linear models

We will fit model 9 and model 8. Model 9 is the obvious choice as our
best performer. Model 8 comes from our EDA and its performance is almost
the same as model 10 but with less complexity.

Here we will use `library(rstanarm)` and `stan_lm()`.

    # Recall our models
    # mod_reg_8 <- lm(response_logit ~  (poly(R, 2) + poly(G, 2) + poly(B, 2) + poly(Hue, 2)) * (Lightness + Saturation), data = training_reg)

    #mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
    #                , data = training_reg)



    #model_reg_7_B <- stan_lm(response_logit ~  (Lightness + Saturation) *
    #                         (R + G + B + Hue + R:G + R:B + R:Hue + G:B + G:Hue + B:Hue), 
    #                        data = training, 
    #                        prior = R2(location = 0.5),  
    #                        seed = 123) 
    #mod_reg_7 <- lm(response_logit ~ (Lightness + Saturation) *
    #                  (R + G + B + Hue + R:G + R:B + R:Hue + G:B + G:Hue + B:Hue), data = df_train)



    library(rstanarm)

    df_train_B <- df_train %>% mutate(
      R2 = R^2,
      G2 = G^2,
      B2 = B^2,
      Hue2 = Hue^2
      )


    set.seed(123)
    inTraining <- createDataPartition(df_train_B$response_logit, p=.5, list = FALSE)
    training <- df_train_B[inTraining,]
    testing  <- df_train_B[-inTraining,]

    model_reg_8_B <- stan_lm(response_logit ~ (R + G + B + R2 + G2 + B2 + Hue + Hue2) * (Lightness + Saturation), 
                            data = training, 
                            prior = R2(location = 0.5),  
                            seed = 123) 

    #model_reg_8_B_permanant <- model_reg_8_B

    # Save Model
    model_reg_8_B %>% readr::write_rds("model_reg_8_B.rds")
    #re_load_model_reg_8_B <- readr::read_rds("model_reg_8_B.rds")


    model_reg_9_B <- stan_lm(response_logit ~  (  (R+R2) * ((B+B2)+(G+G2)+(Hue + Hue2)) + 
                                                                         ((B+B2) * ((G+G2)+(Hue + Hue2)) +
                                                                         (G+G2) *(Hue + Hue2) ) ),
                                      data = training, 
                                      prior = R2(location = 0.5),  
                                      seed = 123)  

    #model_reg_9_B_permanant <- model_reg_9_B

    # Save Model
    model_reg_9_B %>% readr::write_rds("model_reg_9_B.rds")
    #re_load_model_reg_9_B <- readr::read_rds("model_reg_9_B.rds")

    model_reg_8_B <- readr::read_rds("model_reg_8_B.rds")
    model_reg_9_B <- readr::read_rds("model_reg_9_B.rds")

    # performance  metric
    rsquare(model_reg_8_B, data = testing)

    ## Warning in response(model, data) - stats::predict(model, data): longer object
    ## length is not a multiple of shorter object length

    ## [1] 0.5216605

    rsquare(model_reg_9_B, data = testing)

    ## Warning in response(model, data) - stats::predict(model, data): longer object
    ## length is not a multiple of shorter object length

    ## [1] 0.5346566

    rmse(model_reg_8_B, data = testing)

    ## Warning in response(model, data) - stats::predict(model, data): longer object
    ## length is not a multiple of shorter object length

    ## [1] 0.8208032

    rmse(model_reg_9_B, data = testing)

    ## Warning in response(model, data) - stats::predict(model, data): longer object
    ## length is not a multiple of shorter object length

    ## [1] 0.8097224

From both R-squared and RMSE, model 9 has slightly better performance.

    #library(bayesplot)
    #library(rstanarm)

    # coefficient plot
    plot(model_reg_8_B)

![](customer_data_files/figure-markdown_strict/unnamed-chunk-23-1.png)

    plot(model_reg_9_B) 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-23-2.png)

    # posterior interval
    cutoff <- 0.001

    pi_8_B <- posterior_interval(model_reg_8_B)

    significant_8_B <- pi_8_B[!(pi_8_B[,1] <= 0 & pi_8_B[,2] >= 0) & (abs(pi_8_B[,1] + pi_8_B[,2])/2 > cutoff), ]
    print("Model 8")

    ## [1] "Model 8"

    print(significant_8_B)

    ##                                  5%           95%
    ## (Intercept)            -4.171761521 -3.284640e+00
    ## G                       0.012008686  2.380945e-02
    ## Hue                    -0.056599147 -1.429591e-02
    ## Lightnessdeep           0.236799734  1.119573e+00
    ## Lightnesslight          5.400383720  1.658754e+01
    ## Lightnessmidtone        1.902050398  3.936465e+00
    ## Lightnesspale          15.121009322  3.426134e+01
    ## Lightnesssaturated      0.884528485  1.988071e+00
    ## Lightnesssoft           4.229625979  9.147852e+00
    ## Saturationgray         -1.094912807 -1.837054e-01
    ## Saturationneutral      -1.115639115 -3.262075e-01
    ## Saturationshaded       -0.871457125 -6.262533e-02
    ## Saturationsubdued      -0.941339422 -1.473276e-01
    ## R:Lightnessdeep        -0.003351004 -4.947903e-05
    ## R:Lightnessmidtone     -0.010076771 -1.617670e-03
    ## R:Lightnesssoft        -0.022254140 -2.874566e-03
    ## G:Lightnessdeep        -0.012569620 -8.752630e-04
    ## G:Lightnesslight       -0.157152950 -5.907732e-02
    ## G:Lightnessmidtone     -0.032370165 -1.370513e-02
    ## G:Lightnesspale        -0.260954441 -8.677934e-02
    ## G:Lightnesssaturated   -0.023198554 -1.070446e-02
    ## G:Lightnesssoft        -0.075279812 -2.720574e-02
    ## G:Saturationgray        0.004398044  3.985227e-02
    ## G:Saturationmuted       0.002552934  9.832181e-03
    ## G:Saturationneutral     0.003895959  1.882560e-02
    ## G:Saturationpure       -0.007675125 -3.519510e-04
    ## G:Saturationsubdued     0.002790036  1.185167e-02
    ## B:Lightnessmidtone     -0.017691209 -1.722800e-03
    ## B:Lightnesssoft        -0.021253923 -2.820022e-03
    ## B:Saturationmuted      -0.007009615 -1.716616e-03
    ## Hue:Lightnesslight      0.014291429  5.253480e-02
    ## Hue:Lightnessmidtone    0.010438375  4.707467e-02
    ## Hue:Lightnesspale       0.008370480  5.015659e-02
    ## Hue:Lightnesssaturated  0.014344992  4.696160e-02
    ## Hue:Lightnesssoft       0.010222827  4.996968e-02
    ## sigma                   0.051300075  6.106829e-02

    pi_9_B <- posterior_interval(model_reg_9_B)

    significant_9_B <- pi_9_B[!(pi_9_B[,1] <= 0 & pi_9_B[,2] >= 0) & (abs(pi_9_B[,1] + pi_9_B[,2])/2 > cutoff), ]
    print("Model 9")

    ## [1] "Model 9"

    print(significant_9_B)

    ##                        5%          95%
    ## (Intercept) -4.6067261022 -3.288070599
    ## R           -0.0165166524 -0.001426197
    ## G            0.0153637687  0.035025753
    ## Hue         -0.1088015433 -0.042768469
    ## Hue2         0.0007009493  0.002524114
    ## sigma        0.0449613551  0.050939334

Next we study the posterior UNCERTAINTY on the likelihood noise.

    sigma_posterior <- as.data.frame(model_reg_8_B)[, "sigma"]  # Extract the posterior samples of sigma
    sigma_df8 <- data.frame(sigma = as.data.frame(model_reg_8_B)[, "sigma"])


    sigma_df8 %>% ggplot( aes(x = sigma)) +
      geom_histogram(bins = 30, fill = "red") +
      ggtitle("Posterior Distribution of sigma") +
      xlab("sigma") +
      ylab("Density") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-24-1.png)

    sigma_posterior <- as.data.frame(model_reg_9_B)[, "sigma"]  # Extract the posterior samples of sigma
    sigma_df9 <- data.frame(sigma = as.data.frame(model_reg_9_B)[, "sigma"])


    sigma_df9 %>% ggplot( aes(x = sigma)) +
      geom_histogram(bins = 30, fill = "red") +
      ggtitle("Posterior Distribution of sigma") +
      xlab("sigma") +
      ylab("Density") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-24-2.png)
In our case, the variance of *σ* is relatively small, we do have precise
value.

## Linear Model Predictions

We will pick model 8 for it is compatible with EDA and model 9 for its
performance. We at first give a direct comparison on the prediction and
testing results.

    #mod_reg_8 <- lm(response_logit ~  (poly(R, 2) + poly(G, 2) + poly(B, 2) + poly(Hue, 2)) * (Lightness + Saturation), data = training_reg)


    # Can consider interactions of basis functions with other basis functions!
    #mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
    #                , data = training_reg)

    set.seed(123)
    inTraining <- createDataPartition(df_train$response_logit, p=.95, list = FALSE)
    training_reg <- df_train[inTraining,]
    testing_reg  <- df_train[-inTraining,]


    # prediction
    predictions_reg_8 <- predict(mod_reg_8, newdata = testing_reg)
    predictions_reg_9 <- predict(mod_reg_9, newdata = testing_reg)


    ggplot(testing_reg, aes(x = response_logit)) +
      geom_point(aes(y = predictions_reg_8), color = 'blue', alpha = 0.5) +
      #geom_point(aes(y = predictions_reg_9), color = 'red', alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
      facet_wrap(~Lightness)+
        labs(title = "mod_reg_8",
           x = "Actual response_logit",
           y = "Predicted response_logit") 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-25-1.png)

    ggplot(testing_reg, aes(x = response_logit)) +
      geom_point(aes(y = predictions_reg_9), color = 'blue', alpha = 0.5) +
      facet_wrap(~Lightness)+
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
      labs(title = "mod_reg_9",
           x = "Actual response_logit",
           y = "Predicted response_logit") 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-25-2.png)

Next, we include the predictive mean trend, the confidence interval on
the mean, and the prediction interval on the response.

    ## Model 8
    prediction_details <- mod_reg_8 %>% predict(newdata = testing_reg, type = "response", interval = "prediction", level = 0.95)


    testing_reg$predicted_response = prediction_details[, "fit"]
    testing_reg$lower_pi = prediction_details[, "lwr"]
    testing_reg$upper_pi = prediction_details[, "upr"]


    mean_predictions <- mod_reg_8 %>%  predict( newdata = testing_reg, type = "response", interval = "confidence", level = 0.95)
    testing_reg$lower_ci = mean_predictions[, "lwr"]
    testing_reg$upper_ci = mean_predictions[, "upr"]

    ggplot(testing_reg, aes(x = response_logit, y = predicted_response)) +
      geom_point(color = 'blue', alpha = 0.8, size = 2) +
      #geom_line(aes(y = predicted_response), color = "blue") +
      geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "blue", linetype = "dotted") +
      geom_ribbon(aes(ymin = lower_pi, ymax = upper_pi), alpha = 0.1, fill = "red") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
      labs(title = "Comparison of Actual vs. Predicted response_logit (Model reg_8)",
           x = "Actual response_logit",
           y = "Predicted response_logit") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-26-1.png)

    ## Model 9

    prediction_details <- mod_reg_9 %>% predict(newdata = testing_reg, type = "response", interval = "prediction", level = 0.95)


    testing_reg$predicted_response = prediction_details[, "fit"]
    testing_reg$lower_pi = prediction_details[, "lwr"]
    testing_reg$upper_pi = prediction_details[, "upr"]


    mean_predictions <- mod_reg_9 %>%  predict( newdata = testing_reg, type = "response", interval = "confidence", level = 0.95)
    testing_reg$lower_ci = mean_predictions[, "lwr"]
    testing_reg$upper_ci = mean_predictions[, "upr"]

    ggplot(testing_reg, aes(x = response_logit, y = predicted_response)) +
      geom_point(color = 'blue', alpha = 0.8, size = 2) +
      #geom_line(aes(y = predicted_response), color = "blue") +
      geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "blue", linetype = "dotted") +
      geom_ribbon(aes(ymin = lower_pi, ymax = upper_pi), alpha = 0.1, fill = "red") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
      labs(title = "Comparison of Actual vs. Predicted response_logit (Model reg_9)",
           x = "Actual response_logit",
           y = "Predicted response_logit") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-26-2.png)

## Train/Tune with Resampling

1.  Linear models
2.  Regularized regression with Elastic net
3.  Neural network
4.  Random forest
5.  Gradient boosted tree
6.  K-nearest Neighborhoods
7.  Support Vector Regression

We adopt `RMSE` as the training metric.

    library(caret)



    # data set
    set.seed(123)


    df_train_reg <-dplyr::select(df_train, -outcome, -response)
    df_train_cla <-dplyr::select(df_train, -response_logit, -response)

    inTraining <- createDataPartition(df_train_reg$response_logit, p=.95, list = FALSE)

    training <- df_train_reg[inTraining,]
    testing  <- df_train_reg[-inTraining,]


    fitControl <- trainControl(## 10-fold CV
                               method = "repeatedcv",
                               number = 5,
                               ## repeated ten times
                               repeats = 3)

    my_metrics_regress <- 'RMSE'

    #Linear Models

    # All categorical and continuous inputs - linear additive features
    caret_reg_1_1 <- train(response_logit ~ ., 
                         data = training, 
                         method = "lm",
                         metric = my_metrics_regress,
                         trControl = fitControl)

    # Add categorical inputs to all main effect
    # and all pairwise interactions of continuous inputs

    caret_reg_1_2 <- train(response_logit ~ (R + G + B + Hue)^2 + Lightness + Saturation, 
                           data = training, 
                           method = "lm",
                           metric = my_metrics_regress,
                           trControl = fitControl)

    #mod_reg_8 <- lm(response_logit ~  (poly(R, 2) + poly(G, 2) + poly(B, 2) + poly(Hue, 2)) * (Lightness + Saturation), data = training_reg)

    caret_reg_1_3 <- train(response_logit ~ (R  + G + B + I(R^2) + I(G^2) + I(B^2) + Hue + I(Hue^2)) * 
                             (Lightness + Saturation), 
                           data = training, 
                           method = "lm",
                           metric = my_metrics_regress,
                           trControl = fitControl)

    # Can consider interactions of basis functions with other basis functions!
    #mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
    #                , data = training_reg)

    caret_reg_1_4 <- train(response_logit ~  ((R+I(R^2)) * ((B+I(B^2))+(G+I(G^2))+(Hue + I(Hue^2))) + 
                                            ((B+I(B^2)) * ((G+I(G^2))+(Hue + I(Hue^2))) +
                                                          (G+I(G^2)) *(Hue + I(Hue^2)) ) ), 
                           data = training, 
                           method = "lm",
                           metric = my_metrics_regress,
                           trControl = fitControl)


    # Regularized Regression with Elastic Net

    # Define a tuning grid
    grid_elastic <- expand.grid(alpha = seq(0, 1, length = 5), 
                                lambda = seq(0.001, 0.1, length = 5))

    # Elastic net model

    # Add categorical inputs to all main effect 
    # and all pairwise interactions of continuous inputs
    caret_reg_2_1 <- train(response_logit ~ (R + G + B + Hue)^2 + Lightness + Saturation, 
                         data = training, 
                         method = "glmnet",
                         metric = my_metrics_regress,
                         tuneGrid = grid_elastic,
                         trControl = fitControl)

    #mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
    #                , data = training_reg)


    caret_reg_2_2 <- train(response_logit ~  ((R+I(R^2)) * ((B+I(B^2))+(G+I(G^2))+(Hue + I(Hue^2))) + 
                                            ((B+I(B^2)) * ((G+I(G^2))+(Hue + I(Hue^2))) +
                                                          (G+I(G^2)) *(Hue + I(Hue^2)) ) ), 
                         data = training, 
                         method = "glmnet",
                         metric = my_metrics_regress,
                         tuneGrid = grid_elastic,
                         trControl = fitControl)



    # Neural Network 
    caret_reg_3 <- train(response_logit ~ ., 
                         data = training, 
                         method = "nnet",
                         metric = my_metrics_regress,
                         trControl = fitControl,
                         tuneLength = 5,
                         trace = FALSE)

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
    ## : There were missing values in resampled performance measures.

    # Random Forest 

    # method = "rf", not working
    #caret_reg_4 <- train(response_logit ~ ., 
    #                     data = training, 
    #                     method = "rf",
    #                     trControl = fitControl,
    #                     tuneLength = 3) 

    caret_reg_4 <- train(response_logit ~ ., 
                         data = training, 
                         method = "ranger",
                         metric = my_metrics_regress,
                         trControl = fitControl,
                         tuneLength = 3,
                         importance = 'impurity', # set feature importance to use varImp()
                         verbose = FALSE)


    # Gradient Boosted Tree
    caret_reg_5 <- train(response_logit ~ ., 
                     data = training, 
                     method = "gbm",
                     metric = my_metrics_regress,
                     trControl = fitControl,
                     ## for gbm()  passing through
                     verbose = FALSE)


    # K-nearest Neighbors
    caret_reg_6 <- train(response_logit ~ ., 
                         data = training, 
                         method = "knn",
                         metric = my_metrics_regress,
                         trControl = fitControl,
                         tuneLength = 5)



    # Support Vector Regression 
    caret_reg_7 <- train(response_logit ~ ., 
                       data = training, 
                       method = "svmRadial",  
                       metric = my_metrics_regress,
                       trControl = fitControl,
                       preProcess = c("center", "scale"),  
                       tuneLength = 5)

    # Visualization

    #caret_models <- list(caret_reg_1_1, caret_reg_1_2, caret_reg_1_3, caret_reg_1_4, caret_reg_2_1, caret_reg_2_2, 
    #caret_reg_3, caret_reg_4, caret_reg_5, caret_reg_6, caret_reg_7)
    model_list <- list(
      caret_reg_1_1 = caret_reg_1_1,
      caret_reg_1_2 = caret_reg_1_2,
      caret_reg_1_3 = caret_reg_1_3,
      caret_reg_1_4 = caret_reg_1_4,
      caret_reg_2_1 = caret_reg_2_1,
      caret_reg_2_2 = caret_reg_2_2,
      caret_reg_3 = caret_reg_3,
      caret_reg_4 = caret_reg_4,
      caret_reg_5 = caret_reg_5,
      caret_reg_6 = caret_reg_6,
      caret_reg_7 = caret_reg_7
    )


    # model 1_1 to 3
    for(model_name in names(model_list[1:7])) {
      importance <- varImp(model_list[[model_name]], 
                           scale = FALSE)
      print(
        plot(importance) 
      )
    }

![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-1.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-2.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-3.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-4.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-5.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-6.png)![](customer_data_files/figure-markdown_strict/unnamed-chunk-28-7.png)

Next, we resample and plot the metrics for each model.

    resamps <- resamples(list(
      lm_1_1 = caret_reg_1_1,
      lm_1_2 = caret_reg_1_2,
      lm_1_3 = caret_reg_1_3,
      lm_1_4 = caret_reg_1_4,
      glmnet_2_1 = caret_reg_2_1,
      glmnet_2_2 = caret_reg_2_2,
      nnt_3 = caret_reg_3,
      ranger_4 = caret_reg_4,
      gbt_5 = caret_reg_5,
      knn_6 = caret_reg_6,
      svr_7 = caret_reg_7))
    resamps %>% summary()

    ## 
    ## Call:
    ## summary.resamples(object = .)
    ## 
    ## Models: lm_1_1, lm_1_2, lm_1_3, lm_1_4, glmnet_2_1, glmnet_2_2, nnt_3, ranger_4, gbt_5, knn_6, svr_7 
    ## Number of resamples: 15 
    ## 
    ## MAE 
    ##                  Min.    1st Qu.     Median       Mean    3rd Qu.       Max.
    ## lm_1_1     0.05806626 0.06438969 0.06725898 0.06807852 0.07081770 0.08077368
    ## lm_1_2     0.05564725 0.06108733 0.06209633 0.06158427 0.06257796 0.06648236
    ## lm_1_3     0.02982825 0.03258383 0.03368507 0.03400997 0.03431253 0.03942709
    ## lm_1_4     0.02968733 0.03055804 0.03159660 0.03178961 0.03233564 0.03572733
    ## glmnet_2_1 0.05638612 0.05945262 0.06162832 0.06176687 0.06356698 0.06829837
    ## glmnet_2_2 0.06685890 0.06773158 0.06880529 0.06944746 0.07047365 0.07581290
    ## nnt_3      0.63730807 0.66556983 0.68843882 0.71473045 0.71665954 0.98829904
    ## ranger_4   0.04688552 0.05015076 0.05251270 0.05265489 0.05404577 0.06378243
    ## gbt_5      0.04316156 0.04489893 0.04665720 0.04641829 0.04760618 0.04984199
    ## knn_6      0.06701050 0.06837998 0.07105753 0.07239341 0.07629685 0.08060663
    ## svr_7      0.06679633 0.07169684 0.07439665 0.07453010 0.07816439 0.08085337
    ##            NA's
    ## lm_1_1        0
    ## lm_1_2        0
    ## lm_1_3        0
    ## lm_1_4        0
    ## glmnet_2_1    0
    ## glmnet_2_2    0
    ## nnt_3         0
    ## ranger_4      0
    ## gbt_5         0
    ## knn_6         0
    ## svr_7         0
    ## 
    ## RMSE 
    ##                  Min.    1st Qu.     Median       Mean    3rd Qu.       Max.
    ## lm_1_1     0.07581445 0.08599892 0.08795844 0.09050847 0.09277275 0.10725068
    ## lm_1_2     0.07427979 0.07911260 0.08222118 0.08206695 0.08484550 0.08899651
    ## lm_1_3     0.04253855 0.04433830 0.04494631 0.04781227 0.05154214 0.05673478
    ## lm_1_4     0.04073857 0.04120761 0.04330660 0.04464872 0.04781860 0.05145766
    ## glmnet_2_1 0.07475547 0.07828631 0.08251873 0.08189443 0.08519720 0.08902707
    ## glmnet_2_2 0.08726622 0.08864556 0.09210902 0.09229616 0.09497490 0.10201795
    ## nnt_3      0.93045623 0.95377550 0.98716224 0.99430278 1.01974637 1.12891947
    ## ranger_4   0.06251191 0.06726356 0.07013374 0.07179667 0.07579434 0.08994634
    ## gbt_5      0.05481927 0.05951331 0.06171524 0.06137108 0.06371494 0.06654584
    ## knn_6      0.08867917 0.09241799 0.09981116 0.09856004 0.10404982 0.10836221
    ## svr_7      0.08191410 0.09021325 0.09535991 0.09366186 0.09707698 0.10275407
    ##            NA's
    ## lm_1_1        0
    ## lm_1_2        0
    ## lm_1_3        0
    ## lm_1_4        0
    ## glmnet_2_1    0
    ## glmnet_2_2    0
    ## nnt_3         0
    ## ranger_4      0
    ## gbt_5         0
    ## knn_6         0
    ## svr_7         0
    ## 
    ## Rsquared 
    ##                 Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lm_1_1     0.9917342 0.9938802 0.9944880 0.9942055 0.9948599 0.9960823    0
    ## lm_1_2     0.9948996 0.9950147 0.9952614 0.9953456 0.9955565 0.9961846    0
    ## lm_1_3     0.9978881 0.9981421 0.9985201 0.9983998 0.9986166 0.9987362    0
    ## lm_1_4     0.9982300 0.9983658 0.9987299 0.9985971 0.9987843 0.9988389    0
    ## glmnet_2_1 0.9941531 0.9949260 0.9954180 0.9952726 0.9955761 0.9960390    0
    ## glmnet_2_2 0.9933965 0.9938345 0.9942624 0.9940908 0.9943086 0.9946389    0
    ## nnt_3      0.2256897 0.5950161 0.6653017 0.6270525 0.7126365 0.7587802    0
    ## ranger_4   0.9951129 0.9961549 0.9965652 0.9965004 0.9970054 0.9973347    0
    ## gbt_5      0.9970014 0.9972445 0.9973852 0.9973786 0.9975286 0.9977399    0
    ## knn_6      0.9912013 0.9925797 0.9931658 0.9931739 0.9936758 0.9949106    0
    ## svr_7      0.9928057 0.9938766 0.9941496 0.9941686 0.9945466 0.9953933    0

    #trellis.par.set(caretTheme())
    dotplot(resamps)

![](customer_data_files/figure-markdown_strict/unnamed-chunk-29-1.png)

Apparently, the best performers are our linear model 9 and 8 trained
from `lm()`.

# Classification

## General Linear Model

-   intercept-only model – no INPUTS!

-   Categorical variables only – linear additive

-   Continuous variables only – linear additive

-   All categorical and continuous variables – linear additive

-   Interaction of the categorical inputs with all continuous inputs
    main effects

-   Add categorical inputs to all main effect and all pairwise
    interactions of continuous inputs

-   Interaction of the categorical inputs with all main effect and all
    pairwise interactions of continuous inputs

-   3 models with basis functions of your choice

We will answer the following questions accordingly: Which of the 10
models is the best? What performance metric did you use to make your
selection? Visualize the coefficient summaries for your top 3 models.
Which inputs seem important?

    set.seed(123)
    inTraining <- createDataPartition(df_train$outcome, p=.95, list = FALSE)
    training_cla <- df_train[inTraining,]
    testing_cla  <- df_train[-inTraining,]



    # Intercept-only model – no INPUTS!
    mod_cla_1 <- glm(outcome ~ 1, data = training_cla, family = binomial)


    # Categorical variables only – linear additive
    mod_cla_2 <- glm(outcome ~ Lightness + Saturation, data = training_cla, family = binomial)

    # Continuous variables only – linear additive
    mod_cla_3 <- glm(outcome ~ R + G + B + Hue, data = training_cla, family = binomial)

    # All categorical and continuous variables – linear additive
    mod_cla_4 <- glm(outcome ~ Lightness + Saturation + R + G + B + Hue, data = training_cla, family = binomial)

    # Interaction of the categorical inputs with all continuous inputs main effects
    mod_cla_5 <- glm(outcome ~ (Lightness + Saturation) : (R + G + B + Hue), data = training_cla, family = binomial)

    # Add categorical inputs to all main effect and all pairwise interactions of continuous inputs
    mod_cla_6 <- glm(outcome ~ Lightness + Saturation + R + G + B + Hue + 
                    R:G + R:B + R:Hue + G:B + G:Hue + B:Hue, data = training_cla, family = binomial)

    # Interaction of the categorical inputs with all main effect and all pairwise interactions of continuous inputs
    mod_cla_7 <- glm(outcome ~ (Lightness + Saturation) *
                    (R + G + B + Hue + R:G + R:B + R:Hue ),
                    data = training_cla, family = binomial)


    # Try non-linear basis functions based on your EDA
    mod_cla_8 <- glm(outcome ~ B + R + G + Hue +
                       (B+R+G) : ( poly(Hue, 2) + Lightness + Saturation) , data = training_cla, family = binomial)

    # Can consider interactions of basis functions with other basis functions!

    mod_cla_9 <- glm(outcome ~ Lightness + Saturation + Lightness:(R + G + Hue) + Saturation:(B+ G) + B:poly(Hue, 2) ,
                     data = training_cla, family = binomial)




    # Can consider interactions of basis functions with the categorical inputs!
    mod_cla_10 <- glm(outcome ~  R : (G+B + poly(Hue, 2) + Lightness + Saturation) + 
                                    G : (B + poly(Hue, 2) + Lightness + Saturation) + 
                                    B : ((poly(Hue, 2)) + Lightness + Saturation)
                    , data = training_cla, family = binomial)

For Model 1 to 7, they are the same models from the regression problem.
Some model like mod\_cla\_7, had warning:
`fitted probabilities numerically 0 or 1 occurred`.

We now evaluate our general linear models with AIC, BIC metrics.

    library(broom)

    ## 
    ## Attaching package: 'broom'

    ## The following object is masked from 'package:modelr':
    ## 
    ##     bootstrap

    model_list <- list(
      mod_cla_1, mod_cla_2, mod_cla_3, mod_cla_4, mod_cla_5, 
      mod_cla_6, mod_cla_7, mod_cla_8, mod_cla_9, mod_cla_10
    )
    names(model_list) <- paste("Model", 1:10)

    model_summaries <- lapply(names(model_list), function(model_name) {
      model_summary <- glance(model_list[[model_name]])
      model_summary$model <- model_name
      return(model_summary)
    })


    model_summaries_df <- bind_rows(model_summaries)
    model_summaries_df$model <- factor(model_summaries_df$model,
                                       levels = paste("Model", 1:10))

    # Plotting
    model_summaries_df %>%  ggplot( aes(x = model)) +
      geom_point(aes(y = AIC, color = "AIC")) + 
      geom_line(aes(y = AIC, group = 1, color = "AIC")) + 
      geom_point(aes(y = BIC, color = "BIC")) + 
      geom_line(aes(y = BIC, group = 1, color = "BIC")) +
      theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-31-1.png)

    model_summaries_df %>%  arrange(AIC) %>% print()

    ## # A tibble: 10 × 9
    ##    null.deviance df.null logLik   AIC   BIC deviance df.residual  nobs model   
    ##            <dbl>   <int>  <dbl> <dbl> <dbl>    <dbl>       <int> <int> <fct>   
    ##  1          862.     793  -249.  597.  826.     499.         745   794 Model 9 
    ##  2          862.     793  -257.  608.  828.     514.         747   794 Model 8 
    ##  3          862.     793  -260.  612.  827.     520.         748   794 Model 10
    ##  4          862.     793  -205.  617. 1104.     409.         690   794 Model 7 
    ##  5          862.     793  -300.  646.  754.     600.         771   794 Model 6 
    ##  6          862.     793  -293.  691.  939.     585.         741   794 Model 5 
    ##  7          862.     793  -333.  700.  780.     666.         777   794 Model 4 
    ##  8          862.     793  -341.  708.  769.     682.         781   794 Model 2 
    ##  9          862.     793  -418.  845.  868.     835.         789   794 Model 3 
    ## 10          862.     793  -431.  864.  869.     862.         793   794 Model 1

    model_summaries_df %>%  arrange(BIC) %>% print()

    ## # A tibble: 10 × 9
    ##    null.deviance df.null logLik   AIC   BIC deviance df.residual  nobs model   
    ##            <dbl>   <int>  <dbl> <dbl> <dbl>    <dbl>       <int> <int> <fct>   
    ##  1          862.     793  -300.  646.  754.     600.         771   794 Model 6 
    ##  2          862.     793  -341.  708.  769.     682.         781   794 Model 2 
    ##  3          862.     793  -333.  700.  780.     666.         777   794 Model 4 
    ##  4          862.     793  -249.  597.  826.     499.         745   794 Model 9 
    ##  5          862.     793  -260.  612.  827.     520.         748   794 Model 10
    ##  6          862.     793  -257.  608.  828.     514.         747   794 Model 8 
    ##  7          862.     793  -418.  845.  868.     835.         789   794 Model 3 
    ##  8          862.     793  -431.  864.  869.     862.         793   794 Model 1 
    ##  9          862.     793  -293.  691.  939.     585.         741   794 Model 5 
    ## 10          862.     793  -205.  617. 1104.     409.         690   794 Model 7

From AIC metric, Model 9, 8 and 10 are the top 3 performers.

Next, we make an attempt to plot the coefficients of our models.

    # Plotting Coefficients

    #coefplot(mod_cla_10, intercept = TRUE)  + 
    #  labs(title = 'Coefficient Plot for Model 10')
     
    coefplot(mod_cla_9, intercept = TRUE)  + 
      labs(title = 'Coefficient Plot for Model 9')

![](customer_data_files/figure-markdown_strict/unnamed-chunk-32-1.png)

    coefplot(mod_cla_8, intercept = TRUE)  + 
      labs(title = 'Coefficient Plot for Model 8')

![](customer_data_files/figure-markdown_strict/unnamed-chunk-32-2.png)
From `coefplot::coefplot`, the graphs are unreadable due to the
complexity of models. We do the following to simplify.

    library(dplyr)


    # mod 10
    coef_10 <- summary(mod_cla_10)$coefficients %>%
      as.data.frame() %>%  
      rownames_to_column(var = "Term") %>%  
      as_tibble() %>% 
      mutate(
        lower_bound = Estimate - 1.96 * `Std. Error`,
        upper_bound = Estimate + 1.96 * `Std. Error` 
      ) %>%
      dplyr::select(Term, Estimate, Std.Error = `Std. Error`, lower_bound, upper_bound) 


    significant_coefs_10 <- coef_10 %>%
      filter(lower_bound > 0 | upper_bound < 0) 

    significant_coefs_10 %>%  ggplot( aes(x = Term, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 10", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-33-1.png)

    # mod 9

    coef_9 <- summary(mod_cla_9)$coefficients %>%
      as.data.frame() %>%  
      rownames_to_column(var = "Term") %>%  
      as_tibble() %>% 
      mutate(
        lower_bound = Estimate - 1.96 * `Std. Error`,
        upper_bound = Estimate + 1.96 * `Std. Error` 
      ) %>%
      dplyr::select(Term, Estimate, Std.Error = `Std. Error`, lower_bound, upper_bound) 


    significant_coefs_9 <- coef_9 %>%
      filter(lower_bound > 0 | upper_bound < 0) 

    significant_coefs_9 %>%  ggplot( aes(x = Term, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 9", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-33-2.png)

    # mod 8

    coef_8 <- summary(mod_cla_8)$coefficients %>%
      as.data.frame() %>%  
      rownames_to_column(var = "Term") %>%  
      as_tibble() %>% 
      mutate(
        lower_bound = Estimate - 1.96 * `Std. Error`,
        upper_bound = Estimate + 1.96 * `Std. Error` 
      ) %>%
      dplyr::select(Term, Estimate, Std.Error = `Std. Error`, lower_bound, upper_bound) 


    significant_coefs_8 <- coef_8 %>%
      filter(lower_bound > 0 | upper_bound < 0) 

    significant_coefs_8 %>%  ggplot( aes(x = Term, y = Estimate, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Model 8", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-33-3.png)
Throughout the 3 models, we notice that `lightness` and its interaction
with other variable is extremely essential.

## Baysian GLM

We use the Laplace Approximation approach and define the following
function to apply Bayesian methods. We will use general linear models 9
and 8. Model 9 is the best performer from AIC and BIC; model 8 is the
second best performer constructed from EDA.

    logistic_logpost <- function(unknowns, my_info)
    {
      # extract the design matrix and assign to X
      X <- my_info$design_matrix
      
      # calculate the linear predictor
      eta <- X %*% unknowns
      
      # calculate the event probability
      mu <- 1 / (1 + exp(-eta))
      
      # evaluate the log-likelihood
      y <- my_info$yobs
      log_lik <- sum(y * log(mu) + (1 - y) * log(1 - mu))
       
      
      # evaluate the log-prior
      mu_beta <- my_info$mu_beta
      tau_beta <- my_info$tau_beta
      log_prior <- sum(dnorm(unknowns, mean=mu_beta,
                             sd=tau_beta, log=TRUE))
      
      # sum together
      log_post <- log_lik + log_prior
      return(log_post)
    }

    my_laplace <- function(start_guess, logpost_func, ...)
    {
      # code adapted from the `LearnBayes`` function `laplace()`
      fit <- optim(start_guess,
                   logpost_func,
                   gr = NULL,
                   ...,
                   method = "BFGS",
                   hessian = TRUE,
                   control = list(fnscale = -1, maxit = 5001))
      
      mode <- fit$par
      post_var_matrix <- -solve(fit$hessian)
      p <- length(mode)
      int <- p/2 * log(2 * pi) + 0.5 * log(det(post_var_matrix)) + logpost_func(mode, ...)
      # package all of the results into a list
      list(mode = mode,
           var_matrix = post_var_matrix,
           log_evidence = int,
           converge = ifelse(fit$convergence == 0,
                             "YES", 
                             "NO"),
           iter_counts = as.numeric(fit$counts[1]))
    }

    #mod_cla_9 <- glm(outcome ~ Lightness + Saturation + Lightness:(R + G + Hue) + Saturation:(B+ G) + B: (Hue + Hue2) ,
    #                 data = training_cla, family = binomial)



    #df_train_B


    Xmat_9 <- model.matrix(~ Lightness + Saturation+ Lightness:(R + G + Hue) + Saturation:(B+ G)+  (Hue+Hue2),#:B,
                           data=df_train_B)

    info_9 <- list(
      yobs = df_train_B$outcome,
      design_matrix = Xmat_9,
      mu_beta = 0,
      tau_beta = 4.5
    )

    initial_guess_9 <- rep(0, ncol(Xmat_9))

    laplace_9 <- my_laplace(initial_guess_9, logistic_logpost, my_info=info_9)




    #mod_cla_8 <- glm(outcome ~ B + R + G + Hue 
    #                 + (B+R+G) : ( Hue + Hue2 + Lightness + Saturation) , 
    #                 data = training_cla, family = binomial)


    # compute Laplace
    Xmat_8 <- model.matrix(~  B + R + G + Hue + 
                          Hue2  + #(B+R+G) : ( Hue + Hue2) + 
                             (B+R+G):(Lightness + Saturation),
                           data=df_train_B)

    info_8 <- list(
      yobs = df_train_B$outcome,
      design_matrix = Xmat_8,
      mu_beta = 0,
      tau_beta = 4.5
    )

    initial_guess_8 <- rep(0, ncol(Xmat_8))

    laplace_8 <- my_laplace(initial_guess_8, logistic_logpost, my_info=info_8)

    evidences <- c(laplace_8$log_evidence,
                   laplace_9$log_evidence)


    weights_unnormalized <- exp(evidences - max(evidences))
    weights <- weights_unnormalized / sum(weights_unnormalized)


    model_weights_df <- data.frame(
      Model = c('Bayesian Model 8', 'Bayesian Model 9'),
      Weight = weights
    )


    ggplot(model_weights_df, aes(x = Model, y = Weight)) +
      geom_bar(stat = "identity") +
      theme_minimal() 

![](customer_data_files/figure-markdown_strict/unnamed-chunk-35-1.png)
From the weight of models, Model 9 is the best.

To plot the significant coefficients:

    # Model 9


    std_errors <- sqrt(diag(laplace_9$var_matrix))

    ci_lower <- laplace_9$mode - 1.96 * std_errors
    ci_upper <- laplace_9$mode + 1.96 * std_errors

    ci_bounds <- cbind(lower_bound = ci_lower, upper_bound = ci_upper,  mode = laplace_9$mode)

    coef_9_B <- ci_bounds %>%
      as.data.frame() %>%  
      rownames_to_column() %>%  
      as_tibble() 


    significant_coefs_9_B <- coef_9_B %>%
      filter(lower_bound > 0 | upper_bound < 0) 

    significant_coefs_9_B %>%  ggplot( aes(x = rowname, y = mode, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Baysian Model 9 ", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-36-1.png)

    # Model 8



    std_errors <- sqrt(diag(laplace_8$var_matrix))

    ci_lower <- laplace_8$mode - 1.96 * std_errors
    ci_upper <- laplace_8$mode + 1.96 * std_errors

    ci_bounds <- cbind(lower_bound = ci_lower, upper_bound = ci_upper,  mode = laplace_8$mode)

    coef_8_B <- ci_bounds %>%
      as.data.frame() %>%  
      rownames_to_column() %>%  
      as_tibble() 


    significant_coefs_8_B <- coef_8_B %>%
      filter(lower_bound > 0 | upper_bound < 0) 

    significant_coefs_8_B %>%  ggplot( aes(x = rowname, y = mode, ymin = lower_bound, ymax = upper_bound)) +
      geom_pointrange() +  
      coord_flip() +  
      labs(title = "Significant Coefficients Baysian Model 8 ", x = "Term", y = "Estimate") +
      theme_minimal() +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red")  

![](customer_data_files/figure-markdown_strict/unnamed-chunk-36-2.png)

## GLM Predictions

Now we make direct prediction from the best 2 models, model 8 and 9.

    set.seed(123)
    inTraining <- createDataPartition(df_train$outcome, p=.95, list = FALSE)
    training_cla <- df_train[inTraining,]
    testing_cla  <- df_train[-inTraining,]




    # Model 8
    testing_cla$pred_proba_8 <- predict(mod_cla_8, newdata = testing_cla, type = "response")


    ggplot(testing_cla, aes(x = pred_proba_8, fill = as.factor(outcome))) +
      geom_histogram(data = testing_cla, binwidth = 0.2, alpha = 0.5, position = "fill") +
        #facet_wrap(~Lightness) +
      labs(title = "mod_cla_8",
           x = "Predicted Probability",
           y = "Percentage",
           fill = "Actual Outcome") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-37-1.png)

    # Plot for model 9


    testing_cla$pred_proba_9 <- predict(mod_cla_9, newdata = testing_cla, type = "response")


    ggplot(testing_cla, aes(x = pred_proba_9, fill = as.factor(outcome))) +
      geom_histogram(data = testing_cla, binwidth = 0.25, alpha = 0.5, position = "fill") +
      labs(title = "mod_cla_9",
           x = "Predicted Probability",
           y = "Percentage",
           fill = "Actual Outcome") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/unnamed-chunk-37-2.png)

We also can find their ROC value.

    library(pROC)

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

    # model 8
    roc_response_8 <- roc(testing_cla$outcome, testing_cla$pred_proba_8)

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    plot(roc_response_8, main="ROC for mod_cla_8")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-38-1.png)

    # model 9
    roc_response_9 <- roc(testing_cla$outcome, testing_cla$pred_proba_9)

    ## Setting levels: control = 0, case = 1
    ## Setting direction: controls < cases

    plot(roc_response_9, main="ROC for mod_cla_9")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-38-2.png)

    print("model 8")

    ## [1] "model 8"

    auc(roc_response_8)

    ## Area under the curve: 0.9095

    print("model 9")

    ## [1] "model 9"

    auc(roc_response_9)

    ## Area under the curve: 0.8905

## Train/Tune with resampling

We now use `caret` to train more models.

    library(caret)



    # data set
    set.seed(123)


    df_train_reg <- dplyr::select(df_train, -outcome, -response)
    df_train_cla <- dplyr::select(df_train, -response_logit, -response) %>% 
      mutate(outcome = ifelse(outcome == 1, 'event', 'non_event'),
             outcome = factor(outcome, levels = c('event', 'non_event')))

    inTraining <- createDataPartition(df_train_cla$outcome, p=.9, list = FALSE)


    training <- df_train_cla[inTraining,]

    testing  <- df_train_cla[-inTraining,]


    fitControl <- trainControl(## 5-fold CV
                               method = "repeatedcv",
                               number = 10,
                               ## repeated 3 times
                               repeats = 5,
                               classProbs = TRUE
                               )


    #my_binary_method <- 'glm'
    #my_preProcess <- c('center', 'scale')
    #mod_binary_acc <- train(outcome ~ Hue + Lightness,
    #                        data = dfiiiD,
    #                        method = my_binary_method,
    #                        preProcess = c('center', 'scale'),
    #                        metric = my_metrics_acc,
    #                        trControl = my_ctrl_acc)

    my_metrics_cla <- 'Accuracy'
    my_preProcess <- c('center', 'scale')


    #Linear Models

    # All categorical and continuous inputs - linear additive features
    caret_cla_1_1 <- train(outcome ~ ., 
                         data = training, 
                         method = "glm",
                         metric = my_metrics_cla,
                         #my_preProcess = c('center', 'scale'),
                         trControl = fitControl)

    #caret_cla_1_1 %>% readr::write_rds("model_caret_cla_1_1.rds")
    #caret_cla_1_1 <- readr::read_rds("model_caret_cla_1_1.rds")

    # Add categorical inputs to all main effect
    # and all pairwise interactions of continuous inputs

    caret_cla_1_2 <- train(outcome ~ (R + G + B + Hue)^2 + Lightness + Saturation, 
                           data = training, 
                           method = "glm",
                           metric = my_metrics_cla,
                           trControl = fitControl)



    #mod_cla_8 <- glm(outcome ~ B + R + G + Hue +
    #                   (B+R+G) : ( poly(Hue, 2) + Lightness + Saturation), data = training_cla, family = binomial)



    caret_cla_1_3 <- train(outcome ~ B + R + G + Hue +
                       (B+R+G) : ( Hue + I(Hue^2) + Lightness + Saturation), 
                           data = training, 
                           method = "glm",
                           metric = my_metrics_cla,
                           trControl = fitControl)

    #mod_cla_10 <- glm(outcome ~  R * (G+B + poly(Hue, 2) + Lightness + Saturation) + 
    #                                G * (B + poly(Hue, 2) + Lightness + Saturation) + 
    #                                B *( (poly(Hue, 2)) + Lightness + Saturation)
    #                , data = training_cla, family = binomial)             , data = training_reg)

    caret_cla_1_4 <- train(outcome ~  R * (G+B + Hue + I(Hue^2) + Lightness + Saturation) + 
                                    G * (B + Hue + I(Hue^2)+ Lightness + Saturation) + 
                                    B *( Hue + I(Hue^2) + Lightness + Saturation),
                           data = training, 
                           method = "glm",
                           metric = my_metrics_cla,
                           trControl = fitControl)







    # Regularized Regression with Elastic Net

    # Define a tuning grid
    grid_elastic <- expand.grid(alpha = seq(0, 1, length = 5), 
                                lambda = seq(0.001, 0.1, length = 5))

    # Elastic net model

    # Add categorical inputs to all main effect 
    # and all pairwise interactions of continuous inputs
    caret_cla_2_1 <- train(outcome ~ (R + G + B + Hue)^2 + Lightness + Saturation, 
                         data = training, 
                         method = "glmnet",
                         metric = my_metrics_cla,
                         tuneGrid = grid_elastic,
                         trControl = fitControl)

    #mod_reg_9 <- lm(response_logit ~  (poly(R, 2) * poly(G, 2) + poly(B, 2) * poly(Hue, 2) + poly(R, 2) * poly(Hue, 2) + poly(B, 2) * poly(R, 2)  + poly(G, 2) * poly(Hue, 2) + poly(B, 2) * poly(G, 2))
    #                , data = training_reg)


    caret_cla_2_2 <- train(outcome ~ R * (G+B + Hue + I(Hue^2) + Lightness + Saturation) + 
                                    G * (B + Hue + I(Hue^2)+ Lightness + Saturation) + 
                                    B *( Hue + I(Hue^2) + Lightness + Saturation),
                         data = training, 
                         method = "glmnet",
                         metric = my_metrics_cla,
                         tuneGrid = grid_elastic,
                         trControl = fitControl)







    # Neural Network 
    caret_cla_3 <- train(outcome ~ ., 
                         data = training, 
                         method = "nnet",
                         metric = my_metrics_cla,
                         trControl = fitControl,
                         tuneLength = 5,
                         trace = FALSE)


    # Random Forest 

    # method = "rf", not working
    #caret_reg_4 <- train(response_logit ~ ., 
    #                     data = training, 
    #                     method = "rf",
    #                     trControl = fitControl,
    #                     tuneLength = 3) 
    #
    #training <- df_train_cla
    #fitControl <- trainControl(## 5-fold CV
    #                           method = "repeatedcv",
    #                           number = 10,
    #                           ## repeated 7 times
    #                           repeats = 7,
    #                           classProbs = TRUE
    #                           )


    caret_cla_4 <- train(outcome ~ ., 
                         data = training, 
                         method = "ranger",
                         metric = my_metrics_cla,
                         trControl = fitControl,
                         tuneLength = 7,
                         importance = 'impurity', # set feature importance to use varImp()
                         verbose = FALSE)


    # Gradient Boosted Tree


    caret_cla_5 <- train(outcome ~ ., 
                     data = training, 
                     method = "gbm",
                     metric = my_metrics_cla,
                     trControl = fitControl,
                     ## for gbm()  passing through
                     verbose = FALSE)


    # K-nearest Neighbors
    caret_cla_6 <- train(outcome ~ ., 
                         data = training, 
                         method = "knn",
                         metric = my_metrics_cla,
                         trControl = fitControl,
                         tuneLength = 5)



    # Support Vector Regression 
    caret_cla_7 <- train(outcome ~ ., 
                       data = training, 
                       method = "svmRadial",  
                       metric = my_metrics_cla,
                       trControl = fitControl,
                       preProcess = c("center", "scale"),  
                       tuneLength = 5)

    # model management

    caret_cla_1_1 <- readr::read_rds("model_caret_cla_1_1.rds")
    caret_cla_1_2 <- readr::read_rds("model_caret_cla_1_2.rds")
    caret_cla_1_3 <- readr::read_rds("model_caret_cla_1_3.rds")
    caret_cla_1_4 <- readr::read_rds("model_caret_cla_1_4.rds")
    caret_cla_2_1 <- readr::read_rds("model_caret_cla_2_1.rds")
    caret_cla_2_2 <- readr::read_rds("model_caret_cla_2_2.rds")
    caret_cla_3 <- readr::read_rds("model_caret_cla_3.rds")
    caret_cla_4 <- readr::read_rds("model_caret_cla_4.rds")
    caret_cla_5 <- readr::read_rds("model_caret_cla_5.rds")
    caret_cla_6 <- readr::read_rds("model_caret_cla_6.rds")
    caret_cla_7 <- readr::read_rds("model_caret_cla_7.rds")




    #caret_cla_1_1 %>% readr::write_rds("model_caret_cla_1_1.rds")
    #caret_cla_1_2 %>% readr::write_rds("model_caret_cla_1_2.rds")
    #caret_cla_1_3 %>% readr::write_rds("model_caret_cla_1_3.rds")
    #caret_cla_1_4 %>% readr::write_rds("model_caret_cla_1_4.rds")
    #caret_cla_2_1 %>% readr::write_rds("model_caret_cla_2_1.rds")
    #caret_cla_2_2 %>% readr::write_rds("model_caret_cla_2_2.rds")
    #caret_cla_3 %>% readr::write_rds("model_caret_cla_3.rds")
    #caret_cla_4 %>% readr::write_rds("model_caret_cla_4.rds")
    #caret_cla_5 %>% readr::write_rds("model_caret_cla_5.rds")
    #caret_cla_4 %>% readr::write_rds("model_caret_cla_4_enhanced.rds")
    #caret_cla_5 %>% readr::write_rds("model_caret_cla_5_enhanced.rds")
    #caret_cla_6 %>% readr::write_rds("model_caret_cla_6.rds")
    #caret_cla_7 %>% readr::write_rds("model_caret_cla_7.rds")

We perform resamplig and compare the models above via different metrics.

    resamps_cla <- resamples(list(
      glm_1_1 = caret_cla_1_1,
      glm_1_2 = caret_cla_1_2,
      glm_1_3 = caret_cla_1_3,
      glm_1_4 = caret_cla_1_4,
      glmnet_2_1 = caret_cla_2_1,
      glmnet_2_2 = caret_cla_2_2,
      nnt_3 = caret_cla_3,
      ranger_4 = caret_cla_4,
      gbt_5 = caret_cla_5,
      knn_6 = caret_cla_6,
      svr_7 = caret_cla_7))


    resamps_cla %>% summary()

    ## 
    ## Call:
    ## summary.resamples(object = .)
    ## 
    ## Models: glm_1_1, glm_1_2, glm_1_3, glm_1_4, glmnet_2_1, glmnet_2_2, nnt_3, ranger_4, gbt_5, knn_6, svr_7 
    ## Number of resamples: 50 
    ## 
    ## Accuracy 
    ##                 Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## glm_1_1    0.7333333 0.8000000 0.8145614 0.8183333 0.8400000 0.8800000    0
    ## glm_1_2    0.7066667 0.7873684 0.8133333 0.8053158 0.8266667 0.8666667    0
    ## glm_1_3    0.7500000 0.8000000 0.8266667 0.8239719 0.8533333 0.8947368    0
    ## glm_1_4    0.7466667 0.7921053 0.8266667 0.8220737 0.8533333 0.9066667    0
    ## glmnet_2_1 0.7600000 0.8000000 0.8133333 0.8210316 0.8400000 0.8800000    0
    ## glmnet_2_2 0.7600000 0.8000000 0.8133333 0.8188982 0.8400000 0.8800000    0
    ## nnt_3      0.7200000 0.7900000 0.8344737 0.8247719 0.8547807 0.9333333    0
    ## ranger_4   0.7600000 0.8317105 0.8533333 0.8518807 0.8800000 0.9200000    0
    ## gbt_5      0.7866667 0.8272368 0.8533333 0.8479018 0.8666667 0.8933333    0
    ## knn_6      0.7105263 0.7633333 0.8000000 0.7981789 0.8283772 0.8800000    0
    ## svr_7      0.7466667 0.8000000 0.8212281 0.8191368 0.8400000 0.9078947    0
    ## 
    ## Kappa 
    ##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## glm_1_1    -0.01351351 0.2707945 0.3613139 0.3565816 0.4525547 0.5994065    0
    ## glm_1_2     0.01746725 0.3012527 0.3696682 0.3566029 0.4415423 0.6196755    0
    ## glm_1_3     0.23175416 0.3746230 0.4756255 0.4575024 0.5453115 0.6972112    0
    ## glm_1_4     0.26204039 0.3654638 0.4701168 0.4657039 0.5588137 0.7391952    0
    ## glmnet_2_1  0.08783784 0.2924531 0.3613139 0.3630166 0.4305808 0.5789145    0
    ## glmnet_2_2  0.03433476 0.2905405 0.3330467 0.3515928 0.4525547 0.5994065    0
    ## nnt_3      -0.01351351 0.3177193 0.4882653 0.4332642 0.5563558 0.7971877    0
    ## ranger_4    0.34274586 0.4736897 0.5555081 0.5473059 0.6142911 0.7511062    0
    ## gbt_5       0.33234421 0.4493017 0.5103858 0.5112543 0.5789145 0.6681416    0
    ## knn_6       0.15430267 0.2981909 0.3914444 0.3917468 0.4780961 0.6504402    0
    ## svr_7       0.06311637 0.2603550 0.3508181 0.3509533 0.4525547 0.7057522    0

    #trellis.par.set(caretTheme())
    dotplot(resamps_cla)

![](customer_data_files/figure-markdown_strict/unnamed-chunk-41-1.png)
By Accuracy, Random forest and Gradient boosted tree have the best
performance.

To verify the effects of the top performers, we plot the prediction on
testing set.

    # Plot the prediction

    df_train_reg <- dplyr::select(df_train, -outcome, -response)
    df_train_cla <- dplyr::select(df_train, -response_logit, -response) %>% 
      mutate(outcome = ifelse(outcome == 1, 'event', 'non_event'),
             outcome = factor(outcome, levels = c('event', 'non_event')))

    inTraining <- createDataPartition(df_train_cla$outcome, p=.9, list = FALSE)

    training <- df_train_cla[inTraining,]
    testing  <- df_train_cla[-inTraining,]

    # m4
    testing_cla$pred_proba_4 <- predict(caret_cla_4, newdata = testing_cla, type = "prob")[, "event"]


    ggplot(testing_cla, aes(x = pred_proba_4, fill = as.factor(outcome))) +
      geom_histogram(binwidth = 0.1, alpha = 0.5, position = "fill") +  # Adjusted binwidth if needed
      labs(title = "Predicted Probability Distribution for Model caret_cla_4",
           x = "Predicted Probability of Event",
           y = "Percentage",
           fill = "Actual Outcome") +
      theme_minimal()

    ## Warning: Removed 8 rows containing missing values (`geom_bar()`).

![](customer_data_files/figure-markdown_strict/caret%20classification%20prediction-1.png)

    # m5

    testing_cla$pred_proba_5 <- predict(caret_cla_5, newdata = testing_cla, type = "prob")[, "event"]

    ggplot(testing_cla, aes(x = pred_proba_5, fill = as.factor(outcome))) +
      geom_histogram(binwidth = 0.1, alpha = 0.5, position = "fill") +  # Adjusted binwidth if needed
      labs(title = "Predicted Probability Distribution for Model caret_cla_5",
           x = "Predicted Probability of Event",
           y = "Percentage",
           fill = "Actual Outcome") +
      theme_minimal()

![](customer_data_files/figure-markdown_strict/caret%20classification%20prediction-2.png)

    #Density Plots
    #densityplot(caret_cla_4, pch = "|", main="Model 4")
    #densityplot(caret_cla_5, pch = "|", main="Model 5")

In the meantime, we plot the ROC.

    library(pROC)

    set.seed(123)
    inTraining <- createDataPartition(df_train$outcome, p=.95, list = FALSE)
    training_cla <- df_train[inTraining,]
    testing_cla  <- df_train[-inTraining,]


    # Model 4
    testing_cla$pred_proba_4 <- predict(caret_cla_4, newdata = testing_cla, type = "prob")[,"event"]

    roc_response_4 <- roc(testing_cla$outcome, testing_cla$pred_proba_4)

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    plot(roc_response_4, main="ROC for caret_cla_4 ranger")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-42-1.png)

    # model 5
    testing_cla$pred_proba_5 <- predict(caret_cla_5, newdata = testing_cla, type = "prob")[,"event"]

    roc_response_5 <- roc(testing_cla$outcome, testing_cla$pred_proba_5)

    ## Setting levels: control = 0, case = 1
    ## Setting direction: controls < cases

    plot(roc_response_5, main="ROC for caret_cla_5 gbm")

![](customer_data_files/figure-markdown_strict/unnamed-chunk-42-2.png)

    print("model 4")

    ## [1] "model 4"

    auc(roc_response_4)

    ## Area under the curve: 0.9952

    print("model 5")

    ## [1] "model 5"

    auc(roc_response_5)

    ## Area under the curve: 1

Random forest seems to be the best if we are interested in maximizing
Accuracy. By nature, random forest works well with large data set by
increasing the maximal depth.
