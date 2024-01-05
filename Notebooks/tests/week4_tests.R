check_values <- function(problem){
  if (problem == 1){

    set.seed(12345) 
    n_sims_test = 1000
    max_sample_size_test = 100
    start_size_test = 10
    pthresh_test = .05
    step_sizes_test =  c(1, 5, 10, 20)

    stopping_results_test = c()

    for (step_size in step_sizes_test){
      n_sequence = seq(start_size_test, max_sample_size_test, step_size)
      for (i in 1:n_sims_test){
        simdata = rnorm(mean=0, sd=1, n=max_sample_size_test)
        sigp = FALSE
        for (n in n_sequence){
          ttest_result = t.test(simdata[1:n], alternative='greater')
          if (ttest_result$p.value <= pthresh){
            sigp = TRUE
            break
          }
        }
        stopping_results_test = rbind(stopping_results_test, c(step_size, sigp))
      }
    }

    stopping_results_df_test = data.frame(stopping_results_test)
    names(stopping_results_df_test) = c('step_size', 'sigp')
    assert_that(all.equal(stopping_results_df, stopping_results_df_test))


  } else if (problem == 2){
    stopping_summary_df_test = stopping_results_df %>%
      group_by(step_size) %>%
      summarize(error_rate=mean(sigp))
    assert_that(all.equal(stopping_summary_df_test, stopping_summary_df))

    p1_test = ggplot(stopping_summary_df_test, aes(x=step_size, y=error_rate)) +
      geom_line() +
      ylim(0, .3) +
      geom_hline(yintercept = pthresh, linetype='dashed') +
      ylab('Type I error rate') +
      xlab('Step size')
    assert_that(all.equal(p1_test$data, p1$data))

  } else if (problem == 3){
    set.seed(12345)
    NHANES_adult_shuffled_test = NHANES_adult %>%
      mutate(Height_shuf = sample(Height))
    assert_that(all.equal(NHANES_adult_shuffled, NHANES_adult_shuffled_test))

    lm_summary_test = summary(lm(Height_shuf ~ PhysActive, data=NHANES_adult_shuffled))
    assert_that(all.equal(lm_summary_test$coefficients, lm_summary$coefficients))

  } else if (problem == 4){
    set.seed(123456)
    n_sims_test = 1000
    sample_size_test = 50

    sim_results_test = c()

    for (i in 1:n_sims_test){
      NHANES_sample_test = NHANES_adult %>%
        sample_n(sample_size_test) %>%
        ## create a shuffled version of the Height variable that will be our null outcome
        mutate(Height_shuf = sample(Height))
      lm_summary_test = summary(lm(Height_shuf ~ PhysActive, data=NHANES_sample_test))
      sim_results_test = c(sim_results_test, lm_summary_test$coefficients[2, 4] <= pthresh)
    }
    assert_that(all.equal(sim_results, sim_results_test))

  } else if (problem == 5){
    type_1_error_test = mean(sim_results)
    assert_that(type_1_error == type_1_error_test)
    
  } else if (problem==6){
    set.seed(123456)
    n_sims = 1000

    covariates_to_try = c('HomeRooms', 'Weight', 'BPDia1', 'AlcoholYear')
    # remove any participants with missing values on the covariates
    NHANES_adult_test = NHANES_adult %>%
      drop_na(any_of(covariates_to_try))

    sim_results_test = c()

    for (i in 1:n_sims){
      NHANES_sample_test = NHANES_adult_test %>%
        sample_n(sample_size) %>%
        mutate(Height_shuf = sample(Height))
      sigp = FALSE
      for (cv in covariates_to_try){
        cov_samp = NHANES_sample_test[, c('Height_shuf', 'PhysActive',  cv)]
        # use interaction model (denoted by .^2)
        lm_summary = summary(lm(Height_shuf ~ .^2, data=cov_samp))
        if (lm_summary$coefficients[2, 4] <= pthresh) {
          sigp=TRUE
          break
        }
      }
      sim_results_test = c(sim_results_test, sigp)

    }
    type_1_error_covariates_test = mean(sim_results_test)
    assert_that(type_1_error_covariates == type_1_error_covariates_test)

  }
  print('good job!')
}